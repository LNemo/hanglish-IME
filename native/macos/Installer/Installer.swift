import AppKit
import Carbon
import Foundation

private let hanglishBundleIdentifier = "org.hanglish.inputmethod.Hanglish"
private let hanglishModeInputSourceIdentifier = "org.hanglish.inputmethod.Hanglish.Han"
private let abcInputSourceIdentifier = "com.apple.keylayout.ABC"
private let knownHanglishInputSourceIdentifiers = [
    hanglishModeInputSourceIdentifier,
    hanglishBundleIdentifier
]

enum InstallerError: LocalizedError {
    case message(String)

    var errorDescription: String? {
        switch self {
        case .message(let value): return value
        }
    }
}

struct InstallOutcome {
    let destination: URL
    let backup: URL?
    let enabled: Bool
    let switchedFromHanglish: Bool

    var userMessage: String {
        var lines = ["Hanglish 설치를 마쳤습니다."]
        if let backup { lines.append("이전 버전은 \(backup.path)에 보관했습니다.") }
        if enabled {
            lines.append("메뉴 막대의 입력기 메뉴에서 Hanglish를 선택하세요.")
        } else {
            lines.append("시스템 설정 → 키보드 → 텍스트 입력 → 편집에서 Hanglish를 추가하세요.")
        }
        if switchedFromHanglish {
            lines.append("업데이트를 안전하게 마치기 위해 입력기를 ABC로 전환했습니다.")
        }
        return lines.joined(separator: "\n")
    }
}

final class HanglishInstaller {
    typealias SignatureValidator = (URL) throws -> Void

    private let fileManager: FileManager
    private let signatureValidator: SignatureValidator

    init(fileManager: FileManager = .default, signatureValidator: SignatureValidator? = nil) {
        self.fileManager = fileManager
        self.signatureValidator = signatureValidator ?? Self.verifySignature
    }

    static var defaultDestination: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Input Methods/Hanglish.app", isDirectory: true)
    }

    static var defaultBackupRoot: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Hanglish/Backups", isDirectory: true)
    }

    static var bundledPayload: URL? {
        Bundle.main.resourceURL?.appendingPathComponent("Hanglish.app", isDirectory: true)
    }

    func install(
        payload: URL,
        destination: URL,
        backupRoot: URL,
        registerInputSource: Bool
    ) throws -> InstallOutcome {
        let payload = try validatedAbsoluteAppURL(payload, role: "설치 파일")
        let destination = try validatedAbsoluteAppURL(destination, role: "설치 위치", mayNotExist: true)
        let backupRoot = try validatedAbsoluteDirectoryURL(backupRoot, role: "백업 위치")

        try requireRegularBundle(payload, expectedIdentifier: hanglishBundleIdentifier, role: "설치 파일")
        let payloadSourceIdentifiers = try declaredInputSourceIdentifiers(in: payload)
        var sourceIdentifiersToReplace = Set(knownHanglishInputSourceIdentifiers)
        sourceIdentifiersToReplace.formUnion(payloadSourceIdentifiers)
        if fileManager.fileExists(atPath: destination.path) {
            try requireRegularBundle(destination, expectedIdentifier: hanglishBundleIdentifier, role: "기존 설치본")
            sourceIdentifiersToReplace.formUnion(try managedInputSourceIdentifiers(in: destination))
        }

        let parent = destination.deletingLastPathComponent()
        try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent(".Hanglish-staging-\(UUID().uuidString).app", isDirectory: true)
        defer { try? fileManager.removeItem(at: staging) }

        try fileManager.copyItem(at: payload, to: staging)
        try requireRegularBundle(staging, expectedIdentifier: hanglishBundleIdentifier, role: "복사된 설치 파일")
        try signatureValidator(staging)

        var switched = false
        if registerInputSource {
            switched = switchToABCIfHanglishIsCurrent(managedIdentifiers: sourceIdentifiersToReplace)
            try terminateRunningHanglish()
        }

        var backup: URL?
        if fileManager.fileExists(atPath: destination.path) {
            backup = try makeBackupURL(root: backupRoot)
            try fileManager.moveItem(at: destination, to: backup!)
        }

        do {
            try fileManager.moveItem(at: staging, to: destination)
        } catch {
            try restoreBackupIfPresent(backup, to: destination)
            throw InstallerError.message("새 버전을 설치하지 못했습니다. 이전 버전은 복구했습니다. (\(error.localizedDescription))")
        }

        guard registerInputSource else {
            return InstallOutcome(destination: destination, backup: backup, enabled: false, switchedFromHanglish: switched)
        }

        let registerStatus = TISRegisterInputSource(destination as CFURL)
        guard registerStatus == noErr else {
            do {
                try rollbackInstalledBundle(destination: destination, backup: backup)
            } catch {
                let backupDescription = backup?.path ?? "이전 설치본 없음"
                throw InstallerError.message(
                    "macOS 입력기 등록과 자동 복구에 실패했습니다. 백업: \(backupDescription). " +
                    "새 설치본을 변경하지 말고 백업을 확인하세요. (복구 오류: \(error.localizedDescription))"
                )
            }
            if let backup {
                let restoreStatus = TISRegisterInputSource(destination as CFURL)
                guard restoreStatus == noErr else {
                    throw InstallerError.message(
                        "새 버전 등록에 실패해 이전 앱 파일은 복구했지만 macOS에 다시 등록하지 못했습니다. " +
                        "복구된 앱: \(destination.path), 백업 위치: \(backup.path), 오류: \(restoreStatus)"
                    )
                }
                throw InstallerError.message(
                    "macOS에 새 입력기를 등록하지 못해 이전 버전으로 되돌렸습니다. " +
                    "백업 위치: \(backup.path), 오류: \(registerStatus)"
                )
            }
            throw InstallerError.message("macOS에 입력기를 등록하지 못해 새 설치본을 제거했습니다. (오류 \(registerStatus))")
        }

        let enabled = enablePreferredInputSource(identifiers: payloadSourceIdentifiers)
        return InstallOutcome(destination: destination, backup: backup, enabled: enabled, switchedFromHanglish: switched)
    }

    func uninstall(destination: URL, backupRoot: URL, manageInputSource: Bool = true) throws -> URL {
        let destination = try validatedAbsoluteAppURL(destination, role: "설치 위치")
        let backupRoot = try validatedAbsoluteDirectoryURL(backupRoot, role: "백업 위치")
        try requireRegularBundle(destination, expectedIdentifier: hanglishBundleIdentifier, role: "설치본")

        if manageInputSource {
            let managedIdentifiers = try managedInputSourceIdentifiers(in: destination)
            _ = switchToABCIfHanglishIsCurrent(managedIdentifiers: Set(managedIdentifiers))
            try terminateRunningHanglish()
            for identifier in managedIdentifiers {
                for source in inputSources(identifier: identifier) {
                    _ = TISDisableInputSource(source)
                }
            }
        }

        let backup = try makeBackupURL(root: backupRoot)
        try fileManager.moveItem(at: destination, to: backup)
        return backup
    }

    func inspect(destination: URL) throws -> [String: Any] {
        let destination = try validatedAbsoluteAppURL(destination, role: "확인 위치", mayNotExist: true)
        guard fileManager.fileExists(atPath: destination.path) else {
            return ["installed": false, "path": destination.path]
        }
        try requireRegularBundle(destination, expectedIdentifier: hanglishBundleIdentifier, role: "설치본")
        let plist = try bundleInfo(at: destination)
        return [
            "installed": true,
            "path": destination.path,
            "bundleIdentifier": plist["CFBundleIdentifier"] as? String ?? "",
            "version": plist["CFBundleShortVersionString"] as? String ?? "",
            "build": plist["CFBundleVersion"] as? String ?? "",
            "inputSourceIdentifiers": parseDeclaredInputSourceIdentifiers(in: plist),
            "inputSourceEnablementOrder": inputSourceEnablementPlan(
                declaredIdentifiers: parseDeclaredInputSourceIdentifiers(in: plist)
            ).order
        ]
    }

    private func validatedAbsoluteAppURL(_ url: URL, role: String, mayNotExist: Bool = false) throws -> URL {
        guard url.path.hasPrefix("/"), url.pathExtension == "app" else {
            throw InstallerError.message("\(role)은 절대 경로의 .app이어야 합니다.")
        }
        let standardized = url.standardizedFileURL
        if fileManager.fileExists(atPath: standardized.path) {
            try rejectSymbolicLink(standardized, role: role)
        } else if !mayNotExist {
            throw InstallerError.message("\(role)을 찾을 수 없습니다: \(standardized.path)")
        }
        return standardized
    }

    private func validatedAbsoluteDirectoryURL(_ url: URL, role: String) throws -> URL {
        guard url.path.hasPrefix("/") else {
            throw InstallerError.message("\(role)은 절대 경로여야 합니다.")
        }
        let standardized = url.standardizedFileURL
        if fileManager.fileExists(atPath: standardized.path) {
            try rejectSymbolicLink(standardized, role: role)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: standardized.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                throw InstallerError.message("\(role)이 폴더가 아닙니다.")
            }
        }
        return standardized
    }

    private func rejectSymbolicLink(_ url: URL, role: String) throws {
        let attributes = try fileManager.attributesOfItem(atPath: url.path)
        if attributes[.type] as? FileAttributeType == .typeSymbolicLink {
            throw InstallerError.message("\(role)이 심볼릭 링크여서 안전하게 처리할 수 없습니다: \(url.path)")
        }
    }

    private func requireRegularBundle(_ url: URL, expectedIdentifier: String, role: String) throws {
        try rejectSymbolicLink(url, role: role)
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw InstallerError.message("\(role)이 앱 번들이 아닙니다: \(url.path)")
        }
        let info = try bundleInfo(at: url)
        guard info["CFBundleIdentifier"] as? String == expectedIdentifier else {
            throw InstallerError.message("\(role)이 Hanglish 앱이 아니므로 변경하지 않았습니다.")
        }
    }

    private func bundleInfo(at url: URL) throws -> [String: Any] {
        let plistURL = url.appendingPathComponent("Contents/Info.plist")
        let data = try Data(contentsOf: plistURL)
        guard let value = try PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any] else {
            throw InstallerError.message("앱 정보 파일을 읽을 수 없습니다: \(plistURL.path)")
        }
        return value
    }

    private func managedInputSourceIdentifiers(in app: URL) throws -> [String] {
        let declared = try declaredInputSourceIdentifiers(in: app)
        return orderedUnique(declared + knownHanglishInputSourceIdentifiers)
    }

    private func declaredInputSourceIdentifiers(in app: URL) throws -> [String] {
        parseDeclaredInputSourceIdentifiers(in: try bundleInfo(at: app))
    }

    private func makeBackupURL(root: URL) throws -> URL {
        try fileManager.createDirectory(at: root, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let folder = root.appendingPathComponent("\(formatter.string(from: Date()))-\(UUID().uuidString.prefix(8))", isDirectory: true)
        try fileManager.createDirectory(at: folder, withIntermediateDirectories: false)
        return folder.appendingPathComponent("Hanglish.app", isDirectory: true)
    }

    private func restoreBackupIfPresent(_ backup: URL?, to destination: URL) throws {
        guard let backup, fileManager.fileExists(atPath: backup.path) else { return }
        if fileManager.fileExists(atPath: destination.path) { try fileManager.removeItem(at: destination) }
        try fileManager.moveItem(at: backup, to: destination)
    }

    private func rollbackInstalledBundle(destination: URL, backup: URL?) throws {
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try restoreBackupIfPresent(backup, to: destination)
    }

    private static func verifySignature(_ app: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/codesign")
        process.arguments = ["--verify", "--deep", "--strict", app.path]
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw InstallerError.message("설치 파일의 코드 서명을 확인하지 못했습니다.")
        }
    }

    private func terminateRunningHanglish() throws {
        let applications = NSRunningApplication.runningApplications(withBundleIdentifier: hanglishBundleIdentifier)
        for application in applications { _ = application.terminate() }
        let deadline = Date().addingTimeInterval(3)
        while Date() < deadline, applications.contains(where: { !$0.isTerminated }) {
            Thread.sleep(forTimeInterval: 0.1)
        }
        guard !applications.contains(where: { !$0.isTerminated }) else {
            throw InstallerError.message(
                "실행 중인 Hanglish 입력기를 종료하지 못해 설치 파일을 변경하지 않았습니다. " +
                "다른 입력기로 전환하고 Hanglish를 사용하는 앱을 닫은 뒤 다시 시도하세요."
            )
        }
    }
}

private func sourceProperty(_ source: TISInputSource, _ key: CFString) -> AnyObject? {
    guard let pointer = TISGetInputSourceProperty(source, key) else { return nil }
    return Unmanaged<AnyObject>.fromOpaque(pointer).takeUnretainedValue()
}

private func inputSources(identifier: String? = nil) -> [TISInputSource] {
    let filter = identifier.map { [kTISPropertyInputSourceID as String: $0] as CFDictionary }
    guard let unmanaged = TISCreateInputSourceList(filter, true) else { return [] }
    return unmanaged.takeRetainedValue() as? [TISInputSource] ?? []
}

private func inputSourceIdentifier(_ source: TISInputSource) -> String {
    sourceProperty(source, kTISPropertyInputSourceID) as? String ?? ""
}

private func isInputSourceEnabled(_ source: TISInputSource) -> Bool {
    sourceProperty(source, kTISPropertyInputSourceIsEnabled) as? Bool ?? false
}

private func orderedUnique(_ identifiers: [String]) -> [String] {
    var seen = Set<String>()
    return identifiers.filter { !$0.isEmpty && seen.insert($0).inserted }
}

private func parseDeclaredInputSourceIdentifiers(in info: [String: Any]) -> [String] {
    var identifiers: [String] = []
    if let modes = info["ComponentInputModeDict"] {
        identifiers.append(contentsOf: nestedInputSourceIdentifiers(in: modes))
    }
    if let identifier = info["TISInputSourceID"] as? String {
        identifiers.append(identifier)
    }
    return orderedUnique(identifiers)
}

private func nestedInputSourceIdentifiers(in value: Any) -> [String] {
    if let dictionary = value as? [String: Any] {
        var identifiers: [String] = []
        if let identifier = dictionary["TISInputSourceID"] as? String {
            identifiers.append(identifier)
        }
        for key in dictionary.keys.sorted() where key != "TISInputSourceID" {
            if let child = dictionary[key] {
                identifiers.append(contentsOf: nestedInputSourceIdentifiers(in: child))
            }
        }
        return identifiers
    }
    if let array = value as? [Any] {
        return array.flatMap(nestedInputSourceIdentifiers)
    }
    return []
}

private struct InputSourceEnablementPlan {
    let order: [String]
    let selectableIdentifiers: Set<String>
}

private func inputSourceEnablementPlan(declaredIdentifiers: [String]) -> InputSourceEnablementPlan {
    let identifiers = orderedUnique(declaredIdentifiers)
    let parent = identifiers.first { $0 == hanglishBundleIdentifier }
    let modes = identifiers.filter { $0 != hanglishBundleIdentifier }
    let order = orderedUnique((parent.map { [$0] } ?? []) + modes)
    let selectable = modes.isEmpty ? Set(parent.map { [$0] } ?? []) : Set(modes)
    return InputSourceEnablementPlan(order: order, selectableIdentifiers: selectable)
}

private func enablePreferredInputSource(identifiers: [String]) -> Bool {
    let plan = inputSourceEnablementPlan(declaredIdentifiers: identifiers)
    var enabledIdentifiers = Set<String>()
    for identifier in plan.order {
        for source in inputSources(identifier: identifier) {
            if isInputSourceEnabled(source) ||
                (TISEnableInputSource(source) == noErr && isInputSourceEnabled(source)) {
                enabledIdentifiers.insert(identifier)
                break
            }
        }
    }
    let selectableSourceIsEnabled = !plan.selectableIdentifiers.isDisjoint(with: enabledIdentifiers)
    let parentIsEnabled = !plan.order.contains(hanglishBundleIdentifier) ||
        enabledIdentifiers.contains(hanglishBundleIdentifier)
    return selectableSourceIsEnabled && parentIsEnabled
}

@discardableResult
private func switchToABCIfHanglishIsCurrent(managedIdentifiers: Set<String>) -> Bool {
    let current = TISCopyCurrentKeyboardInputSource().takeRetainedValue()
    guard managedIdentifiers.contains(inputSourceIdentifier(current)),
          let abc = inputSources(identifier: abcInputSourceIdentifier).first,
          TISSelectInputSource(abc) == noErr else { return false }
    return true
}

final class InstallerWindowController: NSWindowController {
    private let statusLabel = NSTextField(wrappingLabelWithString: "Hanglish와 변환 엔진을 이 Mac에 설치합니다.")
    private let installButton = NSButton(title: "설치 / 업데이트", target: nil, action: nil)
    private let uninstallButton = NSButton(title: "제거 후 보관", target: nil, action: nil)
    private let guideButton = NSButton(title: "입력기 설정 안내", target: nil, action: nil)
    private let progress = NSProgressIndicator()
    private let installer = HanglishInstaller()

    init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 540, height: 270),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Hanglish 설치"
        window.center()
        super.init(window: window)
        configureUI()
    }

    required init?(coder: NSCoder) { nil }

    private func configureUI() {
        guard let content = window?.contentView else { return }
        let title = NSTextField(labelWithString: "Hanglish 입력기")
        title.font = .systemFont(ofSize: 24, weight: .semibold)
        let detail = NSTextField(wrappingLabelWithString: "입력기와 로컬 변환 엔진·모델을 함께 설치합니다. 관리자 암호는 필요하지 않습니다.")
        detail.textColor = .secondaryLabelColor

        installButton.target = self
        installButton.action = #selector(installPressed)
        installButton.keyEquivalent = "\r"
        uninstallButton.target = self
        uninstallButton.action = #selector(uninstallPressed)
        guideButton.target = self
        guideButton.action = #selector(guidePressed)
        progress.style = .spinning
        progress.controlSize = .small
        progress.isDisplayedWhenStopped = false

        let buttons = NSStackView(views: [installButton, uninstallButton, guideButton])
        buttons.orientation = .horizontal
        buttons.spacing = 10
        let statusRow = NSStackView(views: [progress, statusLabel])
        statusRow.orientation = .horizontal
        statusRow.alignment = .top
        statusRow.spacing = 8
        let stack = NSStackView(views: [title, detail, buttons, statusRow])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 16
        stack.translatesAutoresizingMaskIntoConstraints = false
        detail.maximumNumberOfLines = 2
        statusLabel.maximumNumberOfLines = 4
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 28),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -28),
            stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 28),
            statusLabel.widthAnchor.constraint(lessThanOrEqualToConstant: 460)
        ])
    }

    @objc private func installPressed() {
        guard let payload = HanglishInstaller.bundledPayload else {
            statusLabel.stringValue = "설치 패키지 안에서 Hanglish.app을 찾지 못했습니다."
            return
        }
        runTask(initialStatus: "설치 파일을 확인하고 있습니다…") { [installer] in
            try installer.install(
                payload: payload,
                destination: HanglishInstaller.defaultDestination,
                backupRoot: HanglishInstaller.defaultBackupRoot,
                registerInputSource: true
            ).userMessage
        }
    }

    @objc private func uninstallPressed() {
        let alert = NSAlert()
        alert.messageText = "Hanglish를 제거할까요?"
        alert.informativeText = "설치본은 삭제하지 않고 백업 폴더로 옮깁니다."
        alert.addButton(withTitle: "제거 후 보관")
        alert.addButton(withTitle: "취소")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        runTask(initialStatus: "Hanglish를 안전하게 보관하고 있습니다…") { [installer] in
            let backup = try installer.uninstall(
                destination: HanglishInstaller.defaultDestination,
                backupRoot: HanglishInstaller.defaultBackupRoot
            )
            return "Hanglish를 제거했습니다. 설치본은 \(backup.path)에 보관했습니다."
        }
    }

    @objc private func guidePressed() {
        statusLabel.stringValue = "시스템 설정 → 키보드 → 텍스트 입력 → 편집 → + → 한국어에서 Hanglish를 추가한 뒤, 메뉴 막대의 입력기 메뉴에서 선택하세요."
    }

    private func runTask(initialStatus: String, operation: @escaping () throws -> String) {
        installButton.isEnabled = false
        uninstallButton.isEnabled = false
        progress.startAnimation(nil)
        statusLabel.stringValue = initialStatus
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            let result: Result<String, Error> = Result { try operation() }
            DispatchQueue.main.async {
                guard let self else { return }
                self.progress.stopAnimation(nil)
                self.installButton.isEnabled = true
                self.uninstallButton.isEnabled = true
                switch result {
                case .success(let message): self.statusLabel.stringValue = message
                case .failure(let error): self.statusLabel.stringValue = error.localizedDescription
                }
            }
        }
    }
}

final class InstallerAppDelegate: NSObject, NSApplicationDelegate {
    private var windowController: InstallerWindowController?

    func applicationDidFinishLaunching(_ notification: Notification) {
        let controller = InstallerWindowController()
        windowController = controller
        controller.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

private func writeError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

private func argumentValue(_ name: String, in arguments: [String]) throws -> String? {
    guard let index = arguments.firstIndex(of: name) else { return nil }
    guard index + 1 < arguments.count else { throw InstallerError.message("\(name) 뒤에 경로가 필요합니다.") }
    return arguments[index + 1]
}

private func runCommandLine(_ arguments: [String]) -> Int32 {
    do {
        let installer = HanglishInstaller()
        let destinationPath = try argumentValue("--destination", in: arguments)
        let backupPath = try argumentValue("--backup-root", in: arguments)
        let destination = destinationPath.map { URL(fileURLWithPath: $0) } ?? HanglishInstaller.defaultDestination
        let backupRoot = backupPath.map { URL(fileURLWithPath: $0) } ?? HanglishInstaller.defaultBackupRoot

        if arguments.contains("--inspect") {
            let info = try installer.inspect(destination: destination)
            let data = try JSONSerialization.data(withJSONObject: info, options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: data, as: UTF8.self))
            return 0
        }
        if arguments.contains("--install") {
            guard let payload = HanglishInstaller.bundledPayload else {
                throw InstallerError.message("설치 패키지 안에서 Hanglish.app을 찾지 못했습니다.")
            }
            let outcome = try installer.install(
                payload: payload,
                destination: destination,
                backupRoot: backupRoot,
                registerInputSource: !arguments.contains("--no-register")
            )
            print(outcome.userMessage)
            return 0
        }
        if arguments.contains("--uninstall") {
            let backup = try installer.uninstall(
                destination: destination,
                backupRoot: backupRoot,
                manageInputSource: !arguments.contains("--no-register")
            )
            print("Hanglish를 제거하고 \(backup.path)에 보관했습니다.")
            return 0
        }
        writeError("사용법: Hanglish Installer --install|--inspect|--uninstall [--destination /absolute/Hanglish.app] [--backup-root /absolute/folder] [--no-register]")
        return 2
    } catch {
        writeError(error.localizedDescription)
        return 1
    }
}

let installerArguments = Array(CommandLine.arguments.dropFirst())
if installerArguments.contains("--install") || installerArguments.contains("--inspect") || installerArguments.contains("--uninstall") {
    exit(runCommandLine(installerArguments))
}

let application = NSApplication.shared
let delegate = InstallerAppDelegate()
application.delegate = delegate
application.setActivationPolicy(.regular)
application.run()
