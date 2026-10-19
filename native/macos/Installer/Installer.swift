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
        var lines = [localized("Hanglish installation is complete.")]
        if let backup { lines.append(localized("The previous version was saved at %@.", backup.path)) }
        if enabled {
            lines.append(localized("Select Hanglish from the input menu in the menu bar."))
        } else {
            lines.append(localized("Add Hanglish in System Settings → Keyboard → Text Input → Edit."))
        }
        if switchedFromHanglish {
            lines.append(localized("The input source was switched to ABC to complete the update safely."))
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
        let payload = try validatedAbsoluteAppURL(payload, role: localized("installation package"))
        let destination = try validatedAbsoluteAppURL(destination, role: localized("installation destination"), mayNotExist: true)
        let backupRoot = try validatedAbsoluteDirectoryURL(backupRoot, role: localized("backup destination"))

        try requireRegularBundle(payload, expectedIdentifier: hanglishBundleIdentifier, role: localized("installation package"))
        let payloadSourceIdentifiers = try declaredInputSourceIdentifiers(in: payload)
        var sourceIdentifiersToReplace = Set(knownHanglishInputSourceIdentifiers)
        sourceIdentifiersToReplace.formUnion(payloadSourceIdentifiers)
        if fileManager.fileExists(atPath: destination.path) {
            try requireRegularBundle(destination, expectedIdentifier: hanglishBundleIdentifier, role: localized("existing installation"))
            sourceIdentifiersToReplace.formUnion(try managedInputSourceIdentifiers(in: destination))
        }

        let parent = destination.deletingLastPathComponent()
        try fileManager.createDirectory(at: parent, withIntermediateDirectories: true)
        let staging = parent.appendingPathComponent(".Hanglish-staging-\(UUID().uuidString).app", isDirectory: true)
        defer { try? fileManager.removeItem(at: staging) }

        try fileManager.copyItem(at: payload, to: staging)
        try requireRegularBundle(staging, expectedIdentifier: hanglishBundleIdentifier, role: localized("copied installation package"))
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
            throw InstallerError.message(localized(
                "The new version could not be installed. The previous version was restored. (%@)",
                error.localizedDescription
            ))
        }

        guard registerInputSource else {
            return InstallOutcome(destination: destination, backup: backup, enabled: false, switchedFromHanglish: switched)
        }

        let registerStatus = TISRegisterInputSource(destination as CFURL)
        guard registerStatus == noErr else {
            do {
                try rollbackInstalledBundle(destination: destination, backup: backup)
            } catch {
                let backupDescription = backup?.path ?? localized("No previous installation")
                throw InstallerError.message(localized(
                    "Registering the macOS input source and automatic recovery both failed. Backup: %@. Do not modify the new installation; check the backup. (Recovery error: %@)",
                    backupDescription,
                    error.localizedDescription
                ))
            }
            if let backup {
                let restoreStatus = TISRegisterInputSource(destination as CFURL)
                guard restoreStatus == noErr else {
                    throw InstallerError.message(localized(
                        "The new version could not be registered. The previous app was restored, but macOS could not register it again. Restored app: %@, backup: %@, error: %d",
                        destination.path,
                        backup.path,
                        restoreStatus
                    ))
                }
                throw InstallerError.message(localized(
                    "The new input source could not be registered with macOS, so the previous version was restored. Backup: %@, error: %d",
                    backup.path,
                    registerStatus
                ))
            }
            throw InstallerError.message(localized(
                "The input source could not be registered with macOS, so the new installation was removed. (Error %d)",
                registerStatus
            ))
        }

        let enabled = enablePreferredInputSource(identifiers: payloadSourceIdentifiers)
        return InstallOutcome(destination: destination, backup: backup, enabled: enabled, switchedFromHanglish: switched)
    }

    func uninstall(destination: URL, backupRoot: URL, manageInputSource: Bool = true) throws -> URL {
        let destination = try validatedAbsoluteAppURL(destination, role: localized("installation destination"))
        let backupRoot = try validatedAbsoluteDirectoryURL(backupRoot, role: localized("backup destination"))
        try requireRegularBundle(destination, expectedIdentifier: hanglishBundleIdentifier, role: localized("installed app"))

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
        let destination = try validatedAbsoluteAppURL(destination, role: localized("inspection destination"), mayNotExist: true)
        guard fileManager.fileExists(atPath: destination.path) else {
            return ["installed": false, "path": destination.path]
        }
        try requireRegularBundle(destination, expectedIdentifier: hanglishBundleIdentifier, role: localized("installed app"))
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
            throw InstallerError.message(localized("%@ must be an absolute path to an .app bundle.", role))
        }
        let standardized = url.standardizedFileURL
        if fileManager.fileExists(atPath: standardized.path) {
            try rejectSymbolicLink(standardized, role: role)
        } else if !mayNotExist {
            throw InstallerError.message(localized("%@ could not be found: %@", role, standardized.path))
        }
        return standardized
    }

    private func validatedAbsoluteDirectoryURL(_ url: URL, role: String) throws -> URL {
        guard url.path.hasPrefix("/") else {
            throw InstallerError.message(localized("%@ must be an absolute path.", role))
        }
        let standardized = url.standardizedFileURL
        if fileManager.fileExists(atPath: standardized.path) {
            try rejectSymbolicLink(standardized, role: role)
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: standardized.path, isDirectory: &isDirectory), isDirectory.boolValue else {
                throw InstallerError.message(localized("%@ is not a folder.", role))
            }
        }
        return standardized
    }

    private func rejectSymbolicLink(_ url: URL, role: String) throws {
        let attributes = try fileManager.attributesOfItem(atPath: url.path)
        if attributes[.type] as? FileAttributeType == .typeSymbolicLink {
            throw InstallerError.message(localized("%@ is a symbolic link and cannot be handled safely: %@", role, url.path))
        }
    }

    private func requireRegularBundle(_ url: URL, expectedIdentifier: String, role: String) throws {
        try rejectSymbolicLink(url, role: role)
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw InstallerError.message(localized("%@ is not an app bundle: %@", role, url.path))
        }
        let info = try bundleInfo(at: url)
        guard info["CFBundleIdentifier"] as? String == expectedIdentifier else {
            throw InstallerError.message(localized("%@ is not a Hanglish app, so it was left unchanged.", role))
        }
    }

    private func bundleInfo(at url: URL) throws -> [String: Any] {
        let plistURL = url.appendingPathComponent("Contents/Info.plist")
        let data = try Data(contentsOf: plistURL)
        guard let value = try PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any] else {
            throw InstallerError.message(localized("The app information file could not be read: %@", plistURL.path))
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
            throw InstallerError.message(localized("The installation package's code signature could not be verified."))
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
            throw InstallerError.message(localized(
                "The running Hanglish input source could not be closed, so the installation was left unchanged. Switch to another input source, close apps using Hanglish, and try again."
            ))
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
    private let statusLabel = NSTextField(wrappingLabelWithString: localized("Install Hanglish and its conversion engine on this Mac."))
    private let installButton = NSButton(title: localized("Install / Update"), target: nil, action: nil)
    private let uninstallButton = NSButton(title: localized("Remove and Save Backup"), target: nil, action: nil)
    private let guideButton = NSButton(title: localized("Input Source Setup Help"), target: nil, action: nil)
    private let progress = NSProgressIndicator()
    private let installer = HanglishInstaller()

    init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 540, height: 270),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        window.title = localized("Hanglish Installer")
        window.center()
        super.init(window: window)
        configureUI()
    }

    required init?(coder: NSCoder) { nil }

    private func configureUI() {
        guard let content = window?.contentView else { return }
        let title = NSTextField(labelWithString: localized("Hanglish Input Source"))
        title.font = .systemFont(ofSize: 24, weight: .semibold)
        let detail = NSTextField(wrappingLabelWithString: localized("Installs the input source with its local conversion engine and model. No administrator password is required."))
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
            statusLabel.stringValue = localized("Hanglish.app could not be found in the installation package.")
            return
        }
        runTask(initialStatus: localized("Checking the installation package…")) { [installer] in
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
        alert.messageText = localized("Remove Hanglish?")
        alert.informativeText = localized("The installed app will be moved to the backup folder instead of being deleted.")
        alert.addButton(withTitle: localized("Remove and Save Backup"))
        alert.addButton(withTitle: localized("Cancel"))
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        runTask(initialStatus: localized("Moving Hanglish to a safe backup…")) { [installer] in
            let backup = try installer.uninstall(
                destination: HanglishInstaller.defaultDestination,
                backupRoot: HanglishInstaller.defaultBackupRoot
            )
            return localized("Hanglish was removed. The installed app was saved at %@.", backup.path)
        }
    }

    @objc private func guidePressed() {
        statusLabel.stringValue = localized("Open System Settings → Keyboard → Text Input → Edit → +. Add Hanglish under Korean, then select it from the input menu in the menu bar.")
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
    guard index + 1 < arguments.count else {
        throw InstallerError.message(localized("%@ requires a path after it.", name))
    }
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
                throw InstallerError.message(localized("Hanglish.app could not be found in the installation package."))
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
            print(localized("Hanglish was removed and saved at %@.", backup.path))
            return 0
        }
        writeError(localized("Usage: Hanglish Installer --install|--inspect|--uninstall [--destination /absolute/Hanglish.app] [--backup-root /absolute/folder] [--no-register]"))
        return 2
    } catch {
        writeError(error.localizedDescription)
        return 1
    }
}

@main
enum InstallerMain {
    static func main() {
        let installerArguments = Array(CommandLine.arguments.dropFirst())
        if installerArguments.contains("--install") || installerArguments.contains("--inspect") || installerArguments.contains("--uninstall") {
            exit(runCommandLine(installerArguments))
        }

        let application = NSApplication.shared
        let delegate = InstallerAppDelegate()
        application.delegate = delegate
        application.setActivationPolicy(.regular)
        application.run()
    }
}
