import AppKit
import Foundation

private enum IconError: Error, CustomStringConvertible {
    case usage
    case bitmapCreation(width: Int, height: Int)
    case encoding(String)
    case iconConversion(String)

    var description: String {
        switch self {
        case .usage:
            return "usage: make-icon <menu-icon.tiff> [app-icon.icns] [preview.png]"
        case let .bitmapCreation(width, height):
            return "could not create \(width)x\(height) bitmap"
        case let .encoding(kind):
            return "could not encode \(kind) image"
        case let .iconConversion(message):
            return "app icon conversion failed: \(message)"
        }
    }
}

private func bitmap(
    width: Int,
    height: Int,
    logicalSize: NSSize,
    draw: (NSRect) -> Void
) throws -> NSBitmapImageRep {
    guard let representation = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: width,
        pixelsHigh: height,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else {
        throw IconError.bitmapCreation(width: width, height: height)
    }

    representation.size = logicalSize
    NSGraphicsContext.saveGraphicsState()
    guard let context = NSGraphicsContext(bitmapImageRep: representation) else {
        NSGraphicsContext.restoreGraphicsState()
        throw IconError.bitmapCreation(width: width, height: height)
    }
    NSGraphicsContext.current = context
    context.imageInterpolation = .high
    NSColor.clear.setFill()
    NSRect(origin: .zero, size: logicalSize).fill()
    draw(NSRect(origin: .zero, size: logicalSize))
    context.flushGraphics()
    NSGraphicsContext.restoreGraphicsState()
    return representation
}

private func centeredText(_ text: String, in rect: NSRect, attributes: [NSAttributedString.Key: Any]) {
    let attributed = NSAttributedString(string: text, attributes: attributes)
    let size = attributed.size()
    attributed.draw(at: NSPoint(
        x: rect.midX - size.width / 2,
        y: rect.midY - size.height / 2
    ))
}

private func drawMenuIcon(in bounds: NSRect) {
    centeredText("han", in: bounds.offsetBy(dx: 0, dy: 0.15), attributes: [
        .font: NSFont.systemFont(ofSize: 11, weight: .medium),
        .foregroundColor: NSColor.black,
    ])
}

private final class MenuIconView: NSView {
    override func draw(_ dirtyRect: NSRect) { drawMenuIcon(in: bounds) }
}

private func makeMenuIcon(at outputURL: URL) throws {
    // macOS reuses this transparent glyph in the menu and cursor indicator.
    let logicalSize = NSSize(width: 22, height: 16)
    let image = NSImage(size: logicalSize)

    for scale in [1, 2] {
        let representation = try bitmap(
            width: Int(logicalSize.width) * scale,
            height: Int(logicalSize.height) * scale,
            logicalSize: logicalSize
        ) { bounds in
            drawMenuIcon(in: bounds)
        }
        image.addRepresentation(representation)
    }

    guard let data = image.tiffRepresentation else {
        throw IconError.encoding("TIFF")
    }
    try data.write(to: outputURL, options: .atomic)
    let view = MenuIconView(frame: NSRect(origin: .zero, size: logicalSize))
    let glyph = view.dataWithPDF(inside: view.bounds)
    // Keep the filenames already registered by 0.2.3. Both resources contain
    // the same borderless glyph, including on systems that reuse the menu icon.
    for name in ["HanglishMenu.pdf", "HanglishCursor@2x.pdf"] {
        try glyph.write(
            to: outputURL.deletingLastPathComponent().appendingPathComponent(name),
            options: .atomic)
    }
}

private func makeAppIconRepresentation(pixels: Int) throws -> NSBitmapImageRep {
    let size = CGFloat(pixels)
    return try bitmap(
        width: pixels,
        height: pixels,
        logicalSize: NSSize(width: size, height: size)
    ) { bounds in
        let outerRect = bounds.insetBy(dx: size * 0.06, dy: size * 0.06)
        let outer = NSBezierPath(
            roundedRect: outerRect,
            xRadius: size * 0.22,
            yRadius: size * 0.22
        )

        NSGraphicsContext.saveGraphicsState()
        let shadow = NSShadow()
        shadow.shadowColor = NSColor.black.withAlphaComponent(0.24)
        shadow.shadowBlurRadius = max(1, size * 0.035)
        shadow.shadowOffset = NSSize(width: 0, height: -size * 0.025)
        shadow.set()
        NSGradient(
            starting: NSColor(calibratedRed: 0.18, green: 0.25, blue: 0.42, alpha: 1),
            ending: NSColor(calibratedRed: 0.07, green: 0.10, blue: 0.20, alpha: 1)
        )!.draw(in: outer, angle: -90)
        NSGraphicsContext.restoreGraphicsState()

        let keyRect = outerRect.insetBy(dx: size * 0.115, dy: size * 0.20)
            .offsetBy(dx: 0, dy: size * 0.015)
        let keycap = NSBezierPath(
            roundedRect: keyRect,
            xRadius: size * 0.095,
            yRadius: size * 0.095
        )
        NSColor.white.withAlphaComponent(0.96).setFill()
        keycap.fill()

        centeredText("han", in: keyRect.offsetBy(dx: 0, dy: size * 0.004), attributes: [
            .font: NSFont.monospacedSystemFont(ofSize: size * 0.205, weight: .bold),
            .foregroundColor: NSColor(calibratedRed: 0.08, green: 0.12, blue: 0.23, alpha: 1),
            .kern: -size * 0.009,
        ])
    }
}

private func writePNG(_ representation: NSBitmapImageRep, to outputURL: URL) throws {
    guard let data = representation.representation(using: .png, properties: [:]) else {
        throw IconError.encoding("PNG")
    }
    try data.write(to: outputURL, options: .atomic)
}

private func makeAppIcon(at outputURL: URL, previewURL: URL?) throws {
    let temporaryURL = FileManager.default.temporaryDirectory
        .appendingPathComponent("hanglish-icon-\(UUID().uuidString).png")
    let sourceURL = previewURL ?? temporaryURL
    defer {
        if previewURL == nil {
            try? FileManager.default.removeItem(at: temporaryURL)
        }
    }
    try writePNG(try makeAppIconRepresentation(pixels: 512), to: sourceURL)

    let process = Process()
    let errorPipe = Pipe()
    let outputPipe = Pipe()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/sips")
    process.arguments = ["--setProperty", "format", "icns", sourceURL.path, "--out", outputURL.path]
    process.standardError = errorPipe
    process.standardOutput = outputPipe
    try process.run()
    process.waitUntilExit()
    guard process.terminationStatus == 0 else {
        let data = errorPipe.fileHandleForReading.readDataToEndOfFile()
        let message = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        throw IconError.iconConversion(message ?? "exit status \(process.terminationStatus)")
    }
}

do {
    guard (2...4).contains(CommandLine.arguments.count) else {
        throw IconError.usage
    }

    let menuURL = URL(fileURLWithPath: CommandLine.arguments[1])
    let appIconURL: URL
    if CommandLine.arguments.count >= 3 {
        appIconURL = URL(fileURLWithPath: CommandLine.arguments[2])
    } else {
        appIconURL = menuURL.deletingLastPathComponent().appendingPathComponent("Hanglish.icns")
    }
    let previewURL = CommandLine.arguments.count == 4
        ? URL(fileURLWithPath: CommandLine.arguments[3])
        : nil

    try FileManager.default.createDirectory(
        at: menuURL.deletingLastPathComponent(),
        withIntermediateDirectories: true
    )
    try makeMenuIcon(at: menuURL)
    try makeAppIcon(at: appIconURL, previewURL: previewURL)
} catch {
    FileHandle.standardError.write(Data("make-icon: \(error)\n".utf8))
    exit(1)
}
