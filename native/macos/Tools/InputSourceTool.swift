import Foundation
import Carbon

func property(_ source: TISInputSource, _ key: CFString) -> AnyObject? {
    guard let pointer = TISGetInputSourceProperty(source, key) else { return nil }
    return Unmanaged<AnyObject>.fromOpaque(pointer).takeUnretainedValue()
}
func sources(_ id: String? = nil) -> [TISInputSource] {
    let filter = id.map { [kTISPropertyInputSourceID as String: $0] as CFDictionary }
    return TISCreateInputSourceList(filter, true).takeRetainedValue() as! [TISInputSource]
}
func check(_ status: OSStatus) {
    guard status == noErr else { fputs("Text Input Sources error: \(status)\n", stderr); exit(1) }
}
let args = CommandLine.arguments
guard args.count >= 2 else { fputs("Usage: input-source status | register PATH | enable ID | select ID | disable ID\n", stderr); exit(2) }
switch args[1] {
case "status":
    let current = TISCopyCurrentKeyboardInputSource().takeRetainedValue()
    let items = sources().filter { (property($0, kTISPropertyInputSourceID) as? String ?? "").contains("hanglish") }.map {
        ["id": property($0, kTISPropertyInputSourceID) as? String ?? "",
         "enabled": property($0, kTISPropertyInputSourceIsEnabled) as? Bool ?? false,
         "enableCapable": property($0, kTISPropertyInputSourceIsEnableCapable) as? Bool ?? false,
         "selectCapable": property($0, kTISPropertyInputSourceIsSelectCapable) as? Bool ?? false,
         "type": property($0, kTISPropertyInputSourceType) as? String ?? "",
         "name": property($0, kTISPropertyLocalizedName) as? String ?? "",
         "modeID": property($0, kTISPropertyInputModeID) as? String ?? "",
         "iconURL": (property($0, kTISPropertyIconImageURL) as? URL)?.path ?? "",
         "selected": property($0, kTISPropertyInputSourceIsSelected) as? Bool ?? false] as [String: Any]
    }
    let data = try JSONSerialization.data(withJSONObject: ["current": property(current, kTISPropertyInputSourceID) as? String ?? "", "hanglish": items], options: [.prettyPrinted, .sortedKeys])
    print(String(decoding: data, as: UTF8.self))
case "register":
    guard args.count == 3 else { exit(2) }
    check(TISRegisterInputSource(URL(fileURLWithPath: args[2]) as CFURL))
case "enable", "select", "disable":
    guard args.count == 3, let source = sources(args[2]).first else {
        fputs("Input source is not registered in this login session. Log out/in or add it in Keyboard settings.\n", stderr); exit(1)
    }
    if args[1] == "enable" { check(TISEnableInputSource(source)) }
    else if args[1] == "disable" { check(TISDisableInputSource(source)) }
    else { check(TISSelectInputSource(source)) }
    if args[1] == "enable", !(property(source, kTISPropertyInputSourceIsEnabled) as? Bool ?? false) {
        fputs("Bundle registered, but macOS has not enabled it. Add Hanglish in System Settings > Keyboard > Text Input > Edit. A new login session may be required.\n", stderr)
        exit(3)
    }
default: exit(2)
}
