import Cocoa
import InputMethodKit

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
guard let identifier = Bundle.main.bundleIdentifier,
      let connection = Bundle.main.object(forInfoDictionaryKey: "InputMethodConnectionName") as? String,
      let inputServer = IMKServer(name: connection, bundleIdentifier: identifier) else {
    fputs("Hanglish must run from its input-method app bundle.\n", stderr)
    exit(1)
}
withExtendedLifetime(inputServer) { app.run() }
