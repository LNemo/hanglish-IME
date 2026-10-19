import Foundation

@main
enum LocalizationTests {
    static func require(_ condition: @autoclosure () -> Bool, _ message: String) {
        guard condition() else {
            FileHandle.standardError.write(Data(("Localization test failed: \(message)\n").utf8))
            exit(1)
        }
    }

    static func main() {
        guard CommandLine.arguments.count == 3 else {
            FileHandle.standardError.write(Data("Usage: LocalizationTests installer-resources runtime-resources\n".utf8))
            exit(2)
        }

        require(HanglishLocalization.resolveLanguage(["ko-KR"]) == "ko", "ko-KR should use Korean")
        require(HanglishLocalization.resolveLanguage(["en-GB"]) == "en", "en-GB should use English")
        require(HanglishLocalization.resolveLanguage(["fr-FR", "ko"]) == "ko", "the first supported preference should win")
        require(HanglishLocalization.resolveLanguage(["fr-FR"]) == "en", "unsupported locales should fall back to English")

        let installerResources = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        requireMatchingKeys(in: installerResources, label: "installer")
        let installerEnglish = HanglishLocalization(resourceDirectory: installerResources, preferredLanguages: ["fr-FR"])
        let installerKorean = HanglishLocalization(resourceDirectory: installerResources, preferredLanguages: ["ko-KR"])
        require(
            installerEnglish.text("Install / Update") == "Install / Update",
            "installer should default to English"
        )
        require(
            installerKorean.text("Install / Update") == "설치 / 업데이트",
            "installer should load Korean resources"
        )
        require(
            installerKorean.text("The previous version was saved at %@.", "/tmp/backup") ==
                "이전 버전은 /tmp/backup에 보관했습니다.",
            "localized placeholders should be interpolated"
        )

        let runtimeResources = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
        requireMatchingKeys(in: runtimeResources, label: "runtime")
        let runtimeEnglish = HanglishLocalization(resourceDirectory: runtimeResources, preferredLanguages: ["ja-JP"])
        let runtimeKorean = HanglishLocalization(resourceDirectory: runtimeResources, preferredLanguages: ["ko"])
        let engineError = "The built-in conversion engine could not be started. Press Esc to keep the original text."
        require(runtimeEnglish.text(engineError) == engineError, "runtime should fall back to English")
        require(
            runtimeKorean.text(engineError) == "내장 변환 엔진을 시작할 수 없습니다. Esc로 원문을 확정할 수 있습니다.",
            "runtime should load Korean resources"
        )

        print("Hanglish localization tests: passed")
    }

    static func requireMatchingKeys(in resources: URL, label: String) {
        func keys(for language: String) -> Set<String> {
            let url = resources
                .appendingPathComponent("\(language).lproj", isDirectory: true)
                .appendingPathComponent("Localizable.strings")
            guard let data = try? Data(contentsOf: url),
                  let strings = try? PropertyListSerialization.propertyList(from: data, format: nil),
                  let dictionary = strings as? [String: String] else {
                require(false, "\(label) \(language) strings should be readable")
                return []
            }
            return Set(dictionary.keys)
        }

        let english = keys(for: "en")
        let korean = keys(for: "ko")
        require(!english.isEmpty, "\(label) strings should not be empty")
        require(english == korean, "\(label) English and Korean keys should match")
    }
}
