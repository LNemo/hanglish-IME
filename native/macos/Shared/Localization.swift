import Foundation

struct HanglishLocalization {
    static let supportedLanguages = ["en", "ko"]

    let language: String
    private let stringsBundle: Bundle?

    init(
        bundle: Bundle = .main,
        preferredLanguages: [String] = Locale.preferredLanguages
    ) {
        language = Self.resolveLanguage(preferredLanguages)
        if let path = bundle.path(forResource: language, ofType: "lproj") {
            stringsBundle = Bundle(path: path)
        } else if language != "en",
                  let path = bundle.path(forResource: "en", ofType: "lproj") {
            stringsBundle = Bundle(path: path)
        } else {
            stringsBundle = nil
        }
    }

    init(resourceDirectory: URL, preferredLanguages: [String]) {
        language = Self.resolveLanguage(preferredLanguages)
        let preferred = resourceDirectory.appendingPathComponent("\(language).lproj", isDirectory: true)
        let english = resourceDirectory.appendingPathComponent("en.lproj", isDirectory: true)
        stringsBundle = Bundle(url: preferred) ?? Bundle(url: english)
    }

    static func resolveLanguage(_ preferredLanguages: [String]) -> String {
        for identifier in preferredLanguages {
            let language = Locale(identifier: identifier).language.languageCode?.identifier
                ?? identifier.split(separator: "-", maxSplits: 1).first.map(String.init)
                ?? identifier
            if supportedLanguages.contains(language) { return language }
        }
        return "en"
    }

    func text(_ key: String, _ arguments: CVarArg...) -> String {
        let value = stringsBundle?.localizedString(forKey: key, value: key, table: nil) ?? key
        guard !arguments.isEmpty else { return value }
        return String(format: value, locale: Locale(identifier: language), arguments: arguments)
    }
}

private let hanglishText = HanglishLocalization()

func localized(_ key: String, _ arguments: CVarArg...) -> String {
    let value = hanglishText.text(key)
    guard !arguments.isEmpty else { return value }
    return String(format: value, locale: Locale(identifier: hanglishText.language), arguments: arguments)
}
