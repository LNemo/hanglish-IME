import Foundation

struct CompositionState {
    static let maximumInputLength = 2_000
    static let maximumCandidateCount = 5

    private(set) var raw = ""
    private(set) var candidates: [String] = []
    private(set) var selectedIndex = 0
    private(set) var revision = 0
    private(set) var pendingRevision: Int?
    private(set) var errorMessage: String?

    var displayText: String {
        guard candidates.indices.contains(selectedIndex) else {
            return raw
        }
        return candidates[selectedIndex]
    }

    var isEmpty: Bool {
        raw.isEmpty
    }

    @discardableResult
    mutating func append(_ string: String) -> Bool {
        guard !string.isEmpty,
              raw.count + string.count <= Self.maximumInputLength else {
            return false
        }

        raw.append(contentsOf: string)
        invalidateAfterEdit()
        return true
    }

    @discardableResult
    mutating func backspace() -> Bool {
        guard !raw.isEmpty else {
            return false
        }

        raw.removeLast()
        invalidateAfterEdit()
        return true
    }

    mutating func beginConversion() -> (revision: Int, input: String)? {
        guard !raw.isEmpty, pendingRevision == nil else {
            return nil
        }

        pendingRevision = revision
        errorMessage = nil
        return (revision: revision, input: raw)
    }

    @discardableResult
    mutating func accept(candidates responseCandidates: [String], revision responseRevision: Int) -> Bool {
        guard responseRevision == revision,
              pendingRevision == responseRevision else {
            return false
        }

        var uniqueCandidates: [String] = []
        var seen = Set<String>()
        for candidate in responseCandidates where !candidate.isEmpty {
            if seen.insert(candidate).inserted {
                uniqueCandidates.append(candidate)
            }
        }

        if !seen.contains(raw) {
            uniqueCandidates.append(raw)
        }

        if uniqueCandidates.count > Self.maximumCandidateCount {
            let prefix = Array(uniqueCandidates.prefix(Self.maximumCandidateCount))
            if prefix.contains(raw) {
                uniqueCandidates = prefix
            } else {
                uniqueCandidates = Array(prefix.dropLast()) + [raw]
            }
        }

        candidates = uniqueCandidates
        selectedIndex = 0
        pendingRevision = nil
        errorMessage = nil
        return true
    }

    @discardableResult
    mutating func fail(message: String, revision responseRevision: Int) -> Bool {
        guard responseRevision == revision,
              pendingRevision == responseRevision else {
            return false
        }

        candidates = []
        selectedIndex = 0
        pendingRevision = nil
        errorMessage = message.isEmpty ? "변환에 실패했습니다." : message
        return true
    }

    @discardableResult
    mutating func select(text: String) -> Bool {
        guard let index = candidates.firstIndex(of: text) else {
            return false
        }

        selectedIndex = index
        return true
    }

    @discardableResult
    mutating func cycle(delta: Int) -> Bool {
        guard !candidates.isEmpty else {
            return false
        }

        let offset = delta % candidates.count
        selectedIndex = (selectedIndex + offset + candidates.count) % candidates.count
        return true
    }

    mutating func finish(raw useRaw: Bool = false) -> String {
        let result = useRaw ? raw : displayText
        reset()
        return result
    }

    mutating func reset() {
        raw = ""
        candidates = []
        selectedIndex = 0
        pendingRevision = nil
        errorMessage = nil
        revision += 1
    }

    private mutating func invalidateAfterEdit() {
        candidates = []
        selectedIndex = 0
        pendingRevision = nil
        errorMessage = nil
        revision += 1
    }
}
