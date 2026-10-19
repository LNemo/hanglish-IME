import Foundation

@main
struct CompositionStateTests {
    private static var failures: [String] = []

    static func main() {
        testInitialStateAndEmptyOperations()
        testEnglishAndSpacesArePreserved()
        testUnicodeBackspaceRemovesOneGrapheme()
        testInputLengthCap()
        testConversionAndCandidateSelection()
        testCandidateNormalizationAndRawReserve()
        testEditingAfterPreviewUsesOriginalRawInput()
        testEditInvalidatesPreviewAndPendingResponse()
        testFinishInvalidatesPendingResponse()
        testSameInputAtLaterRevisionRejectsOldResponse()
        testFailureKeepsRawAndCanRetry()
        testRawFinishEscapesPreview()

        if failures.isEmpty {
            print("CompositionStateTests: all tests passed")
        } else {
            for failure in failures {
                print("FAIL: \(failure)")
            }
            fatalError("CompositionStateTests: \(failures.count) failure(s)")
        }
    }

    private static func testInitialStateAndEmptyOperations() {
        var state = CompositionState()
        expect(state.isEmpty, "initial state is empty")
        expectEqual(state.displayText, "", "initial display text")
        expectEqual(state.revision, 0, "initial revision")
        expect(!state.append(""), "empty append is ignored")
        expect(!state.backspace(), "backspace on empty state is ignored")
        expect(state.beginConversion() == nil, "empty input cannot begin conversion")
        expectEqual(state.revision, 0, "ignored operations do not change revision")
    }

    private static func testEnglishAndSpacesArePreserved() {
        var state = CompositionState()
        expect(state.append("testreul sihaenghanda"), "English sentence append succeeds")
        expectEqual(state.raw, "testreul sihaenghanda", "spaces remain in raw input")
        expectEqual(state.displayText, "testreul sihaenghanda", "raw text is displayed before conversion")
    }

    private static func testUnicodeBackspaceRemovesOneGrapheme() {
        var state = CompositionState()
        expect(state.append("a👨‍👩‍👧‍👦"), "Unicode append succeeds")
        let revisionAfterAppend = state.revision
        expect(state.backspace(), "Unicode backspace succeeds")
        expectEqual(state.raw, "a", "backspace removes one extended grapheme")
        expectEqual(state.revision, revisionAfterAppend + 1, "backspace increments revision")
    }

    private static func testInputLengthCap() {
        var state = CompositionState()
        let maximum = String(repeating: "a", count: CompositionState.maximumInputLength)
        expect(state.append(maximum), "input at cap succeeds")
        let revisionAtCap = state.revision
        expect(!state.append("b"), "input beyond cap is rejected")
        expectEqual(state.raw.count, CompositionState.maximumInputLength, "rejected append leaves input unchanged")
        expectEqual(state.revision, revisionAtCap, "rejected append leaves revision unchanged")
    }

    private static func testConversionAndCandidateSelection() {
        var state = CompositionState()
        _ = state.append("imnida")
        guard let request = state.beginConversion() else {
            fail("conversion request should be created")
            return
        }
        expect(state.beginConversion() == nil, "a pending revision cannot be requested twice")
        expect(state.accept(candidates: ["입니다", "임니다"], revision: request.revision), "current response is accepted")
        expectEqual(state.candidates, ["입니다", "임니다", "imnida"], "raw fallback is appended")
        expectEqual(state.displayText, "입니다", "first candidate is initially selected")
        expect(state.cycle(delta: 1), "forward candidate cycle succeeds")
        expectEqual(state.displayText, "임니다", "forward cycle changes selection")
        expect(state.cycle(delta: -1), "backward candidate cycle succeeds")
        expectEqual(state.displayText, "입니다", "backward cycle wraps correctly")
        expect(state.select(text: "imnida"), "candidate can be selected by text")
        expectEqual(state.displayText, "imnida", "selected text is displayed")
        expect(!state.select(text: "missing"), "unknown candidate cannot be selected")
    }

    private static func testCandidateNormalizationAndRawReserve() {
        var state = CompositionState()
        _ = state.append("roman")
        let request = state.beginConversion()!
        let response = ["", "하나", "하나", "둘", "셋", "넷", "다섯", "roman"]
        expect(state.accept(candidates: response, revision: request.revision), "normalized response is accepted")
        expectEqual(state.candidates, ["하나", "둘", "셋", "넷", "roman"], "candidates are nonempty, distinct, capped, and reserve raw")
        expectEqual(Set(state.candidates).count, state.candidates.count, "normalized candidates are distinct")
    }

    private static func testEditingAfterPreviewUsesOriginalRawInput() {
        var state = CompositionState()
        _ = state.append("imnida")
        let request = state.beginConversion()!
        _ = state.accept(candidates: ["입니다"], revision: request.revision)
        expectEqual(state.displayText, "입니다", "accepted candidate is previewed")

        _ = state.append("x")
        expectEqual(state.raw, "imnidax", "preview edit extends original raw input")
        expectEqual(state.displayText, "imnidax", "preview edit returns display to raw input")
        expect(state.candidates.isEmpty, "preview edit clears prior candidates")
    }

    private static func testEditInvalidatesPreviewAndPendingResponse() {
        var state = CompositionState()
        _ = state.append("imnida")
        let oldRequest = state.beginConversion()!
        _ = state.append("x")
        expectEqual(state.raw, "imnidax", "edit applies to original raw text")
        expect(state.candidates.isEmpty, "edit clears preview candidates")
        expect(state.pendingRevision == nil, "edit cancels pending request")
        expect(!state.accept(candidates: ["입니다"], revision: oldRequest.revision), "response from before edit is stale")
        expectEqual(state.displayText, "imnidax", "stale response cannot replace raw display")
    }

    private static func testFinishInvalidatesPendingResponse() {
        var state = CompositionState()
        _ = state.append("imnida")
        let request = state.beginConversion()!
        let revisionBeforeFinish = state.revision
        expectEqual(state.finish(), "imnida", "finish without candidates commits raw")
        expect(state.isEmpty, "finish clears composition")
        expectEqual(state.revision, revisionBeforeFinish + 1, "finish increments revision")
        expect(!state.accept(candidates: ["입니다"], revision: request.revision), "response after finish is stale")
    }

    private static func testSameInputAtLaterRevisionRejectsOldResponse() {
        var state = CompositionState()
        _ = state.append("abc")
        let firstRequest = state.beginConversion()!
        state.reset()
        _ = state.append("abc")
        let secondRequest = state.beginConversion()!
        expect(firstRequest.revision != secondRequest.revision, "same text at a later time has a new revision")
        expect(!state.accept(candidates: ["old"], revision: firstRequest.revision), "old same-input response is rejected")
        expect(state.accept(candidates: ["new"], revision: secondRequest.revision), "current same-input response is accepted")
        expectEqual(state.displayText, "new", "current response is displayed")
    }

    private static func testFailureKeepsRawAndCanRetry() {
        var state = CompositionState()
        _ = state.append("imnida")
        let request = state.beginConversion()!
        expect(state.fail(message: "network unavailable", revision: request.revision), "current failure is accepted")
        expectEqual(state.raw, "imnida", "failure keeps raw input")
        expect(state.candidates.isEmpty, "failure clears candidates")
        expectEqual(state.errorMessage, "network unavailable", "failure exposes status message")
        expect(state.pendingRevision == nil, "failure clears pending state")
        expect(state.beginConversion() != nil, "failed conversion can be retried")
    }

    private static func testRawFinishEscapesPreview() {
        var state = CompositionState()
        _ = state.append("imnida")
        let request = state.beginConversion()!
        _ = state.accept(candidates: ["입니다"], revision: request.revision)
        expectEqual(state.finish(raw: true), "imnida", "raw finish bypasses selected candidate")
        expect(state.isEmpty, "raw finish clears state")
    }

    private static func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
        if !condition() {
            failures.append(message)
        }
    }

    private static func expectEqual<T: Equatable>(_ actual: T, _ expected: T, _ message: String) {
        if actual != expected {
            failures.append("\(message): expected \(expected), got \(actual)")
        }
    }

    private static func fail(_ message: String) {
        failures.append(message)
    }
}
