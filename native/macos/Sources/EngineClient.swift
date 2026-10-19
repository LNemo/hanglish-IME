import Foundation
import Darwin

struct ConversionReply: Decodable {
    struct Candidate: Decodable { let text: String }
    let result: String
    let candidates: [Candidate]
    let version: String
    let request_id: String
}

enum EngineError: Error, LocalizedError, Equatable {
    case unavailable, invalidResponse, busy, timedOut

    var errorDescription: String? {
        switch self {
        case .unavailable:
            return localized("The built-in conversion engine could not be started. Press Esc to keep the original text.")
        case .invalidResponse:
            return localized("The conversion response was invalid. Your original text is preserved.")
        case .busy:
            return localized("The conversion engine is busy. Try again shortly or press Esc to keep the original text.")
        case .timedOut:
            return localized("The conversion timed out. Try again or press Esc to keep the original text.")
        }
    }
}

final class EngineRequest {
    private let lock = NSLock()
    private var cancellation: (() -> Void)?

    fileprivate init(cancellation: @escaping () -> Void) {
        self.cancellation = cancellation
    }

    func cancel() {
        let action: (() -> Void)?
        lock.lock()
        action = cancellation
        cancellation = nil
        lock.unlock()
        action?()
    }
}

final class EngineClient {
    private let ownerID = UUID()
    private let manager: EngineProcessManager
    private let lock = NSLock()
    private var requests: [UUID: EngineRequest] = [:]
    private var closed = false

    convenience init() {
        self.init(manager: .shared)
    }

    convenience init(resourcesURL: URL) {
        self.init(manager: EngineProcessManager(resourcesURL: resourcesURL))
    }

    convenience init(
        executableURL: URL,
        arguments: [String] = [],
        currentDirectoryURL: URL? = nil,
        requestTimeout: TimeInterval = 60,
        idleTimeout: TimeInterval = 120,
        maxPending: Int = 8,
        maxFrameBytes: Int = 1_048_576
    ) {
        self.init(manager: EngineProcessManager(
            executableURL: executableURL,
            arguments: arguments,
            currentDirectoryURL: currentDirectoryURL,
            requestTimeout: requestTimeout,
            idleTimeout: idleTimeout,
            maxPending: maxPending,
            maxFrameBytes: maxFrameBytes
        ))
    }

    private init(manager: EngineProcessManager) {
        self.manager = manager
    }

    func close() {
        let active: [EngineRequest]
        lock.lock()
        closed = true
        active = Array(requests.values)
        requests.removeAll()
        lock.unlock()
        active.forEach { $0.cancel() }
    }

    @discardableResult
    func convert(
        _ input: String,
        requestID: String,
        completion: @escaping (Result<ConversionReply, Error>) -> Void
    ) -> EngineRequest {
        let token = UUID()
        let request = EngineRequest { [weak self, manager, ownerID] in
            self?.forget(token: token)
            manager.cancel(token: token, ownerID: ownerID)
        }

        lock.lock()
        if closed {
            lock.unlock()
            DispatchQueue.main.async { completion(.failure(EngineError.unavailable)) }
            return request
        }
        requests[token] = request
        lock.unlock()

        manager.enqueue(
            input: input,
            requestID: requestID,
            token: token,
            ownerID: ownerID
        ) { [weak self] result in
            guard let self else { return }
            self.lock.lock()
            let stillOwned = self.requests.removeValue(forKey: token) != nil && !self.closed
            self.lock.unlock()
            guard stillOwned else { return }
            completion(result)
        }
        return request
    }

    private func forget(token: UUID) {
        lock.lock()
        requests.removeValue(forKey: token)
        lock.unlock()
    }
}

private final class EngineProcessManager {
    private final class Job {
        let input: String
        let requestID: String
        let token: UUID
        let ownerID: UUID
        let completion: (Result<ConversionReply, Error>) -> Void
        var cancelled = false

        init(input: String, requestID: String, token: UUID, ownerID: UUID,
             completion: @escaping (Result<ConversionReply, Error>) -> Void) {
            self.input = input
            self.requestID = requestID
            self.token = token
            self.ownerID = ownerID
            self.completion = completion
        }
    }

    static let shared = EngineProcessManager(resourcesURL: Bundle.main.resourceURL)

    private let queue = DispatchQueue(label: "org.hanglish.engine-process")
    private let executableURL: URL?
    private let arguments: [String]
    private let currentDirectoryURL: URL?
    private let requestTimeout: TimeInterval
    private let idleTimeout: TimeInterval
    private let maxPending: Int
    private let maxFrameBytes: Int
    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stdoutBuffer = Data()
    private var queued: [Job] = []
    private var current: Job?
    private var timeoutGeneration = 0
    private var idleGeneration = 0
    private var intentionallyStopping = false

    init(resourcesURL: URL?) {
        let python = resourcesURL?.appendingPathComponent("Python/bin/python3.11")
        let engine = resourcesURL?.appendingPathComponent("Engine", isDirectory: true)
        let script = engine?.appendingPathComponent("engine_service.py")
        executableURL = python
        arguments = script.map { ["-I", "-B", $0.path] } ?? []
        currentDirectoryURL = engine
        requestTimeout = 60
        idleTimeout = 120
        maxPending = 8
        maxFrameBytes = 1_048_576
    }

    init(executableURL: URL, arguments: [String], currentDirectoryURL: URL?,
         requestTimeout: TimeInterval, idleTimeout: TimeInterval,
         maxPending: Int, maxFrameBytes: Int) {
        self.executableURL = executableURL
        self.arguments = arguments
        self.currentDirectoryURL = currentDirectoryURL ?? executableURL.deletingLastPathComponent()
        self.requestTimeout = requestTimeout
        self.idleTimeout = idleTimeout
        self.maxPending = max(1, maxPending)
        self.maxFrameBytes = max(1, maxFrameBytes)
    }

    deinit {
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stdinPipe?.fileHandleForWriting.closeFile()
        if let process, process.isRunning { process.terminate() }
    }

    func enqueue(input: String, requestID: String, token: UUID, ownerID: UUID,
                 completion: @escaping (Result<ConversionReply, Error>) -> Void) {
        let job = Job(input: input, requestID: requestID, token: token,
                      ownerID: ownerID, completion: completion)
        queue.async { [weak self] in
            guard let self else { return }
            guard self.pendingCount < self.maxPending else {
                self.finish(job, with: .failure(EngineError.busy))
                return
            }
            self.queued.append(job)
            self.idleGeneration += 1
            self.startNextIfPossible()
        }
    }

    func cancel(token: UUID, ownerID: UUID) {
        queue.async { [weak self] in
            guard let self else { return }
            if let index = self.queued.firstIndex(where: {
                $0.token == token && $0.ownerID == ownerID
            }) {
                self.queued.remove(at: index).cancelled = true
                if self.current == nil { self.scheduleIdleStop() }
                return
            }
            if self.current?.token == token && self.current?.ownerID == ownerID {
                self.current?.cancelled = true
            }
        }
    }

    private var pendingCount: Int { queued.count + (current == nil ? 0 : 1) }

    private func startNextIfPossible() {
        guard current == nil else { return }
        while !queued.isEmpty, queued[0].cancelled { queued.removeFirst() }
        guard !queued.isEmpty else {
            scheduleIdleStop()
            return
        }
        guard ensureProcess() else {
            failAll(with: .unavailable)
            return
        }
        let job = queued.removeFirst()
        current = job
        let payload: [String: Any] = [
            "input": job.input, "context": "", "engine": "neural",
            "mode": "batch", "raw": false, "request_id": job.requestID,
        ]
        guard JSONSerialization.isValidJSONObject(payload),
              var data = try? JSONSerialization.data(withJSONObject: payload) else {
            finishCurrent(with: .failure(EngineError.invalidResponse))
            return
        }
        data.append(0x0a)
        do {
            try stdinPipe?.fileHandleForWriting.write(contentsOf: data)
        } catch {
            handleProcessFailure(.unavailable)
            return
        }
        timeoutGeneration += 1
        let generation = timeoutGeneration
        queue.asyncAfter(deadline: .now() + requestTimeout) { [weak self] in
            guard let self, self.timeoutGeneration == generation, self.current === job else { return }
            self.failAllAndStop(with: .timedOut)
        }
    }

    private func ensureProcess() -> Bool {
        if process?.isRunning == true { return true }
        guard let executableURL,
              FileManager.default.isExecutableFile(atPath: executableURL.path),
              let currentDirectoryURL,
              FileManager.default.fileExists(atPath: currentDirectoryURL.path) else { return false }
        let child = Process()
        let input = Pipe()
        let output = Pipe()
        child.executableURL = executableURL
        child.arguments = arguments
        child.currentDirectoryURL = currentDirectoryURL
        child.environment = [
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": FileManager.default.homeDirectoryForCurrentUser.path,
        ]
        child.standardInput = input
        child.standardOutput = output
        child.standardError = FileHandle.nullDevice
        intentionallyStopping = false
        stdoutBuffer.removeAll(keepingCapacity: true)
        output.fileHandleForReading.readabilityHandler = { [weak self, weak child] handle in
            let data = handle.availableData
            self?.queue.async { [weak self, weak child] in
                guard let self, self.process === child else { return }
                if data.isEmpty { self.handleProcessFailure(.unavailable) }
                else { self.consume(data) }
            }
        }
        child.terminationHandler = { [weak self, weak child] _ in
            self?.queue.async { [weak self, weak child] in
                guard let self, self.process === child else { return }
                if !self.intentionallyStopping { self.handleProcessFailure(.unavailable) }
                else { self.clearProcess() }
            }
        }
        do {
            try child.run()
        } catch {
            output.fileHandleForReading.readabilityHandler = nil
            return false
        }
        process = child
        stdinPipe = input
        stdoutPipe = output
        return true
    }

    private func consume(_ data: Data) {
        stdoutBuffer.append(data)
        if stdoutBuffer.count > maxFrameBytes && !stdoutBuffer.contains(0x0a) {
            failAllAndStop(with: .invalidResponse)
            return
        }
        while let newline = stdoutBuffer.firstIndex(of: 0x0a) {
            let frame = stdoutBuffer[..<newline]
            stdoutBuffer.removeSubrange(...newline)
            guard frame.count <= maxFrameBytes, !frame.isEmpty else {
                failAllAndStop(with: .invalidResponse)
                return
            }
            consumeFrame(Data(frame))
            if process == nil { return }
        }
    }

    private func consumeFrame(_ data: Data) {
        guard let job = current,
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let responseID = object["request_id"] as? String,
              responseID == job.requestID else {
            failAllAndStop(with: .invalidResponse)
            return
        }
        if object["error"] is String {
            finishCurrent(with: .failure(EngineError.unavailable))
            return
        }
        guard let reply = try? JSONDecoder().decode(ConversionReply.self, from: data),
              !reply.result.isEmpty, !reply.version.isEmpty,
              !reply.candidates.isEmpty, reply.candidates.count <= 20,
              reply.candidates.first?.text == reply.result,
              reply.candidates.allSatisfy({ !$0.text.isEmpty && $0.text.count <= 8000 }) else {
            failAllAndStop(with: .invalidResponse)
            return
        }
        finishCurrent(with: .success(reply))
    }

    private func finishCurrent(with result: Result<ConversionReply, Error>) {
        guard let job = current else { return }
        current = nil
        timeoutGeneration += 1
        if !job.cancelled { finish(job, with: result) }
        startNextIfPossible()
    }

    private func finish(_ job: Job, with result: Result<ConversionReply, Error>) {
        DispatchQueue.main.async { job.completion(result) }
    }

    private func handleProcessFailure(_ error: EngineError) {
        guard process != nil else { return }
        clearProcess()
        failAll(with: error)
    }

    private func failAllAndStop(with error: EngineError) {
        stopProcess(gracePeriod: 0)
        failAll(with: error)
    }

    private func failAll(with error: EngineError) {
        timeoutGeneration += 1
        var jobs = queued
        queued.removeAll()
        if let current { jobs.insert(current, at: 0) }
        current = nil
        for job in jobs where !job.cancelled { finish(job, with: .failure(error)) }
        scheduleIdleStop()
    }

    private func scheduleIdleStop() {
        guard current == nil, queued.isEmpty, process != nil else { return }
        idleGeneration += 1
        let generation = idleGeneration
        queue.asyncAfter(deadline: .now() + idleTimeout) { [weak self] in
            guard let self, self.idleGeneration == generation,
                  self.current == nil, self.queued.isEmpty else { return }
            self.stopProcess(gracePeriod: 0.5)
        }
    }

    private func stopProcess(gracePeriod: TimeInterval) {
        guard let child = process else {
            clearProcess()
            return
        }
        intentionallyStopping = true
        stdinPipe?.fileHandleForWriting.closeFile()
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        child.terminationHandler = nil
        clearProcess()

        DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + gracePeriod) {
            guard child.isRunning else { return }
            child.terminate()
            DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + 1) {
                if child.isRunning { _ = Darwin.kill(child.processIdentifier, SIGKILL) }
            }
        }
    }

    private func clearProcess() {
        stdoutPipe?.fileHandleForReading.readabilityHandler = nil
        stdinPipe?.fileHandleForWriting.closeFile()
        process?.terminationHandler = nil
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        stdoutBuffer.removeAll(keepingCapacity: true)
        intentionallyStopping = false
    }
}
