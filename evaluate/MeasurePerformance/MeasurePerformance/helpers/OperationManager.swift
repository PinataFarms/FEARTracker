//
//  OperationManager.swift
//  MeasurePerformance
//
//  Created by Vasyl Borsuk on 08.11.2021.
//

import Foundation
import CoreML

class OperationsManager {
    func launchOnline(duration: Double, fps: Double = 30, function: @escaping () -> Void, completed: @escaping (String?) -> Void) {
        let logger = Logger()
        let lock = NSLock()
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInteractive

        let producerTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / fps, repeats: true, block: { _ in
            queue.addOperation {
                let start = CFAbsoluteTimeGetCurrent()
                function()
                let executionTime = CFAbsoluteTimeGetCurrent() - start

                lock.lock()
                defer { lock.unlock() }
                logger.log(execution: executionTime)
            }
        })
        Timer.scheduledTimer(withTimeInterval: duration, repeats: false, block: { _ in
            producerTimer.invalidate()
            queue.cancelAllOperations()
            completed(logger.logFilePath)
        })
    }

    func launchOffline(duration: Double, fps: Double = 30, function: @escaping () -> Void, completed: @escaping (String?) -> Void) {
        let logger = Logger()
        let lock = NSLock()
        let queue = OperationQueue()
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInteractive

        let frameCount = Int(duration * fps)
        for _ in 0..<frameCount {
            queue.addOperation {
                let start = CFAbsoluteTimeGetCurrent()
                function()
                let executionTime = CFAbsoluteTimeGetCurrent() - start

                lock.lock()
                defer { lock.unlock() }
                logger.log(execution: executionTime)
                if logger.getThermalState() == .serious {
                    queue.cancelAllOperations()
                }
            }
        }
        queue.addOperation {
            completed(logger.logFilePath)
        }
    }
}
