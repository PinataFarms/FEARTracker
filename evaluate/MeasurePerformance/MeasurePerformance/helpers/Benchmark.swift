//
//  Benchmark.swift
//  MeasurePerformance
//
//  Created by Vasyl Borsuk on 06.11.2021.
//

import Foundation
import QuartzCore
import CoreML

class BatteryBenchmark {
    let duration: Double
    let fps: Double

    init(duration: Double, fps: Double = 30) {
        self.duration = duration
        self.fps = fps
    }

    func launch_online(function: @escaping () -> Void, completed: @escaping (String?) -> Void) {
        let logger = Logger()

        let maxTime = duration
        let timeInterval = 1.0 / fps
        var totalTime: Double = 0
        while totalTime < maxTime {
            let start = CFAbsoluteTimeGetCurrent()
            function()
            let executionTime = CFAbsoluteTimeGetCurrent() - start
            logger.log(execution: executionTime)
            if executionTime < timeInterval {
                Thread.sleep(forTimeInterval: timeInterval - executionTime)
                totalTime += timeInterval
            } else {
                totalTime += executionTime
            }
        }
        completed(logger.logFilePath)
    }

    func launch_offline(function: @escaping () -> Void, completed: @escaping (String?) -> Void) {
        let logger = Logger()

        let frameCount = Int(duration * fps)
        for _ in 0..<frameCount {
            let start = CFAbsoluteTimeGetCurrent()
            function()
            let executionTime = CFAbsoluteTimeGetCurrent() - start
            logger.log(execution: executionTime)
        }
        completed(logger.logFilePath)
    }
}

class FPSBenchmark {
    private let numWarmup: Int
    private let numRuns: Int

    init(numWarmup: Int = 20, numRuns: Int = 100) {
        self.numWarmup = numWarmup
        self.numRuns = numRuns
    }

    func launch(function: @escaping () -> Void) -> Double {
        for _ in 0..<numWarmup {
            function()
        }

        let startTime = CACurrentMediaTime()
        for _ in 0..<numRuns {
            function()
        }
        let elapsed = CACurrentMediaTime() - startTime
        let avgTime = elapsed / Double(numRuns)
        return avgTime
    }
}
