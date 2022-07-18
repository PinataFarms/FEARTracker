//
//  Logger.swift
//  MeasurePerformance
//
//  Created by Oleg Gordiichuk on 05.11.2021.
//

import Foundation
import SwiftUI

class Logger {

    private var previousCpuInfo: host_cpu_load_info?

    public var logFilePath: String?

    public func log(execution: Double) {
        let cpuLoad = reportCPU()
        let memoryLoad = reportMemory()
        let batteryLevel = reportBatteryLevel()
        let thermalState = getThermalState()
        let timestamp = Date().timeIntervalSince1970
        let event = Event(cpu: cpuLoad, memory: memoryLoad, batteryLevel: batteryLevel, thermalState: thermalState, timestamp: timestamp, execution: execution)
        writeLogToFile(event: event)
    }

    private func reportCPU() -> Double? {
        let cpuInfo = getHostCPULoadInfo()
        var userUsage: Double?
        if let cpuInfo = cpuInfo, let previousCpuInfo = previousCpuInfo {
            let userDiff = Double(cpuInfo.cpu_ticks.0 - previousCpuInfo.cpu_ticks.0)
            let sysDiff  = Double(cpuInfo.cpu_ticks.1 - previousCpuInfo.cpu_ticks.1)
            let idleDiff = Double(cpuInfo.cpu_ticks.2 - previousCpuInfo.cpu_ticks.2)
            let niceDiff = Double(cpuInfo.cpu_ticks.3 - previousCpuInfo.cpu_ticks.3)
            let totalTicks = sysDiff + userDiff + niceDiff + idleDiff
            if totalTicks != .zero {
                userUsage = userDiff / totalTicks
            }
        }
        previousCpuInfo = cpuInfo
        return userUsage
    }

    private func reportMemory() -> Double? {
        let taskInfo = getTaskInfo()
        if let taskInfo = taskInfo {
            let memoryMb = Double(taskInfo.resident_size) / 1024 / 1024
            return memoryMb
        }
        else {
            return nil
        }
    }

    private func reportBatteryLevel() -> Float {
        return UIDevice.current.batteryLevel
    }

    func getThermalState() -> ProcessInfo.ThermalState {
        return ProcessInfo.processInfo.thermalState
    }

    private func getHostCPULoadInfo() -> host_cpu_load_info? {
        let HOST_CPU_LOAD_INFO_COUNT = MemoryLayout<host_cpu_load_info>.stride / MemoryLayout<integer_t>.stride
        var size = mach_msg_type_number_t(HOST_CPU_LOAD_INFO_COUNT)
        var cpuLoadInfo = host_cpu_load_info()

        let result = withUnsafeMutablePointer(to: &cpuLoadInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: HOST_CPU_LOAD_INFO_COUNT) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &size)
            }
        }
        if result != KERN_SUCCESS {
            print("Error  - \(#file): \(#function) - kern_result_t = \(result)")
            return nil
        }
        return cpuLoadInfo
    }

    private func getTaskInfo() -> mach_task_basic_info? {
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        var taskInfo = mach_task_basic_info()

        let result = withUnsafeMutablePointer(to: &taskInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if result != KERN_SUCCESS {
            print("Error  - \(#file): \(#function) - kern_result_t = \(result)")
            return nil
        }
        return taskInfo
    }

    private func createLogFile() {
        let fileManager = FileManager.default
        do {
            let formatter = DateFormatter()
            formatter.dateFormat = "dd-MM-yyyy-HH:mm:ss"
            let dateString = formatter.string(from: Date())
            let fileName = "\(dateString).csv"
            let path = try fileManager.url(for: .documentDirectory, in: .allDomainsMask, appropriateFor: nil, create: false).appendingPathComponent(fileName).path
            if (FileManager.default.createFile(atPath: path, contents: nil, attributes: nil)) {
                logFilePath = path
                try Event.statsTitleToLog().write(to: URL(fileURLWithPath: path), atomically: false, encoding: .utf8)
            } else {
                print("File not created.")
            }
        } catch {
            print(error)
        }
    }

    private func writeLogToFile(event: Event) {
        guard let path = logFilePath else {
            createLogFile()
            writeLogToFile(event: event)
            return
        }

        let fileURLWithPath = URL(fileURLWithPath: path)
        let data = event.statsToLog().data(using: .utf8) ?? Data()

        if FileManager.default.fileExists(atPath: fileURLWithPath.path) {
           if let fileHandle = try? FileHandle(forWritingTo: fileURLWithPath) {
               fileHandle.seekToEndOfFile()
               fileHandle.write(data)
               fileHandle.closeFile()
           }
       } else {
           try? data.write(to: fileURLWithPath, options: .atomicWrite)
       }
    }
}
