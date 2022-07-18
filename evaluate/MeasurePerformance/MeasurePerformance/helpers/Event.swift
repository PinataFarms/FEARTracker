//
//  Event.swift
//  MeasurePerformance
//
//  Created by Oleg Gordiichuk on 05.11.2021.
//

import Foundation

struct Event {
    // CPU utilization
    let cpu: Double?
    // memory used in MB
    let memory: Double?
    let batteryLevel: Float
    let thermalState: ProcessInfo.ThermalState
    let timestamp: Double
    let execution: Double

    static func statsTitleToLog() -> String {
        return "CPU;MEMORY;BATTERYLEVEL;THERMALSTATE;TIMESTAMP;EXECUTION\r\n"
    }

    func statsToLog() -> String {
        let cpuString = String(format: "%f", cpu ?? 0.0)
        let memoryString = String(format: "%f", memory ?? 0.0)
        let executionString = String(format: "%f", execution)
        return "\(cpuString);\(memoryString);\(batteryLevel);\(thermalStateString());\(Int(timestamp));\(executionString)\r\n"
    }

    private func thermalStateString() -> String {
        switch thermalState {
        case .nominal:
            return "low"
        case .fair:
            return "normal"
        case .serious:
            return "high"
        case .critical:
            return "critical"
        default:
            return "unknown"
        }
    }
}
