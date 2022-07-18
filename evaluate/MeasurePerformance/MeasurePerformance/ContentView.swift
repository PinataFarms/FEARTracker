//
//  ContentView.swift
//  MeasurePerformance
//
//  Created by Vasyl Borsuk on 23.12.2020.
//

import SwiftUI
import CoreML
import Foundation

struct ContentView: View {
    private let testImageName: String = "test_image1.png"
    private let files: [URL]
    private let image: UIImage
    private let device = UIDevice.current

    @State var isHideLoader: Bool = true

    init() {
        files = Bundle.main.urls(forResourcesWithExtension: "mlmodelc", subdirectory: nil)!
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        image = UIImage(named: testImageName)!
        device.isBatteryMonitoringEnabled = true
    }
    
    var body: some View {
        VStack {
            LoaderView(tintColor: .blue, scaleSize: 3.0).padding(.bottom,50).hidden(isHideLoader)
            Button(action: {
                self.isHideLoader = !self.isHideLoader
                for modelPath in files {
                    let avgTime = measureFps(modelPath, numWarmup: 20, numRuns: 100)
                    let avgTimeStr = String(format: "%.4f", avgTime)
                    let avgFPSStr = String(format: "%.4f", 1 / avgTime)
                    print("\(modelPath.deletingPathExtension().lastPathComponent); \(avgTimeStr) ms; \(avgFPSStr) FPS")
                }
                print()
                self.isHideLoader = !self.isHideLoader
            }, label: {
                Text("Benchmark FPS")
            }).padding(.bottom, 50)
            Button(action: {
                self.isHideLoader = !self.isHideLoader
                let modelPath = files[0]
                measureBatteryOnline(modelPath, for: 30 * 60, fps: 30)
            }, label: {
                Text("Benchmark Online")
            }).padding(.bottom, 50)
            Button(action: {
                self.isHideLoader = !self.isHideLoader
                let modelPath = files[0]
                measureBatteryOffline(modelPath, for: 5 * 60, fps: 30)
            }, label: {
                Text("Benchmark Offline")
            }).padding(.bottom, 50)
        }
    }
    
    func measureFps(_ modelPath: URL, numWarmup: Int, numRuns: Int) -> Double {
        let model = loadModel(modelPath)
        let featureProvider = FeatureProvider(image: image, for: model)!

        let benchmark = FPSBenchmark(numWarmup: numWarmup, numRuns: numRuns)
        let avgTime = benchmark.launch {
            let _ = try! model.prediction(from: featureProvider)
        }
        return avgTime
    }

    func measureBatteryOnline(_ modelPath: URL, for duration: Double, fps: Double = 30) {
        let model = loadModel(modelPath)
        let featureProvider = FeatureProvider(image: image, for: model)!

        let manager = OperationsManager()
        manager.launchOnline(duration: duration, fps: fps, function: {
            let _ = try! model.prediction(from: featureProvider)
        }, completed: { path in
            actionSheet(path: path)
            self.isHideLoader = !self.isHideLoader
        })
    }

    func measureBatteryOffline(_ modelPath: URL, for duration: Double, fps: Double = 30) {
        let model = loadModel(modelPath)
        let featureProvider = FeatureProvider(image: image, for: model)!

        let manager = OperationsManager()
        manager.launchOffline(duration: duration, fps: fps, function: {
            let _ = try! model.prediction(from: featureProvider)
        }, completed: { path in
            actionSheet(path: path)
            self.isHideLoader = !self.isHideLoader
        })
    }

    private func loadModel(_ modelPath: URL, computeUnits: MLComputeUnits = .all) -> MLModel {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let model = try! MLModel(contentsOf: modelPath, configuration: configuration)
        return model
    }

    private func actionSheet(path: String?) {
        guard let path = path else { return }
        let urlShare = URL(fileURLWithPath: path)
        let activityVC = UIActivityViewController(activityItems: [urlShare], applicationActivities: nil)
        UIApplication.shared.windows.first?.rootViewController?.present(activityVC, animated: true, completion: nil)
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}

struct LoaderView: View {
    var tintColor: Color = .blue
    var scaleSize: CGFloat = 1.0

    var body: some View {
        ProgressView()
            .scaleEffect(scaleSize, anchor: .center)
            .progressViewStyle(CircularProgressViewStyle(tint: tintColor))
    }
}

extension View {
    @ViewBuilder func hidden(_ shouldHide: Bool) -> some View {
        switch shouldHide {
        case true: self.hidden()
        case false: self
        }
    }
}
