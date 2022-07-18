//
//  VOTracker.swift
//  BreakfastFinder
//
//  Created by Vasyl Borsuk on 03.06.2022.
//  Copyright © 2022 Apple. All rights reserved.
//

import Foundation
import CoreML
import Accelerate

enum VOTrackerError: Error {
    case preprocessingError
    case uninitializedError
    case thresholdError
}

struct VOTrackerResult {
    let bbox: CGRect
    let confidence: Float
}

class VOTracker {

    private enum Constants {
        static let initSize: Int = 128
        static let trackSize: Int = 256
        static let scoreSize: Int = 16
        static let totalStride: Int = 16
        static let threshold: Float = 0.7
    }

    /// Feature extraction network
    private lazy var modelInit: TrackerInit = {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        return try! TrackerInit(configuration: config)
    }()

    /// Tracking network
    private lazy var model: Tracker = {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        return try! Tracker(configuration: config)
    }()

    private var templateFeatures: MLMultiArray? = nil

    public var isInitialized: Bool { templateFeatures != nil }

    public func initialize(image: CVPixelBuffer, rect: CGRect) throws {
        guard let input = resizePixelBuffer(image,
                                            cropX: Int(rect.minX),
                                            cropY: Int(rect.minY),
                                            cropWidth: Int(rect.width),
                                            cropHeight: Int(rect.height),
                                            scaleWidth: Constants.initSize,
                                            scaleHeight: Constants.initSize)
        else {
            throw VOTrackerError.preprocessingError
        }
        let result = try modelInit.prediction(image: input)
        templateFeatures = result.features
    }

    public func track(image: CVPixelBuffer) throws -> VOTrackerResult {
        guard let templateFeatures = templateFeatures else {
            throw VOTrackerError.uninitializedError
        }

        guard let input = resizePixelBuffer(image, width: Constants.trackSize, height: Constants.trackSize) else {
            throw VOTrackerError.preprocessingError
        }
        let output = try model.prediction(image: input, template_features: templateFeatures)
        let result = decode(output: output)
        if result.confidence < Constants.threshold {
            throw VOTrackerError.thresholdError
        }

        let scaleX = CGFloat(CVPixelBufferGetWidth(image)) / CGFloat(Constants.trackSize)
        let scaleY = CGFloat(CVPixelBufferGetHeight(image)) / CGFloat(Constants.trackSize)
        let rect = result.bbox.applying(
            CGAffineTransform(scaleX: scaleX, y: scaleY)
        )
        return VOTrackerResult(bbox: rect, confidence: result.confidence)
    }

    public func clear() {
        templateFeatures = nil
    }

    private func decode(output: TrackerOutput) -> (bbox: CGRect, confidence: Float) {
        let bboxPred = output.bbox
        let clsPred = output.cls
        let mapCols = clsPred.shape[2].intValue
        let mapRows = clsPred.shape[3].intValue

        let clsPointer = clsPred.dataPointer.assumingMemoryBound(to: Float.self)
        var clsMax: Float = 0
        var clsIdx: vDSP_Length = 0
        vDSP_maxvi(clsPointer, vDSP_Stride(1), &clsMax, &clsIdx, vDSP_Length(clsPred.count))
        let row = Int(clsIdx) / mapCols
        let col = Int(clsIdx) % mapCols

        let gridX = (col - Constants.scoreSize / 2) * Constants.totalStride + Constants.trackSize / 2
        let gridY = (row - Constants.scoreSize / 2) * Constants.totalStride + Constants.trackSize / 2
        let x1 = gridX - bboxPred[Int(clsIdx)].intValue
        let y1 = gridY - bboxPred[Int(clsIdx) + 1 * mapCols * mapRows].intValue
        let x2 = gridX + bboxPred[Int(clsIdx) + 2 * mapCols * mapRows].intValue
        let y2 = gridY + bboxPred[Int(clsIdx) + 3 * mapCols * mapRows].intValue

        let bbox = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
        let confidence = 1.0 / (1.0 + expf(-clsPred[Int(clsIdx)].floatValue))
        return (bbox, confidence)
    }
}
