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
import CoreImage

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
        static let templateSize: Int = 128
        static let templateOffset: CGFloat = 0.2
        static let searchSize: Int = 256
        static let searchOffset: CGFloat = 2
        static let scoreSize: Int = 16
        static let totalStride: Int = 16
        static let threshold: Float = 0.7
    }

    private struct TrackerState {
        let templateFeatures: MLMultiArray
        let lastRectangle: CGRect
        let meanColor: CIColor
    }

    /// Feature extraction network
    private let modelInit: TrackerInit = {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        return try! TrackerInit(configuration: config)
    }()

    /// Tracking network
    private let model: Tracker = {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        return try! Tracker(configuration: config)
    }()

    private let ciContext: CIContext = {
        return CIContext(options: nil)
    }()

    private var state: TrackerState? = nil

    public var isInitialized: Bool { state != nil }

    public func initialize(image: CVPixelBuffer, rect: CGRect) throws {
        let meanColor = computeMeanColor(image: image) ?? .black
        let input = try preprocessImage(image: image, rect: rect, cropSize: CGFloat(Constants.templateSize),
                                        offset: Constants.templateOffset, paddingColor: meanColor)
        let result = try modelInit.prediction(image: input.image)
        self.state = TrackerState(templateFeatures: result.features, lastRectangle: rect, meanColor: meanColor)
    }

    public func track(image: CVPixelBuffer) throws -> VOTrackerResult {
        guard let state = state else {
            throw VOTrackerError.uninitializedError
        }

        let input = try preprocessImage(image: image, rect: state.lastRectangle, cropSize: CGFloat(Constants.searchSize),
                                        offset: Constants.searchOffset, paddingColor: state.meanColor)

        // predict and decode
        let output = try model.prediction(image: input.image, template_features: state.templateFeatures)
        let result = decode(output: output)
        if result.confidence < Constants.threshold {
            throw VOTrackerError.thresholdError
        }

        // rescale bbox to original image size
        let scaleX = input.paddedRect.width / CGFloat(Constants.searchSize)
        let scaleY = input.paddedRect.height / CGFloat(Constants.searchSize)
        let rectTransform = CGAffineTransform(translationX: input.paddedRect.minX, y: input.paddedRect.minY)
            .scaledBy(x: scaleX, y: scaleY)
        let rect = result.bbox.applying(rectTransform)

        // update state
        self.state = TrackerState(templateFeatures: state.templateFeatures, lastRectangle: rect, meanColor: state.meanColor)

        return VOTrackerResult(bbox: rect, confidence: result.confidence)
    }

    public func clear() {
        state = nil
    }
}

// MARK: - functions to preprocess image
private extension VOTracker {
    struct PreprocessedImage {
        let image: CVPixelBuffer
        let searchRect: CGRect
        let paddedRect: CGRect
    }

    func preprocessImage(image: CVPixelBuffer, rect: CGRect, cropSize: CGFloat, offset: CGFloat, paddingColor: CIColor) throws -> PreprocessedImage {
        let pixelFormat = CVPixelBufferGetPixelFormatType(image)
        guard let targetBuffer = createPixelBuffer(width: Int(cropSize), height: Int(cropSize), pixelFormat: pixelFormat) else {
            throw VOTrackerError.preprocessingError
        }

        let imageWidth = CGFloat(CVPixelBufferGetWidth(image))
        let imageHeight = CGFloat(CVPixelBufferGetHeight(image))

        let xOffset = offset * rect.width
        let yOffset = offset * rect.height
        let context = rect.insetBy(dx: -xOffset, dy: -yOffset)

        let padLeft = max(-context.minX, 0)
        let padTop = max(-context.minY, 0)
        let padRight = max(context.minX + context.width - imageWidth, 0)
        let padBottom = max(context.minY + context.height - imageHeight, 0)

        let cropRect = CGRect(x: context.minX + padLeft, y: context.minY + padTop,
                              width: context.width - padLeft - padRight, height: context.height - padTop - padBottom)
        let paddedRect = CGRect(x: rect.minX - context.minX, y: rect.minY - context.minY,
                                width: rect.width, height: rect.height)

        let scaleTransform = CGAffineTransform(scaleX: cropSize / context.width, y: cropSize / context.height)

        let ciImage = CIImage(cvPixelBuffer: image)
            // CG -> CI coordinates mapping
            .transformed(by: .init(scaleX: 1, y: -1))
            .transformed(by: .init(translationX: 0, y: imageHeight))
            // crop image and move to origin
            .cropped(to: cropRect)
            .transformed(by: .init(translationX: -cropRect.minX, y: -cropRect.minY))
            // add padding
            .transformed(by: .init(translationX: padLeft, y: padTop))
            // apply scale transform
            .transformed(by: scaleTransform)
            // CI -> CG coordinates mapping
            .transformed(by: .init(scaleX: 1, y: -1))
            .transformed(by: .init(translationX: 0, y: cropSize))

        let ciBackground = CIImage(color: paddingColor).cropped(to: CGRect(x: 0, y: 0, width: cropSize, height: cropSize))
        let ciResult = ciImage.composited(over: ciBackground)

        ciContext.render(ciResult, to: targetBuffer, bounds: CGRect(x: 0, y: 0, width: cropSize, height: cropSize), colorSpace: ciImage.colorSpace)
        return PreprocessedImage(image: targetBuffer, searchRect: paddedRect.applying(scaleTransform), paddedRect: context)
    }

    func computeMeanColor(image: CVPixelBuffer) -> CIColor? {
        let inputImage = CIImage(cvPixelBuffer: image)
        let extentVector = CIVector(x: inputImage.extent.origin.x, y: inputImage.extent.origin.y,
                                    z: inputImage.extent.size.width, w: inputImage.extent.size.height)

        guard let filter = CIFilter(name: "CIAreaAverage", parameters: [kCIInputImageKey: inputImage, kCIInputExtentKey: extentVector]),
              let outputImage = filter.outputImage
        else { return nil }

        var bitmap = [UInt8](repeating: 0, count: 4)
        ciContext.render(outputImage, toBitmap: &bitmap, rowBytes: 4, bounds: CGRect(x: 0, y: 0, width: 1, height: 1), format: .RGBA8, colorSpace: nil)

        return CIColor(red: CGFloat(bitmap[0]) / 255, green: CGFloat(bitmap[1]) / 255, blue: CGFloat(bitmap[2]) / 255, alpha: CGFloat(bitmap[3]) / 255)
    }
}

// MARK: - functions to decode model predictions
private extension VOTracker {
    func decode(output: TrackerOutput) -> (bbox: CGRect, confidence: Float) {
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

        let gridX = (col - Constants.scoreSize / 2) * Constants.totalStride + Constants.searchSize / 2
        let gridY = (row - Constants.scoreSize / 2) * Constants.totalStride + Constants.searchSize / 2
        let x1 = gridX - bboxPred[Int(clsIdx)].intValue
        let y1 = gridY - bboxPred[Int(clsIdx) + 1 * mapCols * mapRows].intValue
        let x2 = gridX + bboxPred[Int(clsIdx) + 2 * mapCols * mapRows].intValue
        let y2 = gridY + bboxPred[Int(clsIdx) + 3 * mapCols * mapRows].intValue

        let bbox = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
        let confidence = 1.0 / (1.0 + expf(-clsPred[Int(clsIdx)].floatValue))
        return (bbox, confidence)
    }
}
