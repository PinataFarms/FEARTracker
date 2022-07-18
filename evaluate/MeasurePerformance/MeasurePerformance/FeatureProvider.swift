//
//  FeatureProvider.swift
//  MeasurePerformance
//
//  Created by Vasyl Borsuk on 05.11.2021.
//

import UIKit
import CoreML

class FeatureProvider: MLFeatureProvider {
    var featureNames: Set<String> {
        return Set(features.keys)
    }
    var features: [String: MLFeatureValue]

    init?(image: UIImage, for model: MLModel, arrayInitFn: (Int) -> Float = { _ in 0 }) {
        features = [:]
        let inputDescirptions = model.modelDescription.inputDescriptionsByName
        for (inputName, inputDesc) in inputDescirptions {
            let feature: MLFeatureValue
            if inputDesc.type == .image, let constraints = inputDesc.imageConstraint {
                let buf = image.pixelBuffer(width: constraints.pixelsWide, height: constraints.pixelsHigh)!
                feature = MLFeatureValue(pixelBuffer: buf)
            } else if inputDesc.type == .multiArray, let constraints = inputDesc.multiArrayConstraint {
                let array = try! MLMultiArray(shape: constraints.shape, dataType: constraints.dataType)
                for i in 0..<array.count {
                    array[i] = NSNumber(value: arrayInitFn(i))
                }
                feature = MLFeatureValue(multiArray: array)
            } else {
                return nil
            }
            features[inputName] = feature
        }
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        return features[featureName]
    }
}
