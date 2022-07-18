import copy
from functools import partial
from typing import List, Dict, Any

import torch
import coremltools
from coremltools.proto import Model_pb2
from coremltools.models.neural_network.quantization_utils import (
    quantize_weights,
    QuantizedLayerSelector,
)


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def coreml4_convert(
    model: torch.nn.Module,
    model_description: Dict[str, Any],
    quantize: bool = True,
):
    traced_model = _torch_to_jit(model, model_description)
    _jit_to_coreml(traced_model, model_description, quantize=quantize)


def _torch_to_jit(model: torch.nn.Module, model_description: Dict[str, Any]):
    inputs = tuple(torch.rand(inp["shape"]) for inp in model_description["inputs"])
    traced_model = torch.jit.trace(model, inputs)
    traced_model.save(f"{model_description['model_name']}.trcd")
    return traced_model


def _jit_to_coreml(
    traced_model: torch.jit.ScriptModule,
    model_description: Dict[str, Any],
    quantize: bool = True,
):
    coreml_inputs, preprocessings = _get_coreml_inputs(model_description)
    coreml_model = coremltools.convert(
        traced_model,
        inputs=coreml_inputs,
        minimum_deployment_target=model_description["minimum_deployment_target"],
    )

    spec = coreml_model.get_spec()
    for preprocessing in preprocessings:
        preprocessing(spec)
    rename_outputs(spec, [out["name"] for out in model_description["outputs"]])
    model = coremltools.models.MLModel(spec)
    _add_metadata(model, model_description)
    model.save(f"{model_description['model_name']}.mlmodel")

    if quantize:
        model_fp16 = quantize_weights(model, nbits=16, quantization_mode="linear", selector=QuantizedLayerSelector())
        if not isinstance(model_fp16, coremltools.models.MLModel):
            model_fp16 = coremltools.models.MLModel(model_fp16)
        model_fp16.save(f"{model_description['model_name']}_quantized.mlmodel")


def _get_coreml_inputs(model_description: Dict[str, Any]):
    coreml_inputs = []
    coreml_inputs_preprocessing = []
    for desc in model_description["inputs"]:
        if desc["type"] == "image":
            # image spec
            mean = desc.get("mean", _MEAN)
            biases = [-(x * 255.0) for x in mean]
            inp = coremltools.ImageType(
                name=desc["name"],
                shape=desc["shape"],
                bias=biases,
                color_layout=desc["color_layout"],
            )
            coreml_inputs.append(inp)

            # image preprocessing
            std = desc.get("std", _STD)
            scales = [1.0 / (x * 255.0) for x in std]
            preprocessing = partial(add_scale_preprocessing, scales=scales, input_name=desc["name"])
            coreml_inputs_preprocessing.append(preprocessing)
        else:
            inp = coremltools.TensorType(name=desc["name"], shape=desc["shape"], dtype=desc["dtype"])
            coreml_inputs.append(inp)
    return coreml_inputs, coreml_inputs_preprocessing


def _add_metadata(model: coremltools.models.MLModel, model_description: Dict[str, Any]):
    metadata = model_description["metadata"]
    model.author = metadata["author"]
    model.short_description = metadata["short_description"]
    model.version = metadata["version"]

    for inp in model_description["inputs"]:
        model.input_description[inp["name"]] = inp["description"]

    for out in model_description["outputs"]:
        model.output_description[out["name"]] = out["description"]


def rename_outputs(spec: Model_pb2, new_names: List[str]):
    output_desc = spec.description.output
    current_names = [out.name for out in output_desc]
    for current_name, new_name in zip(current_names, new_names):
        coremltools.utils.rename_feature(spec, current_name, new_name)


def add_scale_preprocessing(spec: Model_pb2, scales: List[float], input_name: str):
    scale_name = f"{input_name}_scaled"
    nn_spec = spec.neuralNetwork
    layers = nn_spec.layers  # this is a list of all the layers
    layers_copy = copy.deepcopy(layers)  # make a copy of the layers, these will be added back later
    del nn_spec.layers[:]  # delete all the layers

    # add a scale layer now
    # since mlmodel is in protobuf format, we can add proto messages directly
    # To look at more examples on how to add other layers: see "builder.py" file in coremltools repo
    scale_layer = nn_spec.layers.add()
    scale_layer.name = "scale_layer"
    scale_layer.input.append(input_name)
    scale_layer.output.append(scale_name)

    params = scale_layer.scale
    params.scale.floatValue.extend(scales)  # scale values for RGB
    params.shapeScale.extend([3, 1, 1])  # shape of the scale vector

    # now add back the rest of the layers (which happens to be just one in this case: the crop layer)
    nn_spec.layers.extend(layers_copy)

    # need to also change the input of the next layers to match the output of the scale layer
    for layer in nn_spec.layers[1:]:
        for idx, inp in enumerate(layer.input):
            if inp == input_name:
                layer.input[idx] = scale_name
