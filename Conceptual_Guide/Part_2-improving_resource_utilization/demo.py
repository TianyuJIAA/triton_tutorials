import torch
from utils.model import STRModel


def gen_onnx():

    model = STRModel(input_channels=1, output_channels=512, num_classes=37)

    state = torch.load("None-ResNet-None-CTC.pth", map_location=torch.device("cpu"))
    state = {key.replace("module.", ""): value for key, value in state.items()}
    model.load_state_dict(state)

    trace_input = torch.randn(1, 1, 32, 100)
    torch.onnx.export(
        model,
        trace_input,
        "str.onnx",
        verbose=True,
        dynamic_axes={"input.1": [0], "308": [0]},
    )


if __name__ == "__main__":
    gen_onnx()
