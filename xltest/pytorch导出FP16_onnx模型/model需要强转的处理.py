import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)

        kernel = torch.tensor([[0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.1],
                               [0.1, 0.1, 0.1]], dtype=torch.float32, device=x.device).reshape([1, 1, 3, 3])
        x = F.conv2d(x, weight=kernel, bias=None, stride=1)

        return x


def main():
    export_fp16 = True
    export_onnx_path = f"my_model_fp{16 if export_fp16 else 32}.onnx"
    device = torch.device("cuda:0")

    model = MyModel()
    model.eval()
    model.to(device)
    if export_fp16:
        model.half()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.inference_mode():
            dtype = torch.float16 if export_fp16 else torch.float32
            x = torch.randn(size=(1, 3, 224, 224), dtype=dtype, device=device)
            model(x)
            torch.onnx.export(model=model,
                              args=(x,),
                              f=export_onnx_path,
                              input_names=["image"],
                              output_names=["output"],
                              dynamic_axes={"image": {2: "width", 3: "height"}},
                              opset_version=17)


if __name__ == '__main__':
    main()

