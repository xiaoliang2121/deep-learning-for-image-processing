import torch
from torchvision.models import ResNet50_Weights
from torchvision.models import resnet50

def main():
    export_fp16 = False
    export_onnx_path = f"resnet50_fp{16 if export_fp16 else 32}.onnx"
    device = torch.device("cuda:0")
    
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    model.to(device)
    if export_fp16:
        model.half()
        
    with torch.inference_mode():
        dtype = torch.float16 if export_fp16 else torch.float32
        x = torch.randn(size=(1, 3, 224, 224), dtype=dtype, device=device)
        torch.onnx.export(model=model,
                          args=(x,),
                          f=export_onnx_path,
                          input_names=["image"],
                          output_names=["output"],
                          dynamic_axes={"image": {2:"width", 3:"height"}},
                          opset_version=17)
        
if __name__ == '__main__':
    main()