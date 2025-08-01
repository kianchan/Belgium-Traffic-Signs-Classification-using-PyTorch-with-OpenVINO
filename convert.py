import os
import torch
import openvino as ov
from utils.transforms import get_transforms
from utils.dataset import get_dataloaders
from utils.model import TrafficSignNet

def main():
    os.makedirs("models", exist_ok=True)

    _, _, num_classes = get_dataloaders(get_transforms(), batch_size=1)

    model = TrafficSignNet(num_classes)
    model.load_state_dict(torch.load("models/traffic_signs_cnn.pth", map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 3, 64, 64)
    ov_model = ov.convert_model(model, example_input=dummy_input)

    ov_fp16_path = "models/traffic_signs_cnn_fp16.xml"
    ov.save_model(ov_model, ov_fp16_path, compress_to_fp16=True)
    print(f"[INFO] OpenVINO FP16 model saved at {ov_fp16_path}")

if __name__ == "__main__":
    main()