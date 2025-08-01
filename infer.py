import argparse
import time
import openvino as ov
from utils.transforms import get_transforms
from utils.dataset import get_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="AUTO", choices=["CPU", "GPU", "NPU", "AUTO"],
                        help="Target device for OpenVINO inference (default: AUTO)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference (default: 1)")
    args = parser.parse_args()

    core = ov.Core()
    available_devices = core.get_available_devices()

    target_device = args.device
    batch_size = args.batch_size

    if args.device != "AUTO" and args.device not in available_devices:
        print(f"[WARN] {args.device} not available. Falling back to CPU.")
        target_device = "CPU"

    print(f"[INFO] Target device for OpenVINO inference: {target_device}")
    compiled_model = core.compile_model("models/traffic_signs_cnn_fp16.xml", target_device)

    # Detect which device AUTO selected
    actual_device = compiled_model.get_property("EXECUTION_DEVICES")
    if target_device == "AUTO":
        print(f"[INFO] AUTO selected: {actual_device}")
        if "GPU" in actual_device:
            batch_size = 16
            print(f"[INFO] AUTO detected GPU. Batch-size changed to {batch_size} for better throughput.")

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    _, test_loader, _ = get_dataloaders(get_transforms(), batch_size=batch_size)
    images, labels = next(iter(test_loader))

    start_time = time.perf_counter()
    results = compiled_model(images.numpy())
    end_time = time.perf_counter()

    preds = results[output_layer].argmax(axis=1)

    print(f"Predicted classes: {preds.tolist()}, True classes: {labels.tolist()}")
    print(f"Device used: {actual_device}")
    print(f"Batch-size: {batch_size}")
    print(f"Total inference time: {(end_time - start_time):.4f} seconds")
    print(f"Average latency: {(end_time - start_time) / len(labels) * 1000:.2f} ms/image")
    print(f"Throughput: {len(labels) / (end_time - start_time):.2f} images/sec")


if __name__ == "__main__":
    main()
