import argparse
import time
import openvino as ov
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from utils.transforms import get_transforms
from utils.dataset import get_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="AUTO", choices=["CPU", "GPU", "NPU", "AUTO"],
                        help="Target device for OpenVINO inference (default: AUTO)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for evaluation (default: 1)")
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

    all_preds = []
    all_labels = []

    print(f"[INFO] Running evaluation on test set (batch-size: {batch_size}) using {actual_device}...")
    start_time = time.perf_counter()

    for images, labels in test_loader:
        results = compiled_model(images.numpy())
        preds = results[output_layer].argmax(axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    end_time = time.perf_counter()
    total_time = end_time - start_time
    num_images = len(all_labels)
    avg_latency = total_time / num_images
    throughput = num_images / total_time

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n[RESULTS]")
    print(f"Device used: {actual_device}")
    print(f"Batch-size: {batch_size}")
    print(f"Total running time: {total_time:.2f} seconds")
    print(f"Average latency: {avg_latency*1000:.2f} ms/image")
    print(f"Throughput: {throughput:.2f} images/sec")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nDetailed classification report:")
    print(classification_report(all_labels, all_preds, zero_division=0, digits=4))


if __name__ == "__main__":
    main()
