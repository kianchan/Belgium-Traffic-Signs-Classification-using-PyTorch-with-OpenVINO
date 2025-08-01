import argparse
import time
import openvino as ov
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.transforms import get_transforms
from utils.dataset import get_dataloaders


def run_evaluation(core, model_path, device, batch_size):
    """Run evaluation and return throughput & metrics."""
    compiled_model = core.compile_model(model_path, device)
    actual_device = compiled_model.get_property("EXECUTION_DEVICES")

    # Ensure device name is always string
    if isinstance(actual_device, list):
        actual_device = ", ".join(actual_device)

    print(f"[INFO] Actual device being used: {actual_device}")

    _, test_loader, _ = get_dataloaders(get_transforms(), batch_size=batch_size)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    all_preds, all_labels = [], []
    start_time = time.perf_counter()

    for images, labels in test_loader:
        results = compiled_model(images.numpy())
        preds = results[output_layer].argmax(axis=1)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    end_time = time.perf_counter()
    total_time = end_time - start_time
    throughput = len(all_labels) / total_time

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return {
        "device": actual_device,
        "batch_size": batch_size,
        "throughput": throughput,
        "latency": total_time / len(all_labels),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="AUTO", choices=["CPU", "GPU", "NPU", "AUTO", "ALL"],
                        help="Target device for OpenVINO inference (AUTO picks best, ALL tests multiple)")
    parser.add_argument("--batch-sizes", type=str, default="1,8,16",
                        help="Comma-separated batch sizes to test (default: 1,8,16)")
    args = parser.parse_args()

    core = ov.Core()
    available_devices = core.get_available_devices()

    if args.device == "ALL":
        devices_to_test = [d for d in available_devices if d in ["CPU", "GPU"]]
    else:
        devices_to_test = [args.device]

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    model_path = "models/traffic_signs_cnn_fp16.xml"
    print(f"[INFO] Testing devices {devices_to_test} with batch-sizes {batch_sizes}...\n")

    results = []
    for device in devices_to_test:
        for batch_size in batch_sizes:
            print(f"[INFO] Running on {device} (batch-size: {batch_size})...")
            res = run_evaluation(core, model_path, device, batch_size)
            results.append(res)
            print(f"   → Throughput: {res['throughput']:.2f} images/sec, Accuracy: {res['accuracy']:.4f}")

    # Summary
    print("\n" + "="*60)
    print(f"{'Device':<20} {'Batch-size':<10} {'Throughput(img/s)':<20} {'Accuracy':<10}")
    print("="*60)
    for res in results:
        print(f"{res['device']:<20} {res['batch_size']:<10} {res['throughput']:<20.2f} {res['accuracy']:<10.4f}")

    # Best throughput
    best = max(results, key=lambda r: r['throughput'])
    print("\n[SUMMARY] Best combination:")
    print(f"Device: {best['device']} | Batch-size: {best['batch_size']} | "
          f"Throughput: {best['throughput']:.2f} img/s | Accuracy: {best['accuracy']:.4f}")


if __name__ == "__main__":
    main()
