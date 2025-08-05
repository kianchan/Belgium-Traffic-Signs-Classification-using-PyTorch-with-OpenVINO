import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.transforms import get_transforms
from utils.dataset import get_dataloaders
from utils.model import TrafficSignNet

# Try Intel Extension for PyTorch (IPEX) if available
try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except ImportError:
    has_ipex = False


def get_device():
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        print("[INFO] NVIDIA GPU detected: Using CUDA")
        return torch.device("cuda"), False
    elif has_ipex:
        if hasattr(torch, "xpu"):
            if torch.xpu.is_available():
                print("[INFO] Intel GPU detected: Using XPU (IPEX)")
                return torch.device("xpu"), True
            else:
                print("[ERROR] IPEX is installed, but Intel GPU is not available.")
                print("→ Install Intel oneAPI runtime & GPU drivers:")
                print("   https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html")
        else:
            print("[ERROR] IPEX installed but PyTorch does not have 'xpu' support.")
            print("→ Reinstall PyTorch & IPEX with xpu support:")
            print("   pip install torch==<version> intel_extension_for_pytorch[xpu]")
    else:
        print("[INFO] No NVIDIA GPU or Intel GPU detected. Using CPU.")
        print("[HINT] To enable Intel GPU support, install IPEX:")
        print("   pip install torch==<version> intel_extension_for_pytorch[xpu]")
        print("   and Intel oneAPI runtime (Windows):")
        print("   https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html")

    return torch.device("cpu"), False


def main():
    os.makedirs("models", exist_ok=True)

    # Load data
    train_loader, test_loader, num_classes = get_dataloaders(get_transforms(), batch_size=32)

    # Detect device
    device, intel_xpu = get_device()

    # Prompt for AMP if Intel GPU
    use_amp = False
    if intel_xpu:
        choice = input("Do you want to enable Automatic Mixed Precision (AMP) for faster training? (y/n): ").strip().lower()
        use_amp = choice == "y"
        if use_amp:
            print("[INFO] AMP enabled for Intel GPU training.")

    # Initialize model
    model = TrafficSignNet(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            if intel_xpu and use_amp:
                # Intel GPU AMP: autocast only (GradScaler is not supported)
                with torch.amp.autocast("xpu"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "models/traffic_signs_cnn.pth")
    print("[INFO] PyTorch model saved at models/traffic_signs_cnn.pth")


if __name__ == "__main__":
    main()

