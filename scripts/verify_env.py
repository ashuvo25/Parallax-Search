
import torch

print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    print("CUDA available: Yes")
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.rand(3, 3, device="cuda")
    print("GPU tensor OK:", x.sum().item())
else:
    print("CUDA is not available")

