import torch
print("GPU Available:", torch.cuda.is_available())  # Should print True
print("GPU Name:", torch.cuda.get_device_name(0))  # Check GPU model
