import torch
from PIL import Image
from models import MAGVITv2
from training.utils import image_transform
import numpy as np

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VQ model
vq_model = MAGVITv2.from_pretrained("showlab/magvitv2").to(device)
vq_model.eval()

# Load and preprocess image
input_image = Image.open("/home/arshkon/Projects/Show-o/interleaved_output/generated_0.png").convert("RGB")
image_tensor = image_transform(input_image, resolution=256).unsqueeze(0).to(device)  # Add batch dimension

print("Image tensor shape:", image_tensor.shape)

# Encode and decode
with torch.no_grad():
    # Get latent code
    image_tokens = vq_model.get_code(image_tensor)

    print("Latent code shape:", image_tokens.shape)
    
    # Reconstruct image
    reconstructed = vq_model.decode_code(image_tokens)

# Postprocess and save
reconstructed = torch.clamp((reconstructed.squeeze(0).cpu() + 1.0) / 2.0, 0.0, 1.0)
reconstructed = Image.fromarray((reconstructed.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
reconstructed.save("reconstructed_image3.jpg")

print("Original and reconstructed images saved")