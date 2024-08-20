import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load MiDaS model
model_type = "DPT_Large"  # Other options: "DPT_Hybrid", "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# Function to estimate depth
def estimate_depth(image_path):
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = midas_transforms.default_transform(rgb_img).to(device)
    
    with torch.no_grad():
        prediction = midas(input_tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb_img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth_map = prediction.cpu().numpy()
    return depth_map, img

# Load scene and person images
scene_img_path = 'road2.jpg'  # Path to the scene image
person_img_path = 'runner.jpg'  # Path to the person image

# Estimate depth for both scene and person
scene_depth_map, scene_img = estimate_depth(scene_img_path)
person_depth_map, person_img = estimate_depth(person_img_path)

# Normalize the depth maps using the same scale
min_depth = min(scene_depth_map.min(), person_depth_map.min())
max_depth = max(scene_depth_map.max(), person_depth_map.max())

scene_depth_map_normalized = (scene_depth_map - min_depth) / (max_depth - min_depth)
person_depth_map_normalized = (person_depth_map - min_depth) / (max_depth - min_depth)

# Plot the depth maps side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].imshow(scene_depth_map_normalized, cmap='plasma')
axes[0].set_title('Scene Depth Map')
axes[0].axis('off')

axes[1].imshow(person_depth_map_normalized, cmap='plasma')
axes[1].set_title('Person Depth Map')
axes[1].axis('off')

plt.show()

