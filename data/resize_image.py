from PIL import Image

# Load an image
image = Image.open("/Users/arielmalowany/Desktop/Learning/COUNT-GEN/AttGAN-PyTorch/data/custom/magui.jpg")

# Resize to 300x200
resized_image = image.resize((384, 384))

# Save the resized image
resized_image.save("/Users/arielmalowany/Desktop/Learning/COUNT-GEN/AttGAN-PyTorch/data/custom/magui.jpg")
