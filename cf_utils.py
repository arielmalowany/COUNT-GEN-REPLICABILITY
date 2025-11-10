import torch

# Helper functions (from COUNTGEN repo, adapted by me) 
# Map images from (-1, 1) to (0, 1)

def normalize_0_1(_img, fake = False):
  if fake:
    img = (_img - torch.min(_img))/(torch.max(_img) - torch.min(_img))
  else:
    img = (_img + 1)/(2)
  return img

def denormalize(img):
  return img + 1