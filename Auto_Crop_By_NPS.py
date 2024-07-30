import numpy as np
from PIL import Image, ImageOps
import torch

# Ensure PIL can handle large images
Image.MAX_IMAGE_PIXELS = None

class AutoCropByNPS:

    def __init__(self):
        self.crop_top = 0.0
        self.crop_bottom = 0.0
        self.crop_left = 0.0
        self.crop_right = 0.0
        self.rotation = 0.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "crop_top": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.02, "display": "slider"}),
                "crop_bottom": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.02, "display": "slider"}),
                "crop_left": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.02, "display": "slider"}),
                "crop_right": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.02, "display": "slider"}),
                "rotation": ("FLOAT", {"default": 0, "min": -180, "max": 180, "step": 1, "display": "slider"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = 'auto_crop_images'
    CATEGORY = 'NPS1598'

    def auto_crop_images(self, crop_top, crop_bottom, crop_left, crop_right, rotation, image=None, mask=None):
        def tensor2pil(tensor):
            return Image.fromarray((tensor.squeeze().cpu().numpy() * 255).astype(np.uint8))

        def pil2tensor(pil_img):
            return torch.from_numpy(np.array(pil_img)).float().div(255).unsqueeze(0)

        ret_images = []
        ret_masks = []

        if image is not None:
            for img_tensor in image:
                img_tensor = torch.unsqueeze(img_tensor, 0)
                img = tensor2pil(img_tensor)
                width, height = img.size

                # Perform cropping
                left = int(width * abs(crop_left)) if crop_left < 0 else 0
                right = width - int(width * abs(crop_right)) if crop_right < 0 else width
                top = int(height * abs(crop_top)) if crop_top < 0 else 0
                bottom = height - int(height * abs(crop_bottom)) if crop_bottom < 0 else height
                
                img = img.crop((left, top, right, bottom))
                
                # Perform expanding
                if crop_top > 0:
                    img = ImageOps.expand(img, border=(0, int(height * crop_top), 0, 0), fill=(255, 255, 255))
                if crop_bottom > 0:
                    img = ImageOps.expand(img, border=(0, 0, 0, int(height * crop_bottom)), fill=(255, 255, 255))
                if crop_left > 0:
                    img = ImageOps.expand(img, border=(int(width * crop_left), 0, 0, 0), fill=(255, 255, 255))
                if crop_right > 0:
                    img = ImageOps.expand(img, border=(0, 0, int(width * crop_right), 0), fill=(255, 255, 255))

                img = img.rotate(-rotation, expand=True, fillcolor=(255, 255, 255))

                ret_images.append(pil2tensor(img))

        if mask is not None:
            for mask_tensor in mask:
                mask_tensor = torch.unsqueeze(mask_tensor, 0)
                mask_img = tensor2pil(mask_tensor)
                width, height = mask_img.size

                # Perform cropping
                left = int(width * abs(crop_left)) if crop_left < 0 else 0
                right = width - int(width * abs(crop_right)) if crop_right < 0 else width
                top = int(height * abs(crop_top)) if crop_top < 0 else 0
                bottom = height - int(height * abs(crop_bottom)) if crop_bottom < 0 else height
                
                mask_img = mask_img.crop((left, top, right, bottom))
                
                # Perform expanding
                if crop_top > 0:
                    mask_img = ImageOps.expand(mask_img, border=(0, int(height * crop_top), 0, 0), fill=255)
                if crop_bottom > 0:
                    mask_img = ImageOps.expand(mask_img, border=(0, 0, 0, int(height * crop_bottom)), fill=255)
                if crop_left > 0:
                    mask_img = ImageOps.expand(mask_img, border=(int(width * crop_left), 0, 0, 0), fill=255)
                if crop_right > 0:
                    mask_img = ImageOps.expand(mask_img, border=(0, 0, int(width * crop_right), 0), fill=255)

                mask_img = mask_img.rotate(-rotation, expand=True, fillcolor=255)

                ret_masks.append(pil2tensor(mask_img))

        return (
            torch.cat(ret_images, dim=0) if ret_images else None,
            torch.cat(ret_masks, dim=0) if ret_masks else None
        )

NODE_CLASS_MAPPINGS = {
    "AutoCropByNPS": AutoCropByNPS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCropByNPS": "Auto Crop by NPS"
}
