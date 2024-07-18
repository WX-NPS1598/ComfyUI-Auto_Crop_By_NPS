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
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image1", "image2")
    FUNCTION = 'auto_crop_images'
    CATEGORY = 'Image Processing'

    def auto_crop_images(self, crop_top, crop_bottom, crop_left, crop_right, rotation, image1=None, image2=None):
        def tensor2pil(tensor):
            return Image.fromarray((tensor.squeeze().cpu().numpy() * 255).astype(np.uint8))

        def pil2tensor(pil_img):
            return torch.from_numpy(np.array(pil_img)).float().div(255).unsqueeze(0)

        ret_images1 = []
        ret_images2 = []
        
        if image1 is not None:
            for img_tensor in image1:
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
                new_width, new_height = img.size
                if crop_top > 0:
                    img = ImageOps.expand(img, border=(0, int(height * crop_top), 0, 0), fill=(255, 255, 255))
                if crop_bottom > 0:
                    img = ImageOps.expand(img, border=(0, 0, 0, int(height * crop_bottom)), fill=(255, 255, 255))
                if crop_left > 0:
                    img = ImageOps.expand(img, border=(int(width * crop_left), 0, 0, 0), fill=(255, 255, 255))
                if crop_right > 0:
                    img = ImageOps.expand(img, border=(0, 0, int(width * crop_right), 0), fill=(255, 255, 255))

                img = img.rotate(-rotation, expand=True, fillcolor=(255, 255, 255))

                ret_images1.append(pil2tensor(img))

        if image2 is not None:
            for img_tensor in image2:
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
                new_width, new_height = img.size
                if crop_top > 0:
                    img = ImageOps.expand(img, border=(0, int(height * crop_top), 0, 0), fill=(255, 255, 255))
                if crop_bottom > 0:
                    img = ImageOps.expand(img, border=(0, 0, 0, int(height * crop_bottom)), fill=(255, 255, 255))
                if crop_left > 0:
                    img = ImageOps.expand(img, border=(int(width * crop_left), 0, 0, 0), fill=(255, 255, 255))
                if crop_right > 0:
                    img = ImageOps.expand(img, border=(0, 0, int(width * crop_right), 0), fill=(255, 255, 255))

                img = img.rotate(-rotation, expand=True, fillcolor=(255, 255, 255))

                ret_images2.append(pil2tensor(img))

        return (
            torch.cat(ret_images1, dim=0) if ret_images1 else None,
            torch.cat(ret_images2, dim=0) if ret_images2 else None
        )

NODE_CLASS_MAPPINGS = {
    "AutoCropByNPS": AutoCropByNPS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoCropByNPS": "auto crop by NPS"
}
