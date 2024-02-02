from PIL import Image
import requests

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from transformers import CLIPProcessor, CLIPModel

def CLIP(img: torch.Tensor) -> torch.Tensor:
    # incoming tensor size is torch.Size([32, 32, 3, 200, 200])
    # resize each image in the batch to 224x224 keeping the other dimensions of the tensor the same
    img_tensor = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
    print("resized img tensor size: ", img_tensor.size())
    # Visualize some of the resized images
    img = img_tensor[0, 0].permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # img = img.resize((224, 224))
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    perceptual_emb = model.vision_model(img)
    
    return perceptual_emb

class CLIP_(torch.nn.Module):
    def __init__(self):
        super(CLIP_, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.linear = torch.nn.Linear(768, 72)
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # incoming tensor size is torch.Size([32, 32, 3, 200, 200])
        # resize each image in the batch to 224x224 keeping the other dimensions of the tensor the same
        img_tensor = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        print("resized img tensor size: ", img_tensor.size())
        # Visualize some of the resized images
        #img_tensor --> 32*4 x 3, 224, 224
        
        # img = img.resize((224, 224))
        with torch.no_grad():
            perceptual_emb = self.model.vision_model(img_tensor)[-1]
        perceptual_emb = self.linear(perceptual_emb)
        
        
        return perceptual_emb