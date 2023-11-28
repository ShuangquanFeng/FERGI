from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import torch.nn as nn

class PickScore(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device

        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        self.model_PickScore = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
    
    def score(self, prompt, image_paths):
        images = [Image.open(path) for path in image_paths]

        # preprocess
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            image_embs = self.model_PickScore.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model_PickScore.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            scores = self.model_PickScore.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        return scores.cpu().tolist()