#https://huggingface.co/docs/transformers/main/en/model_doc/dinov2
#https://github.com/facebookresearch/sscd-copy-detection
#https://huggingface.co/docs/diffusers/conceptual/evaluation

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import glob
import torch 
from PIL import Image
_ = torch.manual_seed(42)
import PIL
import numpy as np
from torchmetrics.multimodal import CLIPScore
from transformers import AutoImageProcessor, Dinov2Model, CLIPProcessor, CLIPModel
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn

from transformers import (
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor,
)

# Load models

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", cache_dir = './huggingface_models/')
dino_model = Dinov2Model.from_pretrained("facebook/dinov2-base", cache_dir = './huggingface_models/').cuda()
sscd_model = torch.jit.load("./pretrained_models/sscd_disc_mixup.torchscript.pt").cuda()
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir = './huggingface_models/').cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir = './huggingface_models/')

#CLIP Directional Similarity models
clipd_id = "openai/clip-vit-large-patch14"
clipd_tokenizer = CLIPTokenizer.from_pretrained(clipd_id, cache_dir = './huggingface_models/')
clipd_text_encoder = CLIPTextModelWithProjection.from_pretrained(clipd_id, cache_dir = './huggingface_models/').cuda()
clipd_image_processor = CLIPImageProcessor.from_pretrained(clipd_id, cache_dir = './huggingface_models/')
clipd_image_encoder = CLIPVisionModelWithProjection.from_pretrained(clipd_id, cache_dir = './huggingface_models/').cuda()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
)
small_288 = transforms.Compose([
    transforms.Resize(288),
    transforms.ToTensor(),
    normalize,
])


class DirectionalSimilarity(nn.Module):
    def __init__(self, tokenizer, text_encoder, image_processor, image_encoder):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.image_processor = image_processor
        self.image_encoder = image_encoder

    def preprocess_image(self, image):
        image = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return {"pixel_values": image.cuda()}

    def tokenize_text(self, text):
        inputs = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {"input_ids": inputs.input_ids.cuda()}

    def encode_image(self, image):
        preprocessed_image = self.preprocess_image(image)
        image_features = self.image_encoder(**preprocessed_image).image_embeds
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def encode_text(self, text):
        tokenized_text = self.tokenize_text(text)
        text_features = self.text_encoder(**tokenized_text).text_embeds
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    def compute_directional_similarity(self, img_feat_one, img_feat_two, text_feat_one, text_feat_two):
        sim_direction = F.cosine_similarity(img_feat_two - img_feat_one, text_feat_two - text_feat_one)
        return sim_direction

    def forward(self, image_one, image_two, caption_one, caption_two):
        img_feat_one = self.encode_image(image_one)
        img_feat_two = self.encode_image(image_two)
        text_feat_one = self.encode_text(caption_one)
        text_feat_two = self.encode_text(caption_two)
        clipi = F.cosine_similarity(img_feat_one.view(-1),img_feat_two.view(-1),dim=0)
        directional_similarity = self.compute_directional_similarity(
            img_feat_one, img_feat_two, text_feat_one, text_feat_two
        )
        return directional_similarity, clipi 

dir_similarity = DirectionalSimilarity(clipd_tokenizer, clipd_text_encoder, clipd_image_processor, clipd_image_encoder)

ground_images = open('dataset/synthetic_sdxl_images.txt', 'r')
ground_images = ground_images.readlines()
aerial_images_base_path = './results/syntheticsdxl/aerialbooth/' # Sample path 

iteration = 0
total_clip_score = 0 
total_aerialclip_score = 0 
total_sscd_score = 0 
total_dino_score = 0 
total_clipd_score = 0
total_aclipd_score = 0
total_clipi_score = 0

for data_path in ground_images[:]:
    print("We are at image: ", iteration)
    iteration = iteration + 1

    data_path = data_path[:-1]
    data_path = data_path.split('\t')
    print(data_path)
    prompt = data_path[1] 
    data_path = 'dataset/synthetic_sdxl_images/' + data_path[0] + '.png'
    data_name = data_path.split('/')
    data_name = data_name[-1]
    aerial_images_path = aerial_images_base_path + data_name
    aerial_images_path = glob.glob(aerial_images_path + '/*.png')

    max_clip_score = 0
    max_aerialclip_score = 0
    max_sscd_score = 0
    max_dino_score = 0
    max_clipd_score = 0
    max_aclipd_score = 0
    max_sscdclip_score = 0
    max_clipi_score = 0

    for aerial_image_path in aerial_images_path:

        # Load images

        image1 = PIL.Image.open(data_path).convert("RGB")
        image2 = PIL.Image.open(aerial_image_path).convert("RGB")

        # CLIP-T Metric : Image text similarity	

        og_text = prompt
        text = 'aerial view, ' + prompt
        clip_inputs = clip_processor(text=[text], images=image2, return_tensors="pt", padding=True)
        clip_inputs['pixel_values'] = clip_inputs['pixel_values'].cuda()
        clip_inputs['input_ids'] = clip_inputs['input_ids'].cuda()
        clip_inputs['attention_mask'] = clip_inputs['attention_mask'].cuda()
        outputs = clip_model(**clip_inputs)
        logits_per_image = outputs.logits_per_image.abs().cpu().detach().numpy()
        #print("CLIP Score is: ", logits_per_image)

        # CLIP-T Metric : Aerial text similarity 

        aerial_text = 'aerial view'
        clip_inputs = clip_processor(text=[aerial_text], images=image2, return_tensors="pt", padding=True)
        clip_inputs['pixel_values'] = clip_inputs['pixel_values'].cuda()
        clip_inputs['input_ids'] = clip_inputs['input_ids'].cuda()
        clip_inputs['attention_mask'] = clip_inputs['attention_mask'].cuda()
        aerial_outputs = clip_model(**clip_inputs)
        aerial_logits_per_image = aerial_outputs.logits_per_image.abs().cpu().detach().numpy()
        #print("CLIP Score is: ", logits_per_image)

        # DINO Score

        image_inputs1 = image_processor(image1, return_tensors="pt")
        image_inputs2 = image_processor(image2, return_tensors="pt")
        image_inputs1['pixel_values'] = image_inputs1['pixel_values'].cuda()
        image_inputs2['pixel_values'] = image_inputs2['pixel_values'].cuda()

        with torch.no_grad():
        	outputs1 = dino_model(**image_inputs1)
        	outputs2 = dino_model(**image_inputs2)

        last_hidden_states1 = outputs1.last_hidden_state # 1, 257, 768
        last_hidden_states2 = outputs2.last_hidden_state 

        dino_score = F.cosine_similarity(last_hidden_states1.view(-1), last_hidden_states2.view(-1), dim=0)
        #print("DINO Score is: ", dino_score)

        # SSCD Score

        image1_sscd = small_288(image1).unsqueeze(0)
        image2_sscd = small_288(image2).unsqueeze(0)
        image1_sscd = image1_sscd.cuda()
        image2_sscd = image2_sscd.cuda()
        embedding1 = sscd_model(image1_sscd)[0, :]
        embedding2 = sscd_model(image2_sscd)[0, :]
        sscd_score = F.cosine_similarity(embedding1.view(-1), embedding2.view(-1), dim=0).abs().detach().cpu().numpy()
        #print("SSCD Score is: ", sscd_score)

        # CLIP Directional similarity

        clip_direction, clipi = dir_similarity(image1, image2, og_text, text)
        aclip_direction, clipi = dir_similarity(image1, image2, og_text, aerial_text)
        #print("CLIP Directional similarity socre is: ", clip_direction)
        if(0.01 * logits_per_image + sscd_score > max_sscdclip_score):
            max_sscdclip_score = 0.01 * logits_per_image + sscd_score
            max_clip_score = logits_per_image
            max_aerialclip_score = aerial_logits_per_image
            max_dino_score = dino_score
            max_sscd_score = sscd_score
            max_clipd_score = clip_direction
            max_aclipd_score = aclip_direction
            max_clipi_score = clipi 

    total_clip_score = total_clip_score + max_clip_score
    total_aerialclip_score = total_aerialclip_score + max_aerialclip_score
    total_dino_score = total_dino_score + max_dino_score.abs().detach().cpu().numpy()
    total_sscd_score = total_sscd_score + max_sscd_score
    total_clipd_score = total_clipd_score + max_clipd_score.abs().detach().cpu().numpy()
    total_aclipd_score = total_aclipd_score + max_aclipd_score.abs().detach().cpu().numpy()
    total_clipi_score = total_clipi_score + max_clipi_score.abs().detach().cpu().numpy()
                                            


print("CLIP Score is: ", total_clip_score/(iteration))
print("Aerial CLIP Score is: ", total_aerialclip_score/(iteration))
print("SSCD Score is: ", total_sscd_score/(iteration))
print("DINO Score is: ", total_dino_score/(iteration))
print("CLIP-D Score is: ", total_clipd_score/(iteration))
print("A-CLIP-D Score is: ", total_aclipd_score/(iteration))
print("CLIP-I Score is: ", total_clipi_score/(iteration))
