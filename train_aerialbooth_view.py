import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import requests
from PIL import Image
from io import BytesIO
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
import PIL
import cv2
import numpy as np 
from scipy import ndimage 
#import matplotlib.pyplot as plt 

has_cuda = torch.cuda.is_available()

device = torch.device('cpu' if not has_cuda else 'cuda')
torch.hub.set_dir('/scratch0/')
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-base",
        safety_checker=None,
    use_auth_token=False,
    custom_pipeline='./models/aerialbooth_viewarg', cache_dir = 'dir_name',
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
).to(device)
#'''
generator = torch.Generator("cuda").manual_seed(0)
seed = 0

prompt = "A coastal lighthouse with a spiral staircase"

init_image = PIL.Image.open('dataset/synthetic_sdxl_images/architectures34.png').convert("RGB")
init_image = init_image.resize((512, 512))

image_hom = init_image.resize((256, 256))
image_hom = np.array(image_hom)
image_hom = PIL.Image.fromarray(image_hom)
image_hom = image_hom.resize((256, 256))
image_hom = np.array(image_hom)
H = 256
W = 256 
pts1 = np.float32([[0,0],[H,0],[H,W],[0,W]])
pts2 = np.float32([[0,W],[H,0],[H,W],[0,2*W]])
M1 = cv2.getPerspectiveTransform(pts1,pts2)
#M1_inv = np.linalg.inv(M1)
image_hom = cv2.warpPerspective(image_hom,M1,(2*W,2*H))
image_hom = ndimage.rotate(image_hom, -45)
#plt.imsave('abc.png', image_hom)
image_hom = PIL.Image.fromarray(image_hom)
image_hom = image_hom.resize((512, 512))

res = pipe.train(
    prompt,
    image=init_image,
    generator=generator, text_embedding_optimization_steps = 1000,
        model_fine_tuning_optimization_steps = 500, image_hom=image_hom) # 1000, 500


savedir = './results/viewsyn/samples/architectures34/aerialdiffusionv2mi1e-6'
os.makedirs(savedir, exist_ok=True)

eval_prompt = 'bottom view, ' + prompt
#eval_prompt = prompt + ', seen from the bottom'
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/bottom1.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/bottom2.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/bottom3.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/bottom4.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/bottom5.png')

eval_prompt = 'side view, ' + prompt
#eval_prompt = prompt + ', seen from the side'
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/side1.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/side2.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/side3.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/side4.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/side5.png')

eval_prompt = 'back view, ' + prompt
#eval_prompt = prompt + ', seen from the back'
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/back1.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/back2.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/back3.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/back4.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/back5.png')

eval_prompt = 'aerial view, ' + prompt
#eval_prompt = prompt + ', seen from the top'
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/aerial1.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/aerial2.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/aerial3.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/aerial4.png')
res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6, eval_prompt = eval_prompt)
image = res.images[0]
image.save(savedir+'/aerial5.png')
