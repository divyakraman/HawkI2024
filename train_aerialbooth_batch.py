import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

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

#export CUDA_VISIBLE_DEVICES=0,1
#echo $CUDA_VISIBLE_DEVICES

has_cuda = torch.cuda.is_available()

device = torch.device('cpu' if not has_cuda else 'cuda')
torch.hub.set_dir('/scratch0/')

data = open('dataset/synthetic_sdxl_images.txt', 'r')
data = data.readlines()

iteration = 0

for data_path in data[:]:
    print("We are at image: ", iteration)
    iteration = iteration + 1

    data_path = data_path[:-1]
    data_path = data_path.split('\t')
    print(data_path)
    prompt = data_path[1] #+ ', aerial view'
    data_path = 'dataset/synthetic_sdxl_images/' + data_path[0] + '.png'
    data_name = data_path.split('/')
    data_name = data_name[-1]

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
            safety_checker=None,
        use_auth_token=False,
        custom_pipeline='./models/aerialbooth', cache_dir = './huggingface_models/',
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    ).to(device)
    
    generator = torch.Generator("cuda").manual_seed(0)
    seed = 0

    init_image = PIL.Image.open(data_path).convert("RGB")
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
            model_fine_tuning_optimization_steps = 500, 
            image_hom=image_hom) # 1000, 500

    savedir = './results/syntheticsdxl/aerialdiffusionv2_mi1e-5/' + data_name
    os.makedirs(savedir, exist_ok=True)

    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-5)
    image = res.images[0]
    image.save(savedir+'/result1.png')
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-5)
    image = res.images[0]
    image.save(savedir+'/result2.png')
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-5)
    image = res.images[0]
    image.save(savedir+'/result3.png')
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-5)
    image = res.images[0]
    image.save(savedir+'/result4.png')
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-5)
    image = res.images[0]
    image.save(savedir+'/result5.png')

    savedir = './results/syntheticsdxl/aerialdiffusionv2_mi1e-6/' + data_name
    os.makedirs(savedir, exist_ok=True)

    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6)
    image = res.images[0]
    image.save(savedir+'/result1.png')
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6)
    image = res.images[0]
    image.save(savedir+'/result2.png')
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6)
    image = res.images[0]
    image.save(savedir+'/result3.png')
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6)
    image = res.images[0]
    image.save(savedir+'/result4.png')
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=1e-6)
    image = res.images[0]
    image.save(savedir+'/result5.png')

    savedir = './results/syntheticsdxl/aerialdiffusionv2_nomi/' + data_name
    os.makedirs(savedir, exist_ok=True)

    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=0)
    image = res.images[0]
    image.save(savedir+'/result1.png')
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=0)
    image = res.images[0]
    image.save(savedir+'/result2.png')
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=0)
    image = res.images[0]
    image.save(savedir+'/result3.png')
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=0)
    image = res.images[0]
    image.save(savedir+'/result4.png')
    res = pipe(alpha=0.1, guidance_scale=7.5, num_inference_steps=50, mi_lr=0)
    image = res.images[0]
    image.save(savedir+'/result5.png')
