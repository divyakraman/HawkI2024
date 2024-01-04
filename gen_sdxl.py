import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, cache_dir = 'dir_name'
)
pipe = pipe.to("cuda")

'''
prompt = "a cute bear cub exploring the forest"
image = pipe(prompt).images[0]
image.save('sample.png')
'''

data = open('dataset/synthetic_sdxl_images.txt', 'r')
data = data.readlines()

iteration = 0

for data1 in data:
    print("We are at image: ", iteration)
    iteration = iteration + 1

    data1 = data1[:-1]
    data1 = data1.split('\t')
    print(data1)
    prompt = data1[1] 
    data_path = data1[0]
    image = pipe(prompt).images[0]
    data_path = 'dataset/' + 'synthetic_sdxl_images/' + data_path + '.png'
    image.save(data_path)


