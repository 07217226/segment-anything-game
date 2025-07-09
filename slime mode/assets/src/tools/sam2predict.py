# region import
import pygame
import sys
import time
from pygame.locals import *

import os
from six import moves

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

sam2_checkpoint = "/Users/kolmio/PycharmProjects/SAM2kolmio/sam2_master/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)
# endregiongit push


pygame.init()

Pos_position = []
Neg_position = []

input_point = []
input_label = []

#操作图片缩放的
def load_resize_image_pygame(imagePath,BG_SIZE):
    WindowL, WindowH=BG_SIZE
    image=pygame.image.load(imagePath)
    image_width, image_height = image.get_size()
    image_ratio = image_width / image_height
    if image_width > WindowL:
        new_width = WindowL
        new_height = int(WindowL / image_ratio)
    elif image_height > WindowH:
        new_height = WindowH
        new_width = int(WindowH * image_ratio)
    else:
        new_width, new_height = image_width, image_height
    originalimage = pygame.transform.smoothscale(image, (new_width, new_height))
    return originalimage

def load_resize_image_pil(imagePath,BG_SIZE):
    WindowL, WindowH = BG_SIZE
    image = Image.open(imagePath)
    image = image.convert('RGB')
    image_width, image_height = image.size
    image_ratio = image_width / image_height
    if image_width > WindowL:
        new_width = WindowL
        new_height = int(WindowL / image_ratio)
    elif image_height > WindowH:
        new_height = WindowH
        new_width = int(WindowH * image_ratio)
    else:
        new_width, new_height = image_width, image_height
    sam2image = image.resize((new_width, new_height)).rotate(270, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
    return sam2image

#去掉不透明的
def cutoutobject(result):
    first_h_bound = 0
    for i in range(result.shape[0]):
        if np.sum(result[i, :, :]) > 0:
            first_h_bound = i
            break
    second_h_bound = result.shape[0]
    for i in range(result.shape[0] - 1, 0, -1):
        if np.sum(result[i, :, :]) > 0:
            second_h_bound = i
            break

    first_w_bound = 0
    for i in range(result.shape[1]):
        if np.sum(result[:, i, :]) > 0:
            first_w_bound = i
            break
    second_w_bound = result.shape[1]
    for i in range(result.shape[1] - 1, 0, -1):
        if np.sum(result[:, i, :]) > 0:
            second_w_bound = i
            break
    cutout=result[first_h_bound:second_h_bound, first_w_bound:second_w_bound, :].copy()
    cutout = np.transpose(cutout,(1,0,2))
    return cutout, cutout.shape[0], cutout.shape[1]

#切割图片的全流程
def predict(game,screen,original_image,sam2image,event):
    red_mask_surface = pygame.Surface(sam2image.size, pygame.SRCALPHA)
    predictor.set_image(sam2image)
    screen.blit(original_image, (0, 0))
    #删除
    if event.type == pygame.KEYDOWN and event.key == pygame.K_DELETE:
        Pos_position.clear()
        Neg_position.clear()
        input_point.clear()
        input_label.clear()
        print("clear")

    # 正点
    if (event.type == pygame.MOUSEBUTTONUP and not pygame.key.get_mods() & pygame.KMOD_CTRL
            and pygame.mouse.get_pos() not in Pos_position):
        Pos_position.append(pygame.mouse.get_pos())
        (x,y) = pygame.mouse.get_pos()
        input_point.append([y,x])
        input_label.append(1)
        print(pygame.mouse.get_pos())

    # 负点
    if (event.type == pygame.MOUSEBUTTONUP and pygame.key.get_mods() & KMOD_CTRL
            and pygame.mouse.get_pos() not in Neg_position):
        Neg_position.append(pygame.mouse.get_pos())
        (x, y) = pygame.mouse.get_pos()
        input_point.append([y, x])
        input_label.append(0)
        print(pygame.mouse.get_pos())

    if len(Neg_position)>0:
        for pos in Neg_position:
            pygame.draw.circle(screen,(255,0,0),pos,2)
    if len(Pos_position) > 0:
        for pos in Pos_position:
            pygame.draw.circle(screen, (233, 0, 255), pos, 2)

    if len(input_point) > 0:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        red_mask = np.zeros_like(sam2image)
        red_mask[:,:,0] = masks*255

        red_mask_surface = pygame.surfarray.make_surface(red_mask)
        red_mask_surface.set_alpha(128)
        screen.blit(red_mask_surface, (0, 0))


    if len(input_point)>0 and event.type == pygame.KEYDOWN and event.key == K_RETURN:
        Npimage = np.array(sam2image.convert("RGB"))
        h, w = Npimage.shape[:2]
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[masks[0] > 0, :3] = Npimage[masks[0] > 0]
        result[..., 3] = np.where(masks[0] > 0, 255, 0)
        cutout,h,w = cutoutobject(result)
        playerimage = pygame.image.frombuffer(cutout.tobytes(), (w, h), 'RGBA').convert_alpha()

        game.state = "embedding"
        #这里把所有透明的部分都切掉了，还缩小了，返回了一个黏菌基座图片
        return playerimage

    return None

