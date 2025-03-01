#region import
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
from tkinter import Tk, filedialog

def select_file(title="Select a file"):
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    return file_path

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

sam2_checkpoint = "./sam2_master/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)
#endregion

pygame.init()

#region 窗口
WindowL,WindowH = 800, 800
screen = pygame.display.set_mode((WindowL,WindowH))
pygame.display.set_caption('AAAAAA')
#endregion

background = pygame.image.load('MyImages/Sam.jpg')
dirt_img = pygame.image.load('MyImages/IMG_9516.jpeg')


#region sam2 predictor的变量们
Pos_position = []
Neg_position = []

input_point = []
input_label = []
#endregion

#region 角色图片导入
IMAGE = pygame.image.load("MyImages/ren.jpg")
image_width, image_height = IMAGE.get_size()
SAM2image = Image.open("MyImages/ren.jpg")
SAM2image_width, SAM2image_height= SAM2image.size
SAM2image = SAM2image.convert("RGB")
#endregion


#region py图位置
image_ratio = image_width / image_height
if  image_width > WindowL:
    new_width = WindowL
    new_height = int(WindowL / image_ratio)
elif image_height > WindowH:
    new_height = WindowH
    new_width = int(WindowH * image_ratio)
else:
    new_width, new_height = image_width, image_height
IMAGE = pygame.transform.smoothscale(IMAGE, (new_width, new_height))
#endregion

#region 蒙板图位置
SAM2image_ratio = SAM2image_width / SAM2image_height
if  SAM2image_width > WindowL:
    Snew_width = WindowL
    Snew_height = int(WindowL / SAM2image_ratio)
elif SAM2image_height > WindowH:
    Snew_height = WindowH
    Snew_width = int(WindowH * SAM2image_ratio)
else:
    Snew_width, Snew_height = SAM2image_width, SAM2image_height
SAM2image = SAM2image.resize((Snew_width, Snew_height))
SAM2image = SAM2image.rotate(270, expand=True)
SAM2image = SAM2image.transpose(Image.FLIP_LEFT_RIGHT)

#endregion

#region sam2的东西，不要动。
red_mask_surface = pygame.Surface((new_width, new_height), pygame.SRCALPHA)
predictor.set_image(SAM2image)
def predict():
    global ismove, player
    #删除
    if event.type == pygame.KEYDOWN and event.key == 8:
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
            pygame.draw.circle(screen,(255,0,0),pos,5)
    if len(Pos_position) > 0:
        for pos in Pos_position:
                pygame.draw.circle(screen, (233, 0, 255), pos, 5)

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
        red_mask = np.zeros_like(SAM2image)
        red_mask[:,:,0] = masks*255

        red_mask_surface = pygame.surfarray.make_surface(red_mask)
        red_mask_surface.set_alpha(128)
        screen.blit(red_mask_surface, (0, 0))

    if event.type == pygame.KEYDOWN and event.key == K_RETURN:
        ismove = True

        Npimage = np.array(SAM2image.convert("RGB"))
        h, w = Npimage.shape[:2]
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[masks[0] > 0, :3] = Npimage[masks[0] > 0]
        result[..., 3] = np.where(masks[0] > 0, 255, 0)
        cutout,h,w = cutoutobject(result)
        player = pygame.image.frombuffer(cutout.tobytes(), (w, h), 'RGBA').convert_alpha()
        player = pygame.transform.smoothscale(player, (w/h*200,200))
#endregion

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

#字体
font = pygame.font.Font(None, 50)

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



#region 按钮
all_buttons = []
class Button:

    def __init__(self, text, x, y, width, height, color, hover_color):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        self.hover_color = hover_color
        self.is_pressed = False
        all_buttons.append(self)

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        if self.rect.collidepoint(mouse_pos):
            pygame.draw.rect(surface, self.hover_color, self.rect)
        else:
            pygame.draw.rect(surface, self.color, self.rect)
        text_surface = font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def check_click(self, event):
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:  # 左键点击
            if self.rect.collidepoint(event.pos):
                self.is_pressed = True

    def reset(self):
        self.is_pressed = False

def check_all_buttons(event):
    for button in all_buttons:
        button.check_click(event)

def draw_all_buttons(surface, font):
    for button in all_buttons:
        button.draw(surface)
#endregion

player_HIT = pygame.USEREVENT + 1
def player_statu(playerbox, bullet):
    if playerbox.colliderect(bullet):
        pygame.event.post((pygame.event.Event(player_HIT)))

ismove = False
player = None

x = 0
y = 0

button_start = Button("start", 300, 200, 200, 80, GREEN, BLUE)
button_end = Button("end",20,20,20,20, RED, BLUE)

#碰撞箱
playerbox = pygame.Rect(0, 0, new_width - 10, new_height-10 )
bullet = pygame.Rect(100, 100, 50, 50)

#region 玩家信息
player_health = 10
last_hurt_time = 0
hurt_cooldown = 3000
#end region

#region 地图
tile_size = 160
def draw_grid():
    for line in range(0,6):
        pygame.draw.line(screen,(255,255,255),(0,line * tile_size),(WindowL, line * tile_size))
        pygame.draw.line(screen, (255, 255, 255), (line * tile_size,0), (line * tile_size,WindowH))

class World():
    def __init__(self, data):
        self.tile_list= []
        row_count = 0
        for row in data:
            col_count =0
            for tile in row:
                if tile == 1:
                    img =  pygame.transform.scale(dirt_img, (tile_size, tile_size))
                    img_rect = img.get_rect()
                    img_rect.x = col_count*tile_size
                    img_rect.y = row_count * tile_size
                    tile = (img,img_rect)
                    self.tile_list.append(tile)
                col_count+=1
            row_count+=1
    def draw(self):
        for tile in self.tile_list:
            screen.blit(tile[0],tile[1])

world_data =[
[0,0,0,0,0],
[0,0,0,0,0],
[0,0,0,0,0],
[0,0,0,0,0],
[1,1,1,1,1],
]
world = World(world_data)
#endregion

#大写Player是类，小写player是图片


class Player():
    def __init__(self,x,y):

        self.image = player
        self.rect =self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.vel_y = 0

    def update(self):
        dx = 0
        dy = 0
        keys_pressed = pygame.key.get_pressed()
        if keys_pressed[pygame.K_a]:
            dx -= 3
        if keys_pressed[pygame.K_d]:
            dx += 3
        if keys_pressed[pygame.K_SPACE]:
            self.vel_y=-15

        self.vel_y += 5
        if self.vel_y > 10:
            self.vel_y = 10
        dy += self.vel_y

        self.rect.x += dx
        self.rect.y += dy

        screen.blit(self.image, self.rect)




clock = pygame.time.Clock()

running = True
while running:
    fps = clock.get_fps()

    screen.fill((255, 0, 255))
    for event in pygame.event.get():
        if event.type == pygame.QUIT or button_end.is_pressed:
            running = False
        check_all_buttons(event)

    if not ismove and button_start.is_pressed:
        if button_start in all_buttons:
            all_buttons.remove(button_start)
        screen.blit(IMAGE, (0, 0))
        predict()
        if player != None:
            Player1 = Player(0, 0)


    if ismove:

        screen.blit(background, (0, 0))

        current_time = pygame.time.get_ticks()
        #状态
        player_statu(playerbox, bullet)

        #生命值
        UI_player_health = font.render("health" + str(player_health), True, (0,0,0))
        screen.blit(UI_player_health, (600, 10))
        #子弹
        pygame.draw.rect(screen, RED, bullet)

        Player1.update()

        if event.type == player_HIT and current_time - last_hurt_time >= hurt_cooldown:
            last_hurt_time = current_time
            player_health -= 1

        if player_health <= 0:
            last_hurt_time = current_time
            Dead_scence = font.render("you lose", True, (0,0,0))
            screen.fill((255, 255, 255))
            screen.blit(Dead_scence, (400, 400))


    draw_grid()
    world.draw()
    draw_all_buttons(screen, font)


    pygame.display.update()

pygame.quit()
sys.exit()

