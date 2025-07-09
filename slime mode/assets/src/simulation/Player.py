import numpy as np
import pygame

#用来处理黏菌的，记得改名
class git Simulation:
    def __init__(self,seed,width,height,BG_size):
        self.seed = seed
        self.color_threshold = 60

        self.grid_width = width
        self.grid_height = height

        self.grid_position = #算最佳位置

    #这里在把seed这个surface放进更大的透明图片里，太困了之后做
    def embed_seed(self,event,game):
        if (event.type == pygame.MOUSEBUTTONUP):
            grid = pygame.Surface((self.grid_width,self.grid_height), pygame.SRCALPHA)
            seed_width, seed_height = self.seed.get_size()
            grid_width, grid_height = (self.grid_width,self.grid_height)

            mouse_x, mouse_y = pygame.mouse.get_pos()

            grid_x, grid_y = self.grid_position

            if grid_x <= mouse_x < grid_x + grid_width and grid_y <= mouse_y < grid_height + grid_x:
                paste_x = mouse_x - seed_width // 2
                paste_y = mouse_y - seed_height // 2

                # 防止种子超网格）
                paste_x = max(0, min(paste_x, grid_width - seed_width))
                paste_y = max(0, min(paste_y, grid_height - seed_height))
                # 把种子图贴进 grid
                grid.blit(self.seed, (paste_x, paste_y))
                game.state = "playing"
                return grid#这个grid就是要处理的图片
        return None

    def color_dis(self,color1,color2):
        return np.linalg.norm(np.array(color1) - np.array(color2))

    def pic_evolve(self,alive_grid,color_grid,):



