import pygame
import sys
pygame.init()

from tools.sam2predict import predict,load_resize_image_pygame,load_resize_image_pil
from simulation.Player import *
BG_SIZE=(800,800)
#临时
global playerimage
class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode(BG_SIZE)
        pygame.display.set_caption('图片黏菌')
        #状态
        self.state = "start"
        #黏菌基座的类
        self.simulation = Simulation(None)

    def handle_event(self, event):
        #菜单
        if self.state=="start":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                self.state="predicting"
                print(self.state)
        if self.state=="predicting":
            #resize初始图片
            original_image = load_resize_image_pygame("/Users/kolmio/PycharmProjects/SAM2kolmio/slime mode/assets/MyImages/ren.jpg",BG_SIZE)
            sam2image = load_resize_image_pil("/Users/kolmio/PycharmProjects/SAM2kolmio/slime mode/assets/MyImages/ren.jpg",BG_SIZE)
            #获得黏菌基座
            self.simulation.seed = predict(self,self.screen,original_image,sam2image,event)
            print(self.state)

        if self.state=="embedding":
            Simulation.embed_seed(self.simulation,event,self)


        #游戏
        elif self.state=="playing":

            self.screen.blit(self.simulation.seed,(0,0))
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.state="paused"
                    print(self.state)
                elif  event.key == pygame.K_ESCAPE:
                    self.state="game_over"
                    print(self.state)
        #暂停
        elif self.state=="paused":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.state="playing"
                print(self.state)

        elif self.state=="game_over":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                sys.exit()

    def draw(self):
        #好像没啥用但姑且留着
        A=1


    def run(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            else:
                self.handle_event(event)

        self.draw()
        pygame.display.flip()  # 或者 pygame.display.update()


game = Game()

running = True
while running:
    Game.run(game)
