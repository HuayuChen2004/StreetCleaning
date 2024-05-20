import pygame
from utils import *
import random
from Litter.generate_litter import LitterGenerator

road_xranges = [(x + HOUSE_SIZE, x + HOUSE_SIZE * 2) for x in range(0, WIDTH, HOUSE_SIZE * 2)]
road_yranges = [(y + HOUSE_SIZE, y + HOUSE_SIZE * 2) for y in range(0, WIDTH, HOUSE_SIZE * 2)]

class Car(LitterGenerator):
    def __init__(self, x, y, direction: str, litter_prob=0.1):
        self.direction = direction
        self.litter_prob = litter_prob
        if self.direction in ['left', 'right']:
            self.rect = pygame.Rect(x, y, CAR_LENGTH, CAR_WIDTH)
        elif self.direction in ['up', 'down']:
            self.rect = pygame.Rect(x, y, CAR_WIDTH, CAR_LENGTH)
        else:
            raise ValueError("direction should be set (left right up down)")
        self.speed = random.choice([2,5, 5])
    
    def move(self, traffic_light, left_time):
        if self.direction == 'right':
            next_position = self.rect.x + self.speed
            if any(x_range[0] <= next_position + CAR_LENGTH <= x_range[1] for x_range in road_xranges):
                if traffic_light == 'horizontal' and left_time > (road_xranges[0][1] - road_xranges[0][0] + CAR_LENGTH) / self.speed:
                    self.rect.x += self.speed
                else:
                    # 确保车辆停在路口前
                    for x_range in road_xranges:
                        if x_range[0] <= next_position + CAR_LENGTH <= x_range[1]:
                            self.rect.x = x_range[0] - CAR_LENGTH
            else:
                self.rect.x += self.speed
                if self.rect.x > WIDTH:
                    self.rect.x = -CAR_LENGTH
        elif self.direction == 'left':
            next_position = self.rect.x - self.speed
            if any(x_range[0] <= next_position <= x_range[1] for x_range in road_xranges):
                if traffic_light == 'horizontal' and left_time > (road_xranges[0][1] - road_xranges[0][0] + CAR_LENGTH) / self.speed:
                    self.rect.x -= self.speed
                else:
                    # 确保车辆停在路口前
                    for x_range in road_xranges:
                        if x_range[0] <= next_position <= x_range[1]:
                            self.rect.x = x_range[1]
            else:
                self.rect.x -= self.speed
                if self.rect.x < -CAR_LENGTH:
                    self.rect.x = WIDTH
        elif self.direction == 'down':
            next_position = self.rect.y + self.speed
            if any(y_range[0] <= next_position + CAR_LENGTH <= y_range[1] for y_range in road_yranges):
                if traffic_light == 'vertical' and left_time > (road_yranges[0][1] - road_yranges[0][0] + CAR_LENGTH) / self.speed:
                    self.rect.y += self.speed
                else:
                    # 确保车辆停在路口前
                    for y_range in road_yranges:
                        if y_range[0] <= next_position + CAR_LENGTH <= y_range[1]:
                            self.rect.y = y_range[0] - CAR_LENGTH
            else:
                self.rect.y += self.speed
                if self.rect.y > HEIGHT:
                    self.rect.y = -CAR_LENGTH
        elif self.direction == 'up':
            next_position = self.rect.y - self.speed
            if any(y_range[0] <= next_position <= y_range[1] for y_range in road_yranges):
                if traffic_light == 'vertical' and left_time > (road_yranges[0][1] - road_yranges[0][0] + CAR_LENGTH) / self.speed:
                    self.rect.y -= self.speed
                else:
                    # 确保车辆停在路口前
                    for y_range in road_yranges:
                        if y_range[0] <= next_position <= y_range[1]:
                            self.rect.y = y_range[1]
            else:
                self.rect.y -= self.speed
                if self.rect.y < -CAR_LENGTH:
                    self.rect.y = HEIGHT
    
    def draw(self, screen):
        pygame.draw.rect(screen, RED, self.rect)

    def generate_litter(self, x, y):
        p = random.random()
        if p < self.litter_prob:
            litter = super().generate_litter(x, y)
            # litter.draw(screen)
            return litter
        return None


