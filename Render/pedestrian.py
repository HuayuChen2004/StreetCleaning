import pygame
import random
from utils import *
from Litter.litter import Litter
from Litter.generate_litter import LitterGenerator

road_xranges = [(x + HOUSE_SIZE, x + HOUSE_SIZE * 2) for x in range(0, WIDTH, HOUSE_SIZE * 2)]
road_yranges = [(y + HOUSE_SIZE, y + HOUSE_SIZE * 2) for y in range(0, WIDTH, HOUSE_SIZE * 2)]

class Pedestrian(LitterGenerator):
    def __init__(self, x, y, direction, litter_prob=0.1):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, *PEDESTRIAN_SIZE)
        self.speed = random.choice([1.25, 0.75])
        self.direction = direction  # 行人的移动方向，可以是 'up'、'down'、'left' 或 'right'
        self.moving = True
        self.litter_prob = litter_prob
    
    def move(self, traffic_light, left_time):
        if not self.moving:
            return

        next_position = self.calculate_next_position()

        if self.direction in ['up', 'down']:
            road_ranges = road_yranges
            road_width = road_ranges[0][1] - road_ranges[0][0]
            time_to_cross = (road_width + PEDESTRIAN_SIZE[1] + self.distance_to_house(self.x, self.y)) // self.speed + 2
            light_direction = 'vertical'
        else:
            road_ranges = road_xranges
            road_width = road_ranges[0][1] - road_ranges[0][0]
            time_to_cross = (road_width + PEDESTRIAN_SIZE[0] + self.distance_to_house(self.x, self.y)) // self.speed + 2
            light_direction = 'horizontal'

        if any(r[0] <= next_position <= r[1] for r in road_ranges):
            if traffic_light == light_direction and left_time > time_to_cross:
                self.update_position()
            else:
                for r in road_ranges:
                    if r[0] <= next_position <= r[1]:
                        if self.direction in ['up', 'down']:
                            self.y = r[1] if self.direction == 'up' else r[0] - PEDESTRIAN_SIZE[1]
                        else:
                            self.x = r[1] if self.direction == 'left' else r[0] - PEDESTRIAN_SIZE[0]
                        self.moving = False
        else:
            self.update_position()
            self.reset_position_if_needed()

    def calculate_next_position(self):
        if self.direction == 'up':
            return self.y - self.speed
        elif self.direction == 'down':
            return self.y + self.speed + PEDESTRIAN_SIZE[1]
        elif self.direction == 'left':
            return self.x - self.speed
        elif self.direction == 'right':
            return self.x + self.speed + PEDESTRIAN_SIZE[0]

    def update_position(self):
        if self.direction == 'up':
            self.y -= self.speed
        elif self.direction == 'down':
            self.y += self.speed
        elif self.direction == 'left':
            self.x -= self.speed
        elif self.direction == 'right':
            self.x += self.speed

    def reset_position_if_needed(self):
        if self.direction == 'up' and self.y < -PEDESTRIAN_SIZE[1]:
            self.y = HEIGHT
        elif self.direction == 'down' and self.y > HEIGHT:
            self.y = -PEDESTRIAN_SIZE[1]
        elif self.direction == 'left' and self.x < -PEDESTRIAN_SIZE[0]:
            self.x = WIDTH
        elif self.direction == 'right' and self.x > WIDTH:
            self.x = -PEDESTRIAN_SIZE[0]

    def distance_to_house(self, x, y):
        if self.direction == 'up':
            for i in range(len(road_yranges)-1):
                if road_yranges[i][1] <= y < road_yranges[i+1][0]:
                    return y - road_yranges[i][1]
        elif self.direction == 'down':
            for i in range(len(road_yranges)-1):
                if road_yranges[i][1] <= y < road_yranges[i+1][0]:
                    return road_yranges[i+1][0] - y
        elif self.direction == 'left':
            for i in range(len(road_xranges)-1):
                if road_xranges[i][1] <= x < road_xranges[i+1][0]:
                    return x - road_xranges[i][1]
        elif self.direction == 'right':
            for i in range(len(road_xranges)-1):
                if road_xranges[i][1] <= x < road_xranges[i+1][0]:
                    return road_xranges[i+1][0] - x
        return 0

    def draw(self, screen):
        self.rect = pygame.Rect(self.x, self.y, *PEDESTRIAN_SIZE)
        pygame.draw.rect(screen, RED, self.rect)

    def generate_litter(self, x, y):
        p = random.random()
        if p < self.litter_prob:
            litter = super().generate_litter(x, y)
            # litter.draw(screen)
            return litter
        return None

