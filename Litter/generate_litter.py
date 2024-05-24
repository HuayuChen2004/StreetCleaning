import pygame
from Render.utils import *
import random
from Litter.litter import Litter

road_xranges = [(x + HOUSE_SIZE, x + HOUSE_SIZE * 2) for x in range(0, WIDTH - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
road_yranges = [(y + HOUSE_SIZE, y + HOUSE_SIZE * 2) for y in range(0, HEIGHT - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]


class LitterGenerator:
    def __init__(self, x, y, direction):
        self.x = x
        self.y = y
        self.direction = direction

    def generate_litter(self, x, y):
        if not self.is_on_road(x, y):
            return None
        litter_weight = random.randint(5, 10)
        return Litter(x, y, litter_weight)
    
    def is_on_road(self, x, y):
        for x_range in road_xranges:
            if x_range[0] <= x and x + LITTER_SIZE <= x_range[1]:
                return True
        for y_range in road_yranges:
            if y_range[0] <= y and y + LITTER_SIZE <= y_range[1]:
                return True
        return False