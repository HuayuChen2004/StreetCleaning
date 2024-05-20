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
        litter_weight = random.randint(1, 3)
        return Litter(x, y, litter_weight)