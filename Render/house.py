import pygame
from utils import *

class House:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.rect = pygame.Rect(x, y, HOUSE_SIZE, HOUSE_SIZE)
    
    def draw(self, screen):
        self.rect = pygame.Rect(self.x, self.y, HOUSE_SIZE, HOUSE_SIZE)
        pygame.draw.rect(screen, GRAY, self.rect)