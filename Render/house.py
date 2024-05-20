import pygame
from utils import *

class House:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, HOUSE_SIZE, HOUSE_SIZE)
    
    def draw(self, screen):
        pygame.draw.rect(screen, GRAY, self.rect)