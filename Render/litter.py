from utils import *
import pygame

class Litter:
    def __init__(self, x, y, weight=1):
        self.rect = pygame.Rect(x, y, LITTER_SIZE, LITTER_SIZE)
        
    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, self.rect)