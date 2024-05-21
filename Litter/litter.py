from Render.utils import *
import pygame

class Litter:
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.name = "Litter"
        self.weight = weight
        self.rect = pygame.Rect(x, y, LITTER_SIZE, LITTER_SIZE)

    def __str__(self):
        return f"{self.name} ({self.weight} kg)"
    
    def draw(self, screen):
        if self.weight != 0:
            pygame.draw.rect(screen, BLACK, self.rect)

    def clean(self):
        self.weight -= 1
