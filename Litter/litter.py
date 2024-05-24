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
        return f"{self.name} ({self.weight}kg) at ({self.x}, {self.y})"
    
    def __repr__(self) -> str:
        return f"{self.name} ({self.weight}kg) at ({self.x}, {self.y})"
    
    def draw(self, screen):
        if self.weight != 0:
            pygame.draw.rect(screen, BLACK, self.rect)

    def clean(self):
        self.weight -= 1

    def __hash__(self):
        return hash((self.x, self.y, self.weight))
    
    def __eq__(self, value: object) -> bool:
        if isinstance(value, Litter):
            return self.x == value.x and self.y == value.y and self.weight == value.weight
        return False
