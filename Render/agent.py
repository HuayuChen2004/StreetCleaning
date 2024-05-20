import pygame
from utils import *


class Agent:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen):
        rect = pygame.Rect(self.x, self.y, AGENT_SIZE, AGENT_SIZE)
        pygame.draw.rect(screen, BLUE, rect)