from Render.litter import Litter as RenderLitter

class Litter:
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.name = "Litter"
        self.weight = weight

    def __str__(self):
        return f"{self.name} ({self.weight} kg)"
    
    def draw(self, screen):
        RenderLitter(self.x, self.y, self.weight).draw(screen)