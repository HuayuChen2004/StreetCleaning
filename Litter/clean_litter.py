from Render.utils import *

class LitterCleaner:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def is_litter_covered(self, litter):
        return litter.x in range(self.x, self.x + AGENT_SIZE - LITTER_SIZE) and litter.y in range(self.y, self.y + AGENT_SIZE - LITTER_SIZE)

    def clean(self, litter):
        pass

    def visible_litters(self, litters):
        vision_range = [[self.x - VISION_SIZE, self.x + AGENT_SIZE + VISION_SIZE], [self.y - VISION_SIZE, self.y + AGENT_SIZE + VISION_SIZE]]
        return [litter for litter in litters if litter.x in range(vision_range[0][0], vision_range[0][1]) and litter.y in range(vision_range[1][0], vision_range[1][1])]
    
    
