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
        pass