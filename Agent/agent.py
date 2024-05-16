from Litter.clean_litter import LitterCleaner

class Agent(LitterCleaner):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.type = "agent"