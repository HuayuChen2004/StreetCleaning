from Litter.generate_litter import LitterGenerator

class Pedestrian(LitterGenerator):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.type = "pedestrian"