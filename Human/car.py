from Litter.generate_litter import LitterGenerator

class Car(LitterGenerator):
    def __init__(self, name):
        super().__init__(name)
        self.name = name
        self.type = "car"