from Litter.clean_litter import LitterCleaner
import pygame
from Render.utils import *
from Agent.utils import *

class Agent(LitterCleaner):
    def __init__(self, x, y):
        super().__init__()
        self.type = "agent"
        self.x = x
        self.y = y
        self.observed_pedestrian_positions = {}  # Dictionary to store the position history of each pedestrian
        self.observed_car_positions = {}  # Dictionary to store the position history of each car

    def draw(self, screen):
        rect = pygame.Rect(self.x, self.y, AGENT_SIZE, AGENT_SIZE)
        pygame.draw.rect(screen, BLUE, rect)

    def move(self, dx, dy):
        self.x += dx
        self.y += dy

    def observe_pedestrians(self, pedestrians):
        vision_range = [[self.x - VISION_SIZE, self.x + AGENT_SIZE + VISION_SIZE], [self.y - VISION_SIZE, self.y + AGENT_SIZE + VISION_SIZE]]
        observed = [pedestrian for pedestrian in pedestrians if pedestrian.x in range(vision_range[0][0], vision_range[0][1]) and pedestrian.y in range(vision_range[1][0], vision_range[1][1]) or 
                    pedestrian.x + PEDESTRIAN_SIZE[0] in range(vision_range[0][0], vision_range[0][1]) and pedestrian.y in range(vision_range[1][0], vision_range[1][1]) or
                    pedestrian.x in range(vision_range[0][0], vision_range[0][1]) and pedestrian.y + PEDESTRIAN_SIZE[1] in range(vision_range[1][0], vision_range[1][1]) or
                    pedestrian.x + PEDESTRIAN_SIZE[0] in range(vision_range[0][0], vision_range[0][1]) and pedestrian.y + PEDESTRIAN_SIZE[1] in range(vision_range[1][0], vision_range[1][1])]
        self.update_pedestrian_positions(observed)
        return observed
    
    def observe_cars(self, cars):
        vision_range = [[self.x - VISION_SIZE, self.x + AGENT_SIZE + VISION_SIZE], [self.y - VISION_SIZE, self.y + AGENT_SIZE + VISION_SIZE]]
        observed = [car for car in cars if car.x in range(vision_range[0][0], vision_range[0][1]) and car.y in range(vision_range[1][0], vision_range[1][1]) or 
                    car.x + CAR_WIDTH in range(vision_range[0][0], vision_range[0][1]) and car.y in range(vision_range[1][0], vision_range[1][1]) or
                    car.x in range(vision_range[0][0], vision_range[0][1]) and car.y + CAR_WIDTH in range(vision_range[1][0], vision_range[1][1]) or
                    car.x + CAR_WIDTH in range(vision_range[0][0], vision_range[0][1]) and car.y + CAR_WIDTH in range(vision_range[1][0], vision_range[1][1])]
        self.update_car_positions(observed)
        return observed
    
    def observe_litters(self, litters):
        vision_range = [[self.x - VISION_SIZE, self.x + AGENT_SIZE + VISION_SIZE], [self.y - VISION_SIZE, self.y + AGENT_SIZE + VISION_SIZE]]
        return [litter for litter in litters if litter.x in range(vision_range[0][0], vision_range[0][1]) and litter.y in range(vision_range[1][0], vision_range[1][1])]

    def update_observed_pedestrian_positions(self, pedestrians):
        for observed_pedestrian in self.observe_pedestrians(pedestrians):
            if observed_pedestrian not in self.observed_pedestrian_positions:
                self.observed_pedestrian_positions[observed_pedestrian] = []
            self.observed_pedestrian_positions[observed_pedestrian].append((observed_pedestrian.x, observed_pedestrian.y))
            # Keep only the last few positions to limit memory usage
            if len(self.observed_pedestrian_positions[observed_pedestrian]) > 5:
                self.observed_pedestrian_positions[observed_pedestrian].pop(0)
    
    def update_observed_car_positions(self, cars):
        for observed_car in self.observe_cars(cars):
            if observed_car not in self.observed_car_positions:
                self.observed_car_positions[observed_car] = []
            self.observed_car_positions[observed_car].append((observed_car.x, observed_car.y))
            # Keep only the last few positions to limit memory usage
            if len(self.observed_car_positions[observed_car]) > 5:
                self.observed_car_positions[observed_car].pop(0)

    def is_time_sufficient(self, litter, observed_pedestrians, observed_cars):
        time_to_clean = litter.weight

        pedestrian = is_litter_in_pedestrian_path(observed_pedestrians, litter)
        if pedestrian:
            speed = calculate_pedestrian_speed(pedestrian)
            time_to_reach_litter = calculate_time_to_reach(litter, pedestrian, speed)
            return time_to_reach_litter > time_to_clean

        # If litter is not in pedestrian path, check if it is in car path
        car = is_litter_in_car_path(observed_cars, litter)
        if car:
            speed = calculate_car_speed(car)  # Assuming cars have a speed attribute
            time_to_reach_litter = calculate_time_to_reach(litter, car, speed)
            return time_to_reach_litter > time_to_clean
        
        return True
    
    def safe_clean(self, litter, observed_pedestrians, observed_cars):
        if self.is_time_sufficient(litter, observed_pedestrians, observed_cars):
            self.clean_litter(litter)
            return True
        return False
    
    def clean_litter(self, litter):
        # move to litter

    
