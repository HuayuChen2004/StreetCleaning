from Litter.clean_litter import LitterCleaner
import pygame
from Render.utils import *
from Agent.utils import *
import math
import random

class Agent(LitterCleaner):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.type = "agent"
        self.x = x
        self.y = y
        self.speed = 2
        self.observed_pedestrian_positions = {}  # Dictionary to store the position history of each pedestrian
        self.observed_car_positions = {}  # Dictionary to store the position history of each car
        self.direction = random.choice(DIRECTIONS)
        self.vision_range = [[[self.x - VISION_SIZE, self.x + AGENT_SIZE + VISION_SIZE], [self.y - VISION_SIZE, self.y + AGENT_SIZE + VISION_SIZE]]]

    def draw(self, screen):
        rect = pygame.Rect(self.x, self.y, AGENT_SIZE, AGENT_SIZE)
        pygame.draw.rect(screen, BLUE, rect)

    def update_pedestrian_positions(self, pedestrians):
        for pedestrian in pedestrians:
            if pedestrian not in self.observed_pedestrian_positions:
                self.observed_pedestrian_positions[pedestrian] = []
            if (pedestrian.x, pedestrian.y) != self.observed_pedestrian_positions[pedestrian][-1]:
                self.observed_pedestrian_positions[pedestrian].append((pedestrian.x, pedestrian.y))
                # Keep only the last few positions to limit memory usage
                if len(self.observed_pedestrian_positions[pedestrian]) > 5:
                    self.observed_pedestrian_positions[pedestrian].pop(0)
    
    def update_car_positions(self, cars):
        for car in cars:
            if car not in self.observed_car_positions:
                self.observed_car_positions[car] = []
            if (car.x, car.y) != self.observed_car_positions[car][-1]:
                self.observed_car_positions[car].append((car.x, car.y))
                # Keep only the last few positions to limit memory usage
                if len(self.observed_car_positions[car]) > 5:
                    self.observed_car_positions[car].pop(0)
                
    def observe_pedestrians(self, pedestrians):
        observed = [pedestrian for vision_range in self.vision_range for pedestrian in pedestrians 
                    if pedestrian.x in range(vision_range[0][0], vision_range[0][1]) and 
                    pedestrian.y in range(vision_range[1][0], vision_range[1][1]) or 
                    pedestrian.x + PEDESTRIAN_SIZE[0] in range(vision_range[0][0], vision_range[0][1]) and 
                    pedestrian.y in range(vision_range[1][0], vision_range[1][1]) or
                    pedestrian.x in range(vision_range[0][0], vision_range[0][1]) and 
                    pedestrian.y + PEDESTRIAN_SIZE[1] in range(vision_range[1][0], vision_range[1][1]) or
                    pedestrian.x + PEDESTRIAN_SIZE[0] in range(vision_range[0][0], vision_range[0][1]) and 
                    pedestrian.y + PEDESTRIAN_SIZE[1] in range(vision_range[1][0], vision_range[1][1])]
        self.update_pedestrian_positions(observed)
        return observed
    
    def observe_cars(self, cars):
        observed = [car for vision_range in self.vision_range for car in cars 
                    if car.x in range(vision_range[0][0], vision_range[0][1]) and 
                    car.y in range(vision_range[1][0], vision_range[1][1]) or 
                    car.x + CAR_WIDTH in range(vision_range[0][0], vision_range[0][1]) and 
                    car.y in range(vision_range[1][0], vision_range[1][1]) or
                    car.x in range(vision_range[0][0], vision_range[0][1]) and 
                    car.y + CAR_WIDTH in range(vision_range[1][0], vision_range[1][1]) or
                    car.x + CAR_WIDTH in range(vision_range[0][0], vision_range[0][1]) and 
                    car.y + CAR_WIDTH in range(vision_range[1][0], vision_range[1][1])]
        self.update_car_positions(observed)
        return observed
    
    def observe_litters(self, litters):
        return [litter for vision_range in self.vision_range for litter in litters 
                if litter.x in range(vision_range[0][0], vision_range[0][1]) and 
                litter.y in range(vision_range[1][0], vision_range[1][1]) and litter.weight > 0]

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

    def safe_clean(self, litter, observed_pedestrians, observed_cars, observed_houses, traffic_light):
        # Check if there is enough time to clean the litter
        if is_time_sufficient(litter, observed_pedestrians, observed_cars):
            self.clean_litter(litter, observed_pedestrians, observed_cars, observed_houses)
        else:
            self.wander(traffic_light)

    def at_road_intersection(self):
        road_xranges = [(x + HOUSE_SIZE, x + HOUSE_SIZE * 2) for x in range(0, WIDTH - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        road_yranges = [(y + HOUSE_SIZE, y + HOUSE_SIZE * 2) for y in range(0, HEIGHT - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        for xrange in road_xranges:
            if xrange[0] <= self.x and self.x + AGENT_SIZE <= xrange[1]:
                for yrange in road_yranges:
                    if yrange[0] <= self.y and self.y + AGENT_SIZE <= yrange[1]:
                        return True
        return False
    
    def move(self, direction):
        if direction == 'up':
            self.y -= self.speed
        elif direction == 'down':
            self.y += self.speed
        elif direction == 'left':
            self.x -= self.speed
        elif direction == 'right':
            self.x += self.speed
        else:
            raise ValueError("direction should be set (up down left right)")

    def wander(self, traffic_light):
        # at the road intersection
        # follow the light
        print("wander!")
        if self.at_road_intersection():
            print("at road intersection!")
            if traffic_light == 'horizontal':
                self.direction = random.choice(['left', 'right'])
            elif traffic_light =='vertical':
                self.direction = random.choice(['up', 'down'])
            self.move(self.direction)
        else:
            self.move_on_side()


    def move_on_side(self):
        print("move on side!")
        road_xranges = [(x + HOUSE_SIZE, x + HOUSE_SIZE * 2) for x in range(0, WIDTH - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        road_yranges = [(y + HOUSE_SIZE, y + HOUSE_SIZE * 2) for y in range(0, HEIGHT - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        for xrange in road_xranges:
            if xrange[0] <= self.x and self.x + AGENT_SIZE <= xrange[1]:
                if ROAD_WIDTH/2 - CAR_WIDTH/2 - AGENT_SIZE <= self.x - xrange[0] < xrange[1] - self.x - AGENT_SIZE:
                    self.direction = 'left'
                elif ROAD_WIDTH/2 - CAR_WIDTH/2 < xrange[1] - self.x < self.x - xrange[0] + AGENT_SIZE:
                    self.direction = 'right'
                elif self.x - xrange[0] <= PEDESTRIAN_SIZE[0]:
                    self.direction = 'right'
                elif xrange[1] - self.x - AGENT_SIZE <= PEDESTRIAN_SIZE[0]:
                    self.direction = 'left'
                else:
                    self.direction = random.choice(['up', 'down'])
                self.move(self.direction)
                break
        else:
            for yrange in road_yranges:
                if yrange[0] <= self.y and self.y + AGENT_SIZE <= yrange[1]:
                    if ROAD_WIDTH/2 - CAR_WIDTH/2 - AGENT_SIZE <= self.y - yrange[0] < yrange[1] - self.y - AGENT_SIZE:
                        self.direction = 'up'
                    elif ROAD_WIDTH/2 - CAR_WIDTH/2 < yrange[1] - self.y < self.y - yrange[0] + AGENT_SIZE:
                        self.direction = 'down'
                    elif self.y - yrange[0] <= PEDESTRIAN_SIZE[1]:
                        self.direction = 'down'
                    elif yrange[1] - self.y - AGENT_SIZE <= PEDESTRIAN_SIZE[1]:
                        self.direction = 'up'
                    else:
                        self.direction = random.choice(['left', 'right'])
                    self.move(self.direction)
                    break
            else:
                raise ValueError("agent should be on the road")

    
    def clean_litter(self, litter, pedestrians, cars, houses):
        # add street lights: version 2
        # restrict agent's direction to 'up', 'down', 'left', 'right'
        # TODO
        self.direction = find_available_way(litter, self.x, self.y, self.observe_pedestrians(pedestrians), self.observe_cars(cars), self.observe_houses(houses))
        self.move(self.direction)
        if self.x == litter.x and self.y == litter.y and litter.weight > 0:
            litter.clean()
    
    def start_cleaning(self, litters, pedestrians, cars, agents, houses, traffic_light):
        if not self.observe_litters(litters) and not self.observe_agents(agents):
            self.wander(traffic_light)
        elif self.observe_agents(agents):
            self.vision_range = [[[self.x - VISION_SIZE, self.x + AGENT_SIZE + VISION_SIZE], [self.y - VISION_SIZE, self.y + AGENT_SIZE + VISION_SIZE]]]
            for agent in self.observe_agents(agents):
                self.vision_range.append([[agent.x - VISION_SIZE, agent.x + AGENT_SIZE + VISION_SIZE], [agent.y - VISION_SIZE, agent.y + AGENT_SIZE + VISION_SIZE]])
        if not self.observe_litters(litters):
            self.wander(traffic_light)
        else:
            closet_litter = find_closest_litter(litters)
            observed_pedestrians = self.observe_pedestrians(pedestrians)
            observed_cars = self.observe_cars(cars)
            observed_houses = self.observe_houses(houses)
            self.safe_clean(closet_litter, observed_pedestrians, observed_cars, observed_houses, traffic_light)

    def observe_agents(self, agents):
        return [agent for vision_range in self.vision_range for agent in agents 
                if agent.x in range(vision_range[0][0], vision_range[0][1]) and 
                agent.y in range(vision_range[1][0], vision_range[1][1]) or 
                agent.x + AGENT_SIZE in range(vision_range[0][0], vision_range[0][1]) and 
                agent.y in range(vision_range[1][0], vision_range[1][1]) or
                agent.x in range(vision_range[0][0], vision_range[0][1]) and 
                agent.y + AGENT_SIZE in range(vision_range[1][0], vision_range[1][1]) or
                agent.x + AGENT_SIZE in range(vision_range[0][0], vision_range[0][1]) and 
                agent.y + AGENT_SIZE in range(vision_range[1][0], vision_range[1][1])]
    
    def observe_houses(self, houses):
        return [house for vision_range in self.vision_range for house in houses
                if house.x in range(vision_range[0][0], vision_range[0][1]) and 
                house.y in range(vision_range[1][0], vision_range[1][1]) or 
                house.x + HOUSE_SIZE in range(vision_range[0][0], vision_range[0][1]) and 
                house.y in range(vision_range[1][0], vision_range[1][1]) or
                house.x in range(vision_range[0][0], vision_range[0][1]) and 
                house.y + HOUSE_SIZE in range(vision_range[1][0], vision_range[1][1]) or
                house.x + HOUSE_SIZE in range(vision_range[0][0], vision_range[0][1]) and 
                house.y + HOUSE_SIZE in range(vision_range[1][0], vision_range[1][1])]