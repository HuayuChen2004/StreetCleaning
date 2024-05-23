from Litter.clean_litter import LitterCleaner
import pygame
from Render.utils import *
from Agent.utils import *
import math
import random
import time

class Agent(LitterCleaner):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.type = "agent"
        self.x = x
        self.y = y
        self.speed = 1.5
        self.observed_pedestrian_positions = {}  # Dictionary to store the position history of each pedestrian
        self.observed_car_positions = {}  # Dictionary to store the position history of each car
        self.direction = random.choice(DIRECTIONS)
        self.vision_range = []
        self.is_crossing = False
        self.crossing_direction = None
        # print("self.vision range: ", self.vision_range)
        # exit()

    def draw(self, screen):
        rect = pygame.Rect(self.x, self.y, AGENT_SIZE, AGENT_SIZE)
        pygame.draw.rect(screen, BLUE, rect)

    def update_pedestrian_positions(self, pedestrians):
        for pedestrian in pedestrians:
            if pedestrian not in self.observed_pedestrian_positions:
                self.observed_pedestrian_positions[pedestrian] = [(pedestrian.x, pedestrian.y)]
            # if not self.observed_pedestrian_positions[pedestrian]:
            #     self.observed_pedestrian_positions[pedestrian].append((pedestrian.x, pedestrian.y))
            
            if (pedestrian.x, pedestrian.y) != self.observed_pedestrian_positions[pedestrian][-1]:
                self.observed_pedestrian_positions[pedestrian].append((pedestrian.x, pedestrian.y))
                # Keep only the last few positions to limit memory usage
                if len(self.observed_pedestrian_positions[pedestrian]) > 5:
                    self.observed_pedestrian_positions[pedestrian].pop(0)

    def update_car_positions(self, cars):
        for car in cars:
            if car not in self.observed_car_positions:
                self.observed_car_positions[car] = [(car.x, car.y)]
            # if not self.observed_car_positions[car]:
            #     self.observed_car_positions[car].append((car.x, car.y))
            if (car.x, car.y) != self.observed_car_positions[car][-1]:
                self.observed_car_positions[car].append((car.x, car.y))
                # Keep only the last few positions to limit memory usage
                if len(self.observed_car_positions[car]) > 5:
                    self.observed_car_positions[car].pop(0) 
                
    def observe_pedestrians(self, pedestrians):
        observed = [pedestrian for vision_range in self.vision_range for pedestrian in pedestrians 
                    if vision_range[0][0] <= pedestrian.x < vision_range[0][1] and 
                    vision_range[1][0] <= pedestrian.y < vision_range[1][1] or 
                    vision_range[0][0] <= pedestrian.x + PEDESTRIAN_SIZE[0] < vision_range[0][1] and 
                    vision_range[1][0] <= pedestrian.y < vision_range[1][1] or
                    vision_range[0][0] <= pedestrian.x < vision_range[0][1] and 
                    vision_range[1][0] <= pedestrian.y + PEDESTRIAN_SIZE[1] < vision_range[1][1] or
                    vision_range[0][0] <= pedestrian.x + PEDESTRIAN_SIZE[0] < vision_range[0][1] and 
                    vision_range[1][0] <= pedestrian.y + PEDESTRIAN_SIZE[1] < vision_range[1][1]]
        self.update_pedestrian_positions(observed)
        return observed
    
    def observe_cars(self, cars):
        observed = [car for vision_range in self.vision_range for car in cars 
                    if vision_range[0][0] <= car.x < vision_range[0][1] and 
                    vision_range[1][0] <= car.y < vision_range[1][1] or 
                    vision_range[0][0] <= car.x + CAR_WIDTH < vision_range[0][1] and 
                    vision_range[1][0] <= car.y < vision_range[1][1] or
                    vision_range[0][0] <= car.x < vision_range[0][1] and 
                    vision_range[1][0] <= car.y + CAR_WIDTH < vision_range[1][1] or
                    vision_range[0][0] <= car.x + CAR_WIDTH < vision_range[0][1] and 
                    vision_range[1][0] <= car.y + CAR_WIDTH < vision_range[1][1]]
        self.update_car_positions(observed)
        return observed
    
    def observe_litters(self, litters):
        # print("self.vision range: ", self.vision_range)
        # exit()
        observed = [litter for vision_range in self.vision_range for litter in litters 
                if vision_range[0][0] <= litter.x < vision_range[0][1] and 
                vision_range[1][0] <= litter.y < vision_range[1][1] and litter.weight > 0]
        print("observed litters: ", observed)
        return observed

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
        if self.is_time_sufficient(litter, observed_pedestrians, observed_cars):
            print("time is sufficient for cleaning this litter!")
            self.clean_litter(litter, observed_pedestrians, observed_cars, observed_houses)
        # else:
        #     self.wander(traffic_light)
        # 先不动试试看

    def touch_road_intersection(self):
        road_xranges = [(x + HOUSE_SIZE, x + HOUSE_SIZE * 2) for x in range(0, WIDTH - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        road_yranges = [(y + HOUSE_SIZE, y + HOUSE_SIZE * 2) for y in range(0, HEIGHT - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        for xrange in road_xranges:
            if xrange[0] < self.x + AGENT_SIZE and self.x < xrange[1]:
                for yrange in road_yranges:
                    if yrange[0] < self.y + AGENT_SIZE and self.y < yrange[1]:
                        return True
        return False
    
    def at_road_intersection(self):
        road_xranges = [(x + HOUSE_SIZE, x + HOUSE_SIZE * 2) for x in range(0, WIDTH - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        road_yranges = [(y + HOUSE_SIZE, y + HOUSE_SIZE * 2) for y in range(0, HEIGHT - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        for xrange in road_xranges:
            if xrange[0] <= self.x <= xrange[1] - AGENT_SIZE:
                for yrange in road_yranges:
                    if yrange[0] <= self.y <= yrange[1] - AGENT_SIZE:
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
        elif not direction:
            pass
        else:
            raise ValueError("unusual direction!")

        # Check if the agent is out of the screen
        if self.x < 0:
            self.x = WIDTH - self.speed
        elif self.x > WIDTH - self.speed:
            self.x = 0
        if self.y < 0:
            self.y = HEIGHT - self.speed
        elif self.y > HEIGHT - self.speed:
            self.y = 0

    def wander(self, traffic_light):
        # at the road intersection
        # follow the light
        
        if self.at_road_intersection():
            if traffic_light == 'horizontal':
                if self.direction not in ['left', 'right']:
                    self.direction = random.choice(['left', 'right'])
            elif traffic_light =='vertical':
                if self.direction not in ['up', 'down']:
                    self.direction = random.choice(['up', 'down'])
            self.move(self.direction)
        elif self.touch_road_intersection():
            self.move(self.direction)
        else:
            self.move_on_side()


    def move_on_side(self):
        road_xranges = [(x + HOUSE_SIZE, x + HOUSE_SIZE * 2) for x in range(0, WIDTH - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        road_yranges = [(y + HOUSE_SIZE, y + HOUSE_SIZE * 2) for y in range(0, HEIGHT - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]

        for xrange in road_xranges:
            if xrange[0] <= self.x <= xrange[1] - AGENT_SIZE:
                if self.x - xrange[0] < ROAD_WIDTH / 2 - CAR_WIDTH / 2 - AGENT_SIZE or \
                xrange[1] - self.x - AGENT_SIZE < ROAD_WIDTH / 2 - CAR_WIDTH / 2 - AGENT_SIZE:
                    # Not in the center, continue moving in the current direction
                    if self.direction not in ['up', 'down']:
                        self.direction = random.choice(['up', 'down'])
                    self.move(self.direction)
                    # else:
                    #     # Adjust direction to horizontal if needed
                    #     self.direction = 'left' if self.x - xrange[0] <= ROAD_WIDTH / 2 else 'right'
                    #     self.move(self.direction)
                else:
                    # Adjust direction to stay off the center
                    if self.direction not in ['left', 'right']:
                        self.direction = 'left' if self.x - xrange[0] <= ROAD_WIDTH / 2 else 'right'
                    # new_x = self.x + (self.speed if self.direction == 'right' else -self.speed)
                    # if xrange[0] <= new_x <= xrange[1] - AGENT_SIZE:
                    #     self.move(self.direction)
                    # else:
                    #     self.direction = 'left' if self.direction == 'right' else 'right'
                    #     self.move(self.direction)
                    self.move(self.direction)
                return

        for yrange in road_yranges:
            if yrange[0] <= self.y <= yrange[1] - AGENT_SIZE:
                if self.y - yrange[0] < ROAD_WIDTH / 2 - CAR_WIDTH / 2 - AGENT_SIZE or \
                yrange[1] - self.y - AGENT_SIZE < ROAD_WIDTH / 2 - CAR_WIDTH / 2 - AGENT_SIZE:
                    # Not in the center, continue moving in the current direction
                    if self.direction not in ['left', 'right']:
                        self.direction = random.choice(['left', 'right'])
                    self.move(self.direction)
                    # else:
                    #     # Adjust direction to vertical if needed
                    #     self.direction = 'up' if self.y - yrange[0] <= ROAD_WIDTH / 2 else 'down'
                    #     self.move(self.direction)
                else:
                    # Adjust direction to stay off the center
                    if self.direction not in ['up', 'down']:
                        self.direction = 'up' if self.y - yrange[0] <= ROAD_WIDTH / 2 else 'down'
                    # new_y = self.y + (self.speed if self.direction == 'down' else -self.speed)
                    # if yrange[0] <= new_y <= yrange[1] - AGENT_SIZE:
                    #     self.move(self.direction)
                    # else:
                    #     self.direction = 'up' if self.direction == 'down' else 'down'
                    #     self.move(self.direction)
                    self.move(self.direction)
                return

        raise ValueError("Agent should be on the road")
    
    def need_to_cross_road(self, litter):
        road_xranges = [(x + HOUSE_SIZE, x + HOUSE_SIZE * 2) for x in range(0, WIDTH - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        road_yranges = [(y + HOUSE_SIZE, y + HOUSE_SIZE * 2) for y in range(0, HEIGHT - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        for xrange in road_xranges:
            if xrange[0] <= litter.x <= xrange[1] - ROAD_WIDTH/2 - LITTER_SIZE and \
                xrange[0] + ROAD_WIDTH/2 <= self.x <= xrange[1] - AGENT_SIZE or \
                xrange[0] <= self.x <= xrange[1] - ROAD_WIDTH/2 - AGENT_SIZE and \
                xrange[0] + ROAD_WIDTH/2 <= litter.x <= xrange[1] - LITTER_SIZE:
                return True
        for yrange in road_yranges:
            if yrange[0] <= litter.y <= yrange[1] - ROAD_WIDTH/2 - LITTER_SIZE and \
                yrange[0] + ROAD_WIDTH/2 <= self.y <= yrange[1] - AGENT_SIZE or \
                yrange[0] <= self.y <= yrange[1] - ROAD_WIDTH/2 - AGENT_SIZE and \
                yrange[0] + ROAD_WIDTH/2 <= litter.y <= yrange[1] - LITTER_SIZE:
                return True
        return False
    
    def find_crossing_direction(self, litter):
        road_xranges = [(x + HOUSE_SIZE, x + HOUSE_SIZE * 2) for x in range(0, WIDTH - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        road_yranges = [(y + HOUSE_SIZE, y + HOUSE_SIZE * 2) for y in range(0, HEIGHT - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        for xrange in road_xranges:
            if xrange[0] <= litter.x <= xrange[1] - ROAD_WIDTH/2 - LITTER_SIZE and \
                xrange[0] + ROAD_WIDTH/2 <= self.x <= xrange[1] - AGENT_SIZE:
                return 'left' 
            if xrange[0] <= self.x <= xrange[1] - ROAD_WIDTH/2 - AGENT_SIZE and \
                xrange[0] + ROAD_WIDTH/2 <= litter.x <= xrange[1] - LITTER_SIZE:
                return 'right'
        for yrange in road_yranges:
            if yrange[0] <= litter.y <= yrange[1] - ROAD_WIDTH/2 - LITTER_SIZE and \
                yrange[0] + ROAD_WIDTH/2 <= self.y <= yrange[1] - AGENT_SIZE:
                return 'up'
            if yrange[0] <= self.y <= yrange[1] - ROAD_WIDTH/2 - AGENT_SIZE and \
                yrange[0] + ROAD_WIDTH/2 <= litter.y <= yrange[1] - LITTER_SIZE:
                return 'down'
        return None
    
    def has_crossed(self, litter):
        road_xranges = [(x + HOUSE_SIZE, x + HOUSE_SIZE * 2) for x in range(0, WIDTH - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        road_yranges = [(y + HOUSE_SIZE, y + HOUSE_SIZE * 2) for y in range(0, HEIGHT - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
        for xrange in road_xranges:
            if xrange[0] <= litter.x <= xrange[1] - ROAD_WIDTH/2 - LITTER_SIZE and \
                xrange[0] <= self.x <= xrange[1] - ROAD_WIDTH/2 - AGENT_SIZE:
                return True
            if xrange[0] + ROAD_WIDTH/2 <= litter.x <= xrange[1] - LITTER_SIZE and \
                xrange[0] + ROAD_WIDTH/2 <= self.x <= xrange[1] - AGENT_SIZE:
                return True
        for yrange in road_yranges:
            if yrange[0] <= litter.y <= yrange[1] - ROAD_WIDTH/2 - LITTER_SIZE and \
                yrange[0] <= self.y <= yrange[1] - ROAD_WIDTH/2 - AGENT_SIZE:
                return True
            if yrange[0] + ROAD_WIDTH/2 <= litter.y <= yrange[1] - LITTER_SIZE and \
                yrange[0] + ROAD_WIDTH/2 <= self.y <= yrange[1] - AGENT_SIZE:
                return True
        return False

    def clean_litter(self, litter, pedestrians, cars, houses):
        # add street lights: version 2
        # restrict agent's direction to 'up', 'down', 'left', 'right'
        # TODO
        if self.is_crossing:
            self.move(self.crossing_direction)
            if self.has_crossed(litter):
                self.is_crossing = False
                self.crossing_direction = None
        elif self.need_to_cross_road(litter):
            print("need to cross!")
            self.is_crossing = True
            self.crossing_direction = self.find_crossing_direction(litter)
            self.move(self.crossing_direction)
        else:
            self.direction = self.find_available_way(litter, self.observe_pedestrians(pedestrians), self.observe_cars(cars), self.observe_houses(houses))
        # print("direction: ", self.direction)
            if self.direction:
                self.move(self.direction)
        if self.x == litter.x and self.y == litter.y and litter.weight > 0:
            litter.clean()
    
    def start_cleaning(self, litters, pedestrians, cars, agents, houses, traffic_light):
        # print("litters: ", litters)
        # print("observed litters: ", self.observe_litters(litters))
        # start_time = time.time()
        # print("observed_litters: ", self.observe_litters(litters))
        # print("observed_agents: ", self.observe_agents(agents))
        self.vision_range = []
        for agent in self.observe_agents(agents):
            self.vision_range.append([[agent.x - VISION_SIZE, agent.x + AGENT_SIZE + VISION_SIZE], 
                                        [agent.y - VISION_SIZE, agent.y + AGENT_SIZE + VISION_SIZE]])
        if not self.observe_litters(litters):
            self.wander(traffic_light)

        else:
            closet_litter = self.find_closest_litter(litters)
            print("closet litter: ", closet_litter)
            observed_pedestrians = self.observe_pedestrians(pedestrians)
            observed_cars = self.observe_cars(cars)
            observed_houses = self.observe_houses(houses)
            self.safe_clean(closet_litter, observed_pedestrians, observed_cars, observed_houses, traffic_light)

    def observe_agents(self, agents):
        self.vision_range = [[[self.x - VISION_SIZE, self.x + AGENT_SIZE + VISION_SIZE],
                                [self.y - VISION_SIZE, self.y + AGENT_SIZE + VISION_SIZE]]]
        observed = [agent for vision_range in self.vision_range for agent in agents 
                if vision_range[0][0] <= agent.x < vision_range[0][1] and 
                vision_range[1][0] <= agent.y < vision_range[1][1] or 
                vision_range[0][0] <= agent.x + AGENT_SIZE < vision_range[0][1] and 
                vision_range[1][0] <= agent.y < vision_range[1][1] or
                vision_range[0][0] <= agent.x < vision_range[0][1] and 
                vision_range[1][0] <= agent.y + AGENT_SIZE < vision_range[1][1] or
                vision_range[0][0] <= agent.x + AGENT_SIZE < vision_range[0][1] and 
                vision_range[1][0] <= agent.y + AGENT_SIZE < vision_range[1][1]]
        # print("observed agents: ", observed)
        return observed
    
    def observe_houses(self, houses):
        return [house for vision_range in self.vision_range for house in houses 
                if vision_range[0][0] <= house.x < vision_range[0][1] and 
                vision_range[1][0] <= house.y < vision_range[1][1] or 
                vision_range[0][0] <= house.x + HOUSE_SIZE < vision_range[0][1] and 
                vision_range[1][0] <= house.y < vision_range[1][1] or
                vision_range[0][0] <= house.x < vision_range[0][1] and 
                vision_range[1][0] <= house.y + HOUSE_SIZE < vision_range[1][1] or
                vision_range[0][0] <= house.x + HOUSE_SIZE < vision_range[0][1] and 
                vision_range[1][0] <= house.y + HOUSE_SIZE < vision_range[1][1]]
    
    def calculate_pedestrian_speed(self, pedestrian):
        positions = self.observed_pedestrian_positions.get(pedestrian, [])
        if len(positions) < 2:
            return 0  # Not enough data to calculate speed
        # Calculate speed based on the last two positions
        (x1, y1), (x2, y2) = positions[-2], positions[-1]
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return distance  # Assuming each update represents a constant time step

    def calculate_car_speed(self, car):
        positions = self.observed_car_positions.get(car, [])
        if len(positions) < 2:
            return 0  # Not enough data to calculate speed
        # Calculate speed based on the last two positions
        (x1, y1), (x2, y2) = positions[-2], positions[-1]
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return distance  # Assuming each update represents a constant time step

    def calculate_litter_distance(self, litter):
        dx = litter.x - self.x
        dy = litter.y - self.y
        return math.sqrt(dx**2 + dy**2)

    def calculate_litters_distances(self, litters):
        return [self.calculate_litter_distance(litter) for litter in litters]

    def find_closest_litter(self, litters):
        distances = self.calculate_litters_distances(litters)
        return litters[distances.index(min(distances))] if distances else None

    def is_time_sufficient(self, litter, observed_pedestrians, observed_cars):
        time_to_clean = litter.weight

        pedestrian = is_litter_in_pedestrian_path(observed_pedestrians, litter)
        if pedestrian:
            speed = self.calculate_pedestrian_speed(pedestrian)
            time_to_reach_litter = calculate_time_to_reach(litter, pedestrian, speed)
            return time_to_reach_litter > time_to_clean

        # If litter is not in pedestrian path, check if it is in car path
        car = is_litter_in_car_path(observed_cars, litter)
        if car:
            speed = self.calculate_car_speed(car)  # Assuming cars have a speed attribute
            time_to_reach_litter = calculate_time_to_reach(litter, car, speed)
            return time_to_reach_litter > time_to_clean
        
        return True
    
    def find_available_way(self, litter, pedestrians, cars, houses):
        directions = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }

        def is_collision(new_x, new_y):
            agent_rect = pygame.Rect(new_x, new_y, AGENT_SIZE, AGENT_SIZE)

            for pedestrian in pedestrians:
                if agent_rect.colliderect(pedestrian.rect):
                    return True

            for car in cars:
                if agent_rect.colliderect(car.rect):
                    return True

            for house in houses:
                if agent_rect.colliderect(house.rect):
                    return True

            return False

        if litter.x > self.x and litter.y > self.y:
            preferred_directions = ['down', 'right']
        elif litter.x < self.x and litter.y < self.y:
            preferred_directions = ['up', 'left']
        elif litter.x > self.x and litter.y < self.y:
            preferred_directions = ['up', 'right']
        else:  # litter.x < x and litter.y > y
            preferred_directions = ['down', 'left']

        for direction in preferred_directions:
            dx, dy = directions[direction]
            new_x, new_y = self.x + dx, self.y + dy
            if not is_collision(new_x, new_y):
                return direction

        # 如果所有偏好的方向都被阻挡，尝试其他方向
        for direction in directions:
            dx, dy = directions[direction]
            new_x, new_y = self.x + dx, self.y + dy
            if not is_collision(new_x, new_y):
                return direction

        return None  # 如果所有方向都被阻挡，返回 None