from Render.utils import *

def calculate_time_to_reach(self, litter, moving_object, speed):
    if moving_object.direction == "down":
        distance = litter.y - moving_object.y
    elif moving_object.direction == "up":
        distance = moving_object.y - litter.y
    elif moving_object.direction == "left":
        distance = moving_object.x - litter.x
    elif moving_object.direction == "right":
        distance = litter.x - moving_object.x
    else:
        distance = float('inf')  # in case direction is unknown, set a large distance

    if speed > 0:
        return distance / speed
    else:
        return float('inf')  # if speed is zero or negative, set a large time

def is_litter_in_pedestrian_path(self, observed_pedestrians, litter):
    for pedestrian in observed_pedestrians:
        if (litter.y > pedestrian.y and 
            ((pedestrian.x <= litter.x < pedestrian.x + PEDESTRIAN_SIZE) or
            (pedestrian.x <= litter.x + LITTER_SIZE <= pedestrian.x + PEDESTRIAN_SIZE)) and 
            pedestrian.direction == "down"):
            return pedestrian
        
        if (litter.y < pedestrian.y and 
            ((pedestrian.x <= litter.x < pedestrian.x + PEDESTRIAN_SIZE) or
            (pedestrian.x <= litter.x + LITTER_SIZE <= pedestrian.x + PEDESTRIAN_SIZE)) and 
            pedestrian.direction == "up"):
            return pedestrian
        
        if (litter.x < pedestrian.x and 
            ((pedestrian.y <= litter.y < pedestrian.y + PEDESTRIAN_SIZE) or
            (pedestrian.y <= litter.y + LITTER_SIZE <= pedestrian.y + PEDESTRIAN_SIZE)) and 
            pedestrian.direction == "left"):
            return pedestrian
        
        if (litter.x > pedestrian.x and 
            ((pedestrian.y <= litter.y < pedestrian.y + PEDESTRIAN_SIZE) or
            (pedestrian.y <= litter.y + LITTER_SIZE <= pedestrian.y + PEDESTRIAN_SIZE)) and 
            pedestrian.direction == "right"):
            return pedestrian
    return None

def is_litter_in_car_path(self, observed_cars, litter):
    for car in observed_cars:
        if (car.direction == "down" and 
            litter.y > car.y and 
            ((car.x <= litter.x < car.x + CAR_WIDTH) or
            (car.x <= litter.x + LITTER_SIZE <= car.x + CAR_WIDTH))):
            return car

        if (car.direction == "up" and 
            litter.y < car.y and 
            ((car.x <= litter.x < car.x + CAR_WIDTH) or
            (car.x <= litter.x + LITTER_SIZE <= car.x + CAR_WIDTH))):
            return car

        if (car.direction == "right" and 
            litter.x > car.x and 
            ((car.y <= litter.y < car.y + CAR_WIDTH) or
            (car.y <= litter.y + LITTER_SIZE <= car.y + CAR_WIDTH))):
            return car

        if (car.direction == "left" and 
            litter.x < car.x and 
            ((car.y <= litter.y < car.y + CAR_WIDTH) or
            (car.y <= litter.y + LITTER_SIZE <= car.y + CAR_WIDTH))):
            return car

    return None

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