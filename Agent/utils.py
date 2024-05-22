from Render.utils import *
import math
import random
import pygame

# calculation, decision, and observation functions

def calculate_time_to_reach(litter, moving_object, speed):
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

def is_litter_in_pedestrian_path(observed_pedestrians, litter):
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

def is_litter_in_car_path(observed_cars, litter):
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



# def find_available_way(litter, x, y, pedestrians, cars, houses):
#     if litter.x > x and litter.y > y:
#         for direction in ['down', 'right']:
#             new_x, new_y = x, y
#             if direction == 'up':
#                 new_y -= 1
#             elif direction == 'down':
#                 new_y += 1
#             elif direction == 'left':
#                 new_x -= 1
#             elif direction == 'right':
#                 new_x += 1

#             # Check if the new position will collide with a pedestrian, car, or house
#             for pedestrian in pedestrians:
#                 if pedestrian.x == new_x and pedestrian.y == new_y:
#                     break
#             else:
#                 for car in cars:
#                     if car.x == new_x and car.y == new_y:
#                         break
#                 else:
#                     for house in houses:
#                         if house.x == new_x and house.y == new_y:
#                             break
#                     else:
#                         # If the new position does not collide with any obstacles, return this direction
#                         return direction
#         return random.choice(['up', 'left'])  # If all directions are blocked, return a random direction
    
#     elif litter.x < x and litter.y < y:
#         for direction in ['up', 'left']:
#             new_x, new_y = x, y
#             if direction == 'up':
#                 new_y -= 1
#             elif direction == 'down':
#                 new_y += 1
#             elif direction == 'left':
#                 new_x -= 1
#             elif direction == 'right':
#                 new_x += 1

#             # Check if the new position will collide with a pedestrian, car, or house
#             for pedestrian in pedestrians:
#                 if pedestrian.x == new_x and pedestrian.y == new_y:
#                     break
#             else:
#                 for car in cars:
#                     if car.x == new_x and car.y == new_y:
#                         break
#                 else:
#                     for house in houses:
#                         if house.x == new_x and house.y == new_y:
#                             break
#                     else:
#                         # If the new position does not collide with any obstacles, return this direction
#                         return direction
#         return random.choice(['down', 'right'])  # If all directions are blocked, return a random direction
    
#     elif litter.x > x and litter.y < y:
#         for direction in ['up', 'right']:
#             new_x, new_y = x, y
#             if direction == 'up':
#                 new_y -= 1
#             elif direction == 'down':
#                 new_y += 1
#             elif direction == 'left':
#                 new_x -= 1
#             elif direction == 'right':
#                 new_x += 1

#             # Check if the new position will collide with a pedestrian, car, or house
#             for pedestrian in pedestrians:
#                 if pedestrian.x == new_x and pedestrian.y == new_y:
#                     break
#             else:
#                 for car in cars:
#                     if car.x == new_x and car.y == new_y:
#                         break
#                 else:
#                     for house in houses:
#                         if house.x == new_x and house.y == new_y:
#                             break
#                     else:
#                         # If the new position does not collide with any obstacles, return this direction
#                         return direction
#         return random.choice(['down', 'left'])  # If all directions are blocked, return a random direction
    
#     elif litter.x < x and litter.y > y:
#         for direction in ['down', 'left']:
#             new_x, new_y = x, y
#             if direction == 'up':
#                 new_y -= 1
#             elif direction == 'down':
#                 new_y += 1
#             elif direction == 'left':
#                 new_x -= 1
#             elif direction == 'right':
#                 new_x += 1

#             # Check if the new position will collide with a pedestrian, car, or house
#             for pedestrian in pedestrians:
#                 if pedestrian.x == new_x and pedestrian.y == new_y:
#                     break
#             else:
#                 for car in cars:
#                     if car.x == new_x and car.y == new_y:
#                         break
#                 else:
#                     for house in houses:
#                         if house.x == new_x and house.y == new_y:
#                             break
#                     else:
#                         # If the new position does not collide with any obstacles, return this direction
#                         return direction
#         return random.choice(['up', 'right'])  # If all directions are blocked, return a random direction

#     # If all directions are blocked, return None
#     return None

def find_available_way(litter, x, y, pedestrians, cars, houses):
    directions = {
        'up': (0, -1),
        'down': (0, 1),
        'left': (-1, 0),
        'right': (1, 0)
    }

    def is_collision(new_x, new_y):
        agent_rect = pygame.Rect(new_x, new_y, AGENT_SIZE[0], AGENT_SIZE[1])

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

    if litter.x > x and litter.y > y:
        preferred_directions = ['down', 'right']
    elif litter.x < x and litter.y < y:
        preferred_directions = ['up', 'left']
    elif litter.x > x and litter.y < y:
        preferred_directions = ['up', 'right']
    else:  # litter.x < x and litter.y > y
        preferred_directions = ['down', 'left']

    for direction in preferred_directions:
        dx, dy = directions[direction]
        new_x, new_y = x + dx, y + dy
        if not is_collision(new_x, new_y):
            return direction

    # 如果所有偏好的方向都被阻挡，尝试其他方向
    for direction in directions:
        dx, dy = directions[direction]
        new_x, new_y = x + dx, y + dy
        if not is_collision(new_x, new_y):
            return direction

    return None  # 如果所有方向都被阻挡，返回 None
