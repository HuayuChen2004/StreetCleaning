import pygame
import random
from utils import *
from house import House
from car import Car
from pedestrian import Pedestrian
from Agent.agent import Agent

# parameters
num_cars = 8
num_pedestrians = 15
num_agent = 6
traffic_light = 'horizontal'
light_duration = 300  # Duration of each traffic light in frames

def main():
    global traffic_light

    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("blocks")

    CHANGE_SPEED_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(CHANGE_SPEED_EVENT, 10000)

    houses = [House(x, y) for x in range(0, WIDTH, HOUSE_SIZE * 2) 
            for y in range(0, HEIGHT, HOUSE_SIZE * 2)]
    # get the xranges and yranges of roads
    road_xranges = [(x + HOUSE_SIZE, x + HOUSE_SIZE * 2) for x in range(0, WIDTH - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]
    road_yranges = [(y + HOUSE_SIZE, y + HOUSE_SIZE * 2) for y in range(0, HEIGHT - HOUSE_SIZE * 2, HOUSE_SIZE * 2)]

    cars = []
    pedestrians = []
    litters = []
    agents = []

    road_used_x = [False] * len(road_xranges)
    road_used_y = [False] * len(road_yranges)
    count = 0

    # 生成车辆
    while count < num_cars:
        direction = random.choice(DIRECTIONS)
        litter_prob = random.randint(1, 3) / 5000
        if direction in ['left', 'right']:
            # 如果车辆是左右移动的，那么在 x 范围内选择一个未使用的道路
            available_roads = [i for i, used in enumerate(road_used_y) if not used]
            if not available_roads:
                continue
            road_index = random.choice(available_roads)
            random_road = road_yranges[road_index]
            # 让车在马路中央行驶
            y_position = (random_road[0] + random_road[1] - CAR_WIDTH) // 2
            cars.append(Car(random.randint(0, WIDTH - CAR_LENGTH),
                            y_position, 
                            direction,
                            litter_prob))
            road_used_y[road_index] = True
            count += 1
        else:
            # 如果车辆是上下移动的，那么在 y 范围内选择一个未使用的道路
            available_roads = [i for i, used in enumerate(road_used_x) if not used]
            if not available_roads:
                continue
            road_index = random.choice(available_roads)
            random_road = road_xranges[road_index]
            # 让车在马路中央行驶
            x_position = (random_road[0] + random_road[1] - CAR_WIDTH) // 2
            cars.append(Car(x_position, 
                            random.randint(0, HEIGHT - CAR_LENGTH),
                            direction,
                            litter_prob))
            road_used_x[road_index] = True
            count += 1

    
    road_sides = {(i, side): False for i in range(len(road_yranges)) for side in ['left', 'right']}
    count = 0
    while count < num_pedestrians:
        direction = random.choice(DIRECTIONS)
        litter_prob = random.randint(1, 3) / 10000  # 行人乱丢垃圾的概率
        if direction in ['left', 'right']:
            # 如果行人是左右移动的，那么在 y 范围内选择一个随机的道路
            available_roads = [i for i in range(len(road_yranges)) if not road_sides[(i, direction)]]
            if not available_roads:
                continue
            road_index = random.choice(available_roads)
            random_road = road_yranges[road_index]
            # 行人应该在道路的一侧，靠着房子行走，但不能和房子重合
            y_position = random_road[0] - PEDESTRIAN_SIZE[1] - HOUSE_SIZE if direction == 'left' else random_road[1] + HOUSE_SIZE
            if y_position < 0 or y_position > HEIGHT - PEDESTRIAN_SIZE[1]:
                continue
            pedestrians.append(Pedestrian(random.randint(0, WIDTH - PEDESTRIAN_SIZE[0]), 
                                        y_position, 
                                        direction,
                                        litter_prob))
            road_sides[(road_index, direction)] = True
            count += 1
        else:
            # 如果行人是上下移动的，那么在 x 范围内选择一个随机的道路
            random_road = random.choice(road_xranges)
            # 行人应该在道路的一侧，靠着房子行走，但不能和房子重合
            x_position = random_road[0] - PEDESTRIAN_SIZE[0] - HOUSE_SIZE if direction == 'up' else random_road[1] + HOUSE_SIZE
            if x_position < 0 or x_position > WIDTH - PEDESTRIAN_SIZE[0]:
                continue
            pedestrians.append(Pedestrian(x_position, 
                                        random.randint(0, HEIGHT - PEDESTRIAN_SIZE[1]), 
                                        direction,
                                        litter_prob))
            count += 1
    count = 0
    road_sides = {(i, side): False for i in range(len(road_yranges)) for side in ['left', 'right']}
    while count < num_agent:
        available_roads = [i for i in range(len(road_yranges)) if not road_sides[(i, 'left')]]
        if not available_roads:
            break
        road_index = random.choice(available_roads)
        random_road = road_yranges[road_index]
        # let the agent land on the road
        y_position = random_road[0] 
        available_xranges = [[x, x + HOUSE_SIZE] for x in range(0, WIDTH - 2 * HOUSE_SIZE, 2 * HOUSE_SIZE)]
        xrange = random.choice(available_xranges)
        x_position = random.randint(xrange[0], xrange[1] - AGENT_SIZE)
        # Check if the agent would overlap with a car, pedestrian or house
        for house in houses:
            if house.rect.collidepoint(x_position, y_position):
                continue
        for car in cars:
            if car.rect.collidepoint(x_position, y_position):
                continue
        # for pedestrian in pedestrians:
        #     if pedestrian.rect.collidepoint(x_position, y_position):
        #         continue
        agent = Agent(x_position, y_position)
        agents.append(agent)
        road_sides[(road_index, 'left')] = True
        count += 1


    # 主循环
    running = True
    clock = pygame.time.Clock()

    frame_count = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == CHANGE_SPEED_EVENT:
                for car in cars:
                    car.change_speed()

        # 计算剩余时间
        left_time = light_duration - (frame_count % light_duration)

        # 更新车辆和行人位置
        for car in cars:
            # car.speed = random.choice([3, 4, 5])
            car.move(traffic_light, left_time)
            # litter_x = random.randint(car.x, car.x + CAR_LENGTH)
            # litter_y = random.randint(car.y, car.y + CAR_WIDTH)
            litter = car.generate_litter(car.x, car.y)
            if litter:
                litters.append(litter)
        for pedestrian in pedestrians:
            pedestrian.move(traffic_light, left_time)
            # litter_x = random.randint(pedestrian.x - 3, pedestrian.x + PEDESTRIAN_SIZE[0] + 3)
            # litter_y = random.randint(pedestrian.y - 3, pedestrian.y + PEDESTRIAN_SIZE[1] + 3)
            litter = pedestrian.generate_litter(pedestrian.x, pedestrian.y)
            if litter:
                litters.append(litter)
        for litter in litters:
            if litter.weight == 0:
                litters.remove(litter)
        for agent in agents:
            agent.start_cleaning(litters, pedestrians, cars, agents, houses, traffic_light)
            for litter in litters:
                if agent.is_litter_covered(litter):
                    litter.weight -= 1
                    print("litter is being cleaned")

        # 清屏
        screen.fill(WHITE)

        # 绘制房子、车辆和行人
        for house in houses:
            house.draw(screen)
        for car in cars:
            car.draw(screen)
        for pedestrian in pedestrians:
            pedestrian.draw(screen)
        for litter in litters:
            litter.draw(screen)
        for agent in agents:
            agent.draw(screen)

        font = pygame.font.Font(None, 36)
        text = font.render(f"Traffic Light: {traffic_light}", True, BLACK)
        text_background = pygame.Surface((text.get_width() + 10, text.get_height() + 10))
        text_background.fill((128, 128, 128))  # 灰色背景
        screen.blit(text_background, (WIDTH - text.get_width() - 20, HEIGHT - text.get_height() - 20))
        screen.blit(text, (WIDTH - text.get_width() - 10, HEIGHT - text.get_height() - 10))

        font = pygame.font.Font(None, 36)
        left_time_text = font.render(f"Left Time: {left_time}", True, BLACK)
        left_time_background = pygame.Surface((left_time_text.get_width() + 10, left_time_text.get_height() + 10))
        left_time_background.fill((128, 128, 128))  # 灰色背景
        screen.blit(left_time_background, (10, HEIGHT - left_time_text.get_height() - 20))
        screen.blit(left_time_text, (20, HEIGHT - left_time_text.get_height() - 10))

        # 刷新屏幕
        pygame.display.flip()
        clock.tick(60)

        if frame_count % light_duration == 0:
            traffic_light = 'vertical' if traffic_light == 'horizontal' else 'horizontal'
            for pedestrian in pedestrians:
                pedestrian.moving = True

        frame_count += 1

    pygame.quit()

if __name__ == '__main__':
    main()
