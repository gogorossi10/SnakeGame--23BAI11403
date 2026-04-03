import numpy as np
import random
import pygame
import matplotlib.pyplot as plt
import pickle
import os

# ---------------- CONFIG ----------------
GRID_SIZE = 15
CELL_SIZE = 30
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = GRID_SIZE * CELL_SIZE

EPISODES = 2500
MAX_STEPS = 200
MODEL_FILE = "snake_q_model.pkl"

ACTIONS = [0, 1, 2, 3]
DIRS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}

# ---------------- FOOD SPAWN (FIXED) ----------------
def spawn_food(snake, obstacles):
    while True:
        pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
        if pos not in snake and pos not in obstacles:
            return pos

# ---------------- DIFFICULTY ----------------
def get_difficulty():
    print("\nSelect Difficulty:")
    print("1. Easy")
    print("2. Medium")
    print("3. Hard")

    choice = input("Enter choice: ")

    if choice == "1":
        return 5, 5
    elif choice == "2":
        return 10, 15
    else:
        return 15, 25

# ---------------- STATE ----------------
def get_state(snake, food, obstacles):
    head = snake[0]

    def danger(pos):
        return (
            pos in snake or
            pos in obstacles or
            pos[0] < 0 or pos[0] >= GRID_SIZE or
            pos[1] < 0 or pos[1] >= GRID_SIZE
        )

    up = (head[0]-1, head[1])
    down = (head[0]+1, head[1])
    left = (head[0], head[1]-1)
    right = (head[0], head[1]+1)

    return (
        danger(up), danger(down), danger(left), danger(right),
        food[0] < head[0], food[0] > head[0],
        food[1] < head[1], food[1] > head[1]
    )

# ---------------- AGENT ----------------
class QAgent:
    def __init__(self):
        self.Q = {}
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        if state not in self.Q:
            self.Q[state] = [0]*4

        return np.argmax(self.Q[state])

    def update(self, s, a, r, ns):
        if s not in self.Q:
            self.Q[s] = [0]*4
        if ns not in self.Q:
            self.Q[ns] = [0]*4

        self.Q[s][a] += self.alpha * (
            r + self.gamma * max(self.Q[ns]) - self.Q[s][a]
        )

    def decay(self, episode):
        self.epsilon = max(self.epsilon_min, 1/(1+0.005*episode))

# ---------------- SAVE / LOAD ----------------
def save_model(agent):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(agent.Q, f)
    print("Model saved!")

def load_model(agent):
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            agent.Q = pickle.load(f)
        print("Model loaded!")
    else:
        print("No saved model found!")

# ---------------- TRAIN ----------------
def train(agent):
    rewards = []

    for ep in range(EPISODES):
        snake = [(7, 7)]
        obstacles = []
        food = spawn_food(snake, obstacles)

        total_reward = 0
        steps_without_food = 0
        prev_action = None

        for step in range(MAX_STEPS):
            state = get_state(snake, food, obstacles)
            action = agent.choose_action(state)

            move = DIRS[action]
            new_head = (snake[0][0]+move[0], snake[0][1]+move[1])

            # Collision
            if (new_head in snake or new_head in obstacles or
                new_head[0] < 0 or new_head[0] >= GRID_SIZE or
                new_head[1] < 0 or new_head[1] >= GRID_SIZE):
                reward = -20
                agent.update(state, action, reward, state)
                break

            old_dist = abs(food[0]-snake[0][0]) + abs(food[1]-snake[0][1])

            snake.insert(0, new_head)

            if new_head == food:
                reward = 30
                food = spawn_food(snake, obstacles)
                steps_without_food = 0
            else:
                snake.pop()
                new_dist = abs(food[0]-new_head[0]) + abs(food[1]-new_head[1])
                reward = 5 if new_dist < old_dist else -5
                steps_without_food += 1

            # Loop penalty
            if action == prev_action:
                reward -= 1
            prev_action = action

            # Prevent infinite loop
            if steps_without_food > 50:
                reward = -20
                break

            next_state = get_state(snake, food, obstacles)
            agent.update(state, action, reward, next_state)

            total_reward += reward

        agent.decay(ep)
        rewards.append(total_reward)

        if ep % 300 == 0:
            print(f"Episode {ep}, Reward {total_reward}")

    return rewards

# ---------------- GAME ----------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake RL Final")
font = pygame.font.SysFont(None, 28)

def draw(snake, food, obstacles, score):
    screen.fill((20, 20, 20))

    for ob in obstacles:
        pygame.draw.rect(screen, (120,120,120),
                         (ob[1]*CELL_SIZE, ob[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    for i, seg in enumerate(snake):
        color = (0, 255 - i*5, 0)
        pygame.draw.rect(screen, color,
                         (seg[1]*CELL_SIZE, seg[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.draw.circle(screen, (255, 0, 0),
                       (food[1]*CELL_SIZE+15, food[0]*CELL_SIZE+15), 10)

    text = font.render(f"Score: {score} | R: Restart", True, (255,255,255))
    screen.blit(text, (10, 10))

    pygame.display.update()

def play(agent):
    speed, obstacle_count = get_difficulty()
    agent.epsilon = 0

    while True:
        snake = [(7,7)]
        obstacles = []
        score = 0

        while len(obstacles) < obstacle_count:
            pos = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if pos not in snake:
                obstacles.append(pos)

        food = spawn_food(snake, obstacles)

        clock = pygame.time.Clock()
        running = True
        step_count = 0

        while running:
            clock.tick(speed)
            step_count += 1

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        running = False

            # Prevent infinite loop
            if step_count > 300:
                print("Stopped due to loop")
                running = False
                continue

            state = get_state(snake, food, obstacles)
            action = agent.choose_action(state)

            move = DIRS[action]
            new_head = (snake[0][0]+move[0], snake[0][1]+move[1])

            if (new_head in snake or new_head in obstacles or
                new_head[0] < 0 or new_head[0] >= GRID_SIZE or
                new_head[1] < 0 or new_head[1] >= GRID_SIZE):
                print("Game Over! Score:", score)
                pygame.time.delay(1500)
                running = False
                continue

            snake.insert(0, new_head)

            if new_head == food:
                score += 1
                food = spawn_food(snake, obstacles)
            else:
                snake.pop()

            draw(snake, food, obstacles, score)

# ---------------- MAIN ----------------
agent = QAgent()

print("1. Train Model")
print("2. Load Model")
choice = input("Enter choice: ")

if choice == "1":
    rewards = train(agent)
    save_model(agent)
    plt.plot(rewards)
    plt.title("Training Reward")
    plt.show()

else:
    load_model(agent)

print("Starting Game...")
play(agent)