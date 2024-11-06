import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pygame
import random
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the Q-network
def create_q_network(state_size, action_size):
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=(state_size,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Initialize parameters
state_size = 2
action_size = 4
main_network = create_q_network(state_size, action_size)
target_network = create_q_network(state_size, action_size)
target_network.set_weights(main_network.get_weights())
replay_buffer = deque(maxlen=2000)
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.95
batch_size = 32
learning_rate = 0.001

# Visualization setup
scores = []
episodes = []

# Slider class
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.label = label
        self.dragging = False

    def draw(self, screen):
        pygame.draw.rect(screen, (100, 100, 100), self.rect)
        handle_x = self.rect.x + (self.val - self.min_val) / (self.max_val - self.min_val) * self.rect.w
        pygame.draw.circle(screen, (200, 200, 200), (int(handle_x), self.rect.centery), self.rect.h // 2)
        font = pygame.font.Font(None, 24)
        text = font.render(f"{self.label}: {self.val:.3f}", True, (0, 0, 0))
        screen.blit(text, (self.rect.x, self.rect.y - 25))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                rel_x = event.pos[0] - self.rect.x
                self.val = self.min_val + (rel_x / self.rect.w) * (self.max_val - self.min_val)
                self.val = max(self.min_val, min(self.val, self.max_val))

# Initialize sliders
learning_rate_slider = Slider(10, 10, 200, 20, 0.0001, 0.01, learning_rate, "Learning Rate")
epsilon_slider = Slider(10, 50, 200, 20, 0.01, 1.0, epsilon, "Epsilon")

# Function to choose action
def choose_action(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    q_values = main_network.predict(state)
    return np.argmax(q_values[0])

# Function to train the network
def train_network():
    if len(replay_buffer) < batch_size:
        return
    minibatch = random.sample(replay_buffer, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target += gamma * np.amax(target_network.predict(next_state)[0])
        target_f = main_network.predict(state)
        target_f[0][action] = target
        main_network.fit(state, target_f, epochs=1, verbose=0)
    global epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI-agent med Deep Q-learning")
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
agent_size = 20
agent_x, agent_y = WIDTH // 2, HEIGHT // 2
agent_speed = 20
object_size = 15
object_x = random.randint(0, WIDTH - object_size)
object_y = random.randint(0, HEIGHT - object_size)
score = 0
font = pygame.font.Font(None, 36)
running = True
episode = 0

# Matplotlib setup
fig, ax = plt.subplots()
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)
line, = ax.plot([], [], lw=2)
ax.set_xlabel('Episodes')
ax.set_ylabel('Scores')
ax.set_title('Scores over Episodes')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    global episode, score, agent_x, agent_y, object_x, object_y, epsilon, learning_rate
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            plt.close()
            return line,
        learning_rate_slider.handle_event(event)
        epsilon_slider.handle_event(event)

    # Update parameters from sliders
    learning_rate = learning_rate_slider.val
    epsilon = epsilon_slider.val

    state = np.array([[agent_x / WIDTH, agent_y / HEIGHT]])
    action = choose_action(state)
    if action == 0 and agent_x - agent_speed >= 0:
        agent_x -= agent_speed
    elif action == 1 and agent_x + agent_speed < WIDTH:
        agent_x += agent_speed
    elif action == 2 and agent_y - agent_speed >= 0:
        agent_y -= agent_speed
    elif action == 3 and agent_y + agent_speed < HEIGHT:
        agent_y += agent_speed

    reward = 0
    if (
        object_x < agent_x < object_x + object_size
        or object_x < agent_x + agent_size < object_x + object_size
    ) and (
        object_y < agent_y < object_y + object_size
        or object_y < agent_y + agent_size < object_y + object_size
    ):
        reward = 1
        score += 1
        object_x = random.randint(0, WIDTH - object_size)
        object_y = random.randint(0, HEIGHT - object_size)

    next_state = np.array([[agent_x / WIDTH, agent_y / HEIGHT]])
    replay_buffer.append((state, action, reward, next_state, False))
    train_network()
    target_network.set_weights(main_network.get_weights())

    pygame.draw.rect(screen, GREEN, (agent_x, agent_y, agent_size, agent_size))
    pygame.draw.rect(screen, RED, (object_x, object_y, object_size, object_size))
    score_text = font.render("Poäng: " + str(score), True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

    # Draw sliders
    learning_rate_slider.draw(screen)
    epsilon_slider.draw(screen)

    pygame.display.flip()
    pygame.time.Clock().tick(30)

    if len(replay_buffer) >= batch_size:
        episode += 1
        episodes.append(episode)
        scores.append(score)
        score = 0

    line.set_data(episodes, scores)
    ax.set_xlim(0, max(100, episode))
    if scores:
        ax.set_ylim(0, max(10, max(scores) + 1))
    else:
        ax.set_ylim(0, 10)
    return line,

ani = FuncAnimation(fig, update, init_func=init, blit=True)
plt.show()