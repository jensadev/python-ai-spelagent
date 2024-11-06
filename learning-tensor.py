import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pygame
import random

# Initiera Pygame
pygame.init()

# Skärminställningar
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI-agent med Deep Q-learning")

# Färger
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# AI-agentens inställningar
agent_size = 20
agent_x, agent_y = WIDTH // 2, HEIGHT // 2
agent_speed = 20

# Objektinställningar
object_size = 15
object_x = random.randint(0, WIDTH - object_size)
object_y = random.randint(0, HEIGHT - object_size)

# Deep Q-learning inställningar
num_actions = 4
state_size = 2
try:
    model = tf.keras.models.load_model('my_model.keras')
except:
    model = tf.keras.Sequential([
        layers.Dense(24, activation='relu', input_shape=(state_size,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(num_actions, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Poängräknare
score = 0
font = pygame.font.Font(None, 36)

# Spelloop
running = True

while running:
    screen.fill(WHITE)

    # Händelsehantering
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Diskretisera agentens position
    state = np.array([agent_x / WIDTH, agent_y / HEIGHT])

    # Välj en åtgärd
    q_values = model.predict(state.reshape(1, -1))
    action = np.argmax(q_values[0])

    # Utför åtgärden
    if action == 0 and agent_x - agent_speed >= 0:
        agent_x -= agent_speed
    elif action == 1 and agent_x + agent_speed < WIDTH:
        agent_x += agent_speed
    elif action == 2 and agent_y - agent_speed >= 0:
        agent_y -= agent_speed
    elif action == 3 and agent_y + agent_speed < HEIGHT:
        agent_y += agent_speed

    # Kontrollera om agenten träffar objektet
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

    # Uppdatera modellen
    new_state = np.array([agent_x / WIDTH, agent_y / HEIGHT])
    target = reward + 0.9 * np.max(model.predict(new_state.reshape(1, -1)))
    target_f = model.predict(state.reshape(1, -1))
    target_f[0][action] = target
    model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)

    # Rita agent och objekt
    pygame.draw.rect(screen, GREEN, (agent_x, agent_y, agent_size, agent_size))
    pygame.draw.rect(screen, RED, (object_x, object_y, object_size, object_size))

    # Visa poäng
    score_text = font.render("Poäng: " + str(score), True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

    # Uppdatera skärmen
    pygame.display.flip()
    pygame.time.Clock().tick(30)

# Spara modellen när spelet avslutas
model.save('my_model.keras')

pygame.quit()
