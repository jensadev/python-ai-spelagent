import numpy as np
import random
import pygame

# Initiera Pygame
pygame.init()

# Skärminställningar
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI-agent med Q-learning")

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

# Q-learning inställningar
actions = ['left', 'right', 'up', 'down']
try:
    q_table = np.load('q_table.npy')
except FileNotFoundError:
    q_table = np.zeros((WIDTH // agent_speed, HEIGHT // agent_speed, len(actions)))
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.1

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
    state = (agent_x // agent_speed, agent_y // agent_speed)

    # Välj en åtgärd (exploration vs exploitation)
    if random.uniform(0, 1) < epsilon:
        action = random.choice(actions)
    else:
        action = actions[np.argmax(q_table[state])]

    # Utför åtgärden
    if action == 'left' and agent_x - agent_speed >= 0:
        agent_x -= agent_speed
    elif action == 'right' and agent_x + agent_speed < WIDTH:
        agent_x += agent_speed
    elif action == 'up' and agent_y - agent_speed >= 0:
        agent_y -= agent_speed
    elif action == 'down' and agent_y + agent_speed < HEIGHT:
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

    # Uppdatera Q-tabellen
    new_state = (agent_x // agent_speed, agent_y // agent_speed)
    q_table[state][actions.index(action)] = (1 - learning_rate) * q_table[state][actions.index(action)] + learning_rate * (reward + discount_factor * np.max(q_table[new_state]))

    # Rita agent och objekt
    pygame.draw.rect(screen, GREEN, (agent_x, agent_y, agent_size, agent_size))
    pygame.draw.rect(screen, RED, (object_x, object_y, object_size, object_size))

    # Visa poäng
    score_text = font.render("Poäng: " + str(score), True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

    # Uppdatera skärmen
    pygame.display.flip()
    pygame.time.Clock().tick(30)

# Spara Q-tabellen när spelet avslutas
np.save('q_table.npy', q_table)

pygame.quit()
