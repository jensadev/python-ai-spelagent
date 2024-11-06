import pygame
import random

# Initiera Pygame
pygame.init()

# Skärminställningar
WIDTH, HEIGHT = 600, 400

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Spelagent som samlar objekt")

# Färger
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Spelagentens inställningar
agent_size = 20
agent_x, agent_y = WIDTH // 2, HEIGHT // 2
agent_speed = 5

# Objektinställningar
object_size = 15
object_x = random.randint(0, WIDTH - object_size)
object_y = random.randint(0, HEIGHT - object_size)

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

    # Spelagentens rörelse
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and agent_x - agent_speed > 0:
        agent_x -= agent_speed

    if keys[pygame.K_RIGHT] and agent_x + agent_speed < WIDTH - agent_size:
        agent_x += agent_speed

    if keys[pygame.K_UP] and agent_y - agent_speed > 0:
        agent_y -= agent_speed

    if keys[pygame.K_DOWN] and agent_y + agent_speed < HEIGHT - agent_size:
        agent_y += agent_speed

    # Kontrollera om agenten träffar objektet
    if (
        object_x < agent_x < object_x + object_size
        or object_x < agent_x + agent_size < object_x + object_size
    ) and (
        object_y < agent_y < object_y + object_size
        or object_y < agent_y + agent_size < object_y + object_size
    ):
        score += 1
        object_x = random.randint(0, WIDTH - object_size)
        object_y = random.randint(0, HEIGHT - object_size)

    # Rita agent och objekt
    pygame.draw.rect(screen, GREEN, (agent_x, agent_y, agent_size, agent_size))
    pygame.draw.rect(screen, RED, (object_x, object_y, object_size, object_size))

    # Visa poäng
    score_text = font.render("Poäng: " + str(score), True, (0, 0, 0))
    screen.blit(score_text, (10, 10))

    # Uppdatera skärmen
    pygame.display.flip()
    pygame.time.Clock().tick(30)

pygame.quit()
