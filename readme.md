Andreas Perssons prompt:

Du är en lärare i med många års erfarenhet. Din uppgift är att ta fram en övning till dina gymnasieelever som går att genomföra på en lektion. Lektionen ska beröra följande delar av kursplanen. Syfte: Undervisningen ska varva teoretiska moment med praktiska och ge elev­erna möjlighet att använda olika digitala verktyg och programvaror. På så vis ska eleverna ges möjlighet att utveckla förståelse av hur AI kan användas i problemlösning och förmåga att omsätta sina kunskaper till praktiska handlingar. Central innehåll: Enklare typ av problemlösning med hjälp av AI, till användning av spelagent. Betygskriterier E/C/A: Eleven använder med viss säkerhet/säkerhet/ god säkerhet

Kod i [human](human-agent.py) 

Mina tillägg:
> Gör inget ännu, utan skriv bara att du har förstått din uppgift.
> Jag har från en föregående följande kod, spara den för framtida anvädning. 
> Bra, det som uppgiften går ut på och som det svar jag fick av copilot tidigare rör AI agenter. Tolkningen som du har gjort tidigare, tolkar jag som att du inte förstått vad som menas med en agent. I koden som skapats så är spelaren agenten, eller spelaren, människan styr agenten.
>Här önskar jag en förändring av koden, hur kan vi använda någon form av AI för att styra en 

Kod i [AI](ai-agent.py)

Bra nu styr datorn agenten, men är det verkligen en AI. Kan du skriva en förklaring, ett förytydligande av din förståelse för vad den hör delen av skolverkets ämnesplan betyder, [och användning av spelagent.] Vad är en spelagent, vad avses i det här fallet, är det en spelmotor, eller är det att ai styr spelet?

Det är en bra fråga! Låt oss klargöra vad som menas med en spelagent och hur det relaterar till AI i undervisningen.

### Förklaring av Skolverkets Ämnesplan

**Syfte:**
Undervisningen ska varva teoretiska moment med praktiska och ge eleverna möjlighet att använda olika digitala verktyg och programvaror. På så vis ska eleverna ges möjlighet att utveckla förståelse av hur AI kan användas i problemlösning och förmåga att omsätta sina kunskaper till praktiska handlingar.

**Centralt innehåll:**
Enklare typ av problemlösning med hjälp av AI, till exempel användning av spelagent.

### Vad är en Spelagent?

En spelagent är en autonom enhet i ett spel som kan fatta beslut och agera baserat på dessa beslut. I detta sammanhang kan en spelagent vara en karaktär eller en enhet som styrs av AI-algoritmer snarare än av en mänsklig spelare. Spelagenten kan utföra uppgifter som att navigera i spelvärlden, interagera med andra objekt eller karaktärer, och uppnå specifika mål.

### AI i Spelagenten

I det tidigare exemplet med koden, styrde en enkel algoritm agenten mot objektet. Även om detta är en grundläggande form av AI, kan vi utveckla mer avancerade AI-agenter som använder tekniker som maskininlärning, beslutsfattande träd, eller förstärkningsinlärning för att förbättra deras förmåga att fatta beslut och agera i spelmiljön.

### Skillnaden mellan Spelagent och Spelmotor

- **Spelagent:** En autonom enhet i spelet som kan fatta beslut och agera baserat på dessa beslut. I vårt exempel är det den gröna fyrkanten som rör sig mot det röda objektet.
- **Spelmotor:** En programvara som tillhandahåller de grundläggande verktygen och funktionerna för att skapa och köra spel. Pygame är ett exempel på en spelmotor som vi använder för att skapa och hantera spelets grafik, ljud och användarinteraktion.

### Användning av AI i Spel

När vi talar om att använda AI i spel, menar vi att skapa algoritmer som kan styra spelagenter på ett intelligent sätt. Detta kan innebära att agenterna lär sig från sina erfarenheter, anpassar sig till förändringar i spelmiljön, och fattar beslut som maximerar deras chanser att uppnå sina mål.

### Sammanfattning

I undervisningen kan vi använda spelagenter för att ge eleverna praktisk erfarenhet av hur AI kan användas för att lösa problem. Genom att programmera spelagenter och se hur de interagerar med spelmiljön, får eleverna en bättre förståelse för AI och dess tillämpningar.

Hoppas detta klargör vad som menas med en spelagent och hur AI kan användas i undervisningen! Låt mig veta om du har fler frågor eller behöver ytterligare förtydliganden.

Att implementera maskininlärning i ditt spel kan vara en spännande och lärorik uppgift! Här är en grundläggande översikt över hur du kan göra detta:

### Steg för att Implementera Maskininlärning

1. **Välj en Maskininlärningsmetod:**
   - **Q-learning:** En typ av förstärkningsinlärning där agenten lär sig genom att utforska och få belöningar eller straff.
   - **Neurala Nätverk:** Används för mer komplexa inlärningsuppgifter där agenten kan lära sig mönster och fatta beslut baserat på dessa.

2. **Installera Nödvändiga Bibliotek:**
   - För Python kan du använda bibliotek som `numpy`, `pandas`, `scikit-learn`, och `tensorflow` eller `pytorch` för neurala nätverk.

3. **Förbered Datan:**
   - Samla in data om agentens rörelser och resultat. Detta kan vara i form av positioner, poäng, och andra relevanta variabler.

4. **Träna Modellen:**
   - Använd den insamlade datan för att träna din maskininlärningsmodell. För Q-learning, kan du använda en Q-tabell för att lagra och uppdatera värden baserat på agentens erfarenheter.

5. **Implementera Modellen i Spelet:**
   - Använd den tränade modellen för att styra agentens rörelser i spelet.

### Exempel på Q-learning

Här är ett enkelt exempel på hur du kan implementera Q-learning i ditt spel:

```python
import numpy as np
import pygame
import random

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

pygame.quit()
```

### Förklaring av Koden

- **Q-tabell:** En tabell som lagrar värden för varje tillstånd och åtgärd.
- **Exploration vs Exploitation:** Agenten väljer ibland slumpmässiga åtgärder (exploration) och ibland de bästa kända åtgärderna (exploitation).
- **Uppdatering av Q-tabellen:** Q-värden uppdateras baserat på agentens erfarenheter och belöningar.

Denna kod ger en grundläggande introduktion till hur du kan använda Q-learning för att styra en agent i ett spel. För mer avancerade tillämpningar kan du utforska djupare maskininlärningsmetoder och algoritmer.

Låt mig veta om du behöver ytterligare hjälp eller förklaringar!

# Hur kan jag förbättra agentens prestanda?

För att förbättra agentens prestanda i ditt spel kan du överväga flera strategier och tekniker. Här är några förslag:

### 1. Förbättra Q-learning Parametrar
- **Learning Rate (α):** Justera inlärningshastigheten för att hitta en balans mellan att lära sig snabbt och att stabilisera inlärningen.
- **Discount Factor (γ):** Justera diskonteringsfaktorn för att balansera mellan kortsiktiga och långsiktiga belöningar.
- **Epsilon (ε):** Justera epsilon för att balansera mellan exploration (utforskning) och exploitation (utnyttjande). Du kan använda en epsilon-greedy strategi där epsilon minskar över tid.

### 2. Använda Avancerade Algoritmer
- **Deep Q-Learning (DQN):** Använd ett neuralt nätverk för att approximera Q-värden istället för en Q-tabell. Detta är särskilt användbart för större och mer komplexa miljöer.
- **Policy Gradient Methods:** Algoritmer som REINFORCE eller Proximal Policy Optimization (PPO) kan användas för att direkt optimera agentens policy.

### 3. Förbättra Miljön och Belöningsstrukturen
- **Belöningsstruktur:** Se till att belöningsstrukturen är väl utformad för att uppmuntra önskat beteende. Du kan lägga till delbelöningar för delmål.
- **Miljövariation:** Introducera variation i miljön för att göra agenten mer robust och anpassningsbar.

### 4. Dataförbättring och Förbehandling
- **Normalisering:** Normalisera indata för att förbättra inlärningens effektivitet.
- **Feature Engineering:** Lägg till eller förbättra funktioner som agenten kan använda för att fatta bättre beslut.

### 5. Hyperparameteroptimering
- **Grid Search eller Random Search:** Använd dessa tekniker för att hitta de bästa hyperparametrarna för din modell.
- **Bayesian Optimization:** En mer avancerad metod för att optimera hyperparametrar.

### Exempel på Deep Q-Learning

Här är ett förenklat exempel på hur du kan implementera Deep Q-Learning med hjälp av `tensorflow`:

```python
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

pygame.quit()
```

### Förklaring av Koden

- **Neuralt Nätverk:** Används för att approximera Q-värden.
- **Predict och Fit:** Används för att uppdatera modellen baserat på agentens erfarenheter.
- **State Normalization:** Normaliserar tillståndet för att förbättra inlärningens effektivitet.

Genom att använda dessa tekniker kan du förbättra agentens prestanda och göra den mer effektiv i att uppnå sina mål. Låt mig veta om du behöver ytterligare hjälp eller förklaringar!


# Detta är fantstiskt intressant, Hur kan jag om möjligt spara den data som båda kodexemplena har tränats på, dels med tensforflow och dels med q-learning. Hur kan jag göra detta så att resultatet inte alltid börjar om?

Det är verkligen spännande att arbeta med AI och maskininlärning! Att spara träningsdata och modeller är viktigt för att undvika att börja om från början varje gång. Här är hur du kan göra detta för både Q-learning och TensorFlow-modellen:

### Spara och Ladda Q-learning Data

För Q-learning kan du spara Q-tabellen till en fil och ladda den vid behov. Här är ett exempel på hur du kan göra detta med `numpy`:

#### Spara Q-tabellen

```python
import numpy as np

# Spara Q-tabellen till en fil
np.save('q_table.npy', q_table)
```

#### Ladda Q-tabellen

```python
import numpy as np

# Ladda Q-tabellen från en fil
q_table = np.load('q_table.npy')
```

### Spara och Ladda TensorFlow-modellen

För TensorFlow-modellen kan du använda inbyggda funktioner för att spara och ladda modellen. Här är ett exempel:

#### Spara Modellen

```python
import tensorflow as tf

# Spara modellen till en fil
model.save('my_model.h5')
```

#### Ladda Modellen

```python
import tensorflow as tf

# Ladda modellen från en fil
model = tf.keras.models.load_model('my_model.h5')
```

### Implementera i Spelkoden

Här är hur du kan integrera dessa funktioner i din spelkod:

#### För Q-learning

```python
import numpy as np
import random

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
```

#### För TensorFlow

```python
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
    model = tf.keras.models.load_model('my_model.h5')
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
model.save('my_model.h5')

pygame.quit()
```

Genom att spara och ladda träningsdata


for the tensorflow example I got this warning, WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')

Can you modify the code to use this new non legacy way of saving the model

# What is deep-q learning?

Deep Q-learning (DQN) is an advanced form of Q-learning that leverages deep neural networks to approximate the Q-values, which represent the expected future rewards for taking a given action in a given state. This approach is particularly useful for handling environments with large or continuous state spaces where traditional Q-learning would be impractical due to the size of the Q-table.

### Key Concepts of Deep Q-learning

1. **Q-learning Recap:**
   - Q-learning is a reinforcement learning algorithm where an agent learns to take actions in an environment to maximize cumulative rewards. It uses a Q-table to store Q-values for state-action pairs.

2. **Deep Neural Networks:**
   - Instead of using a Q-table, DQN uses a neural network to approximate the Q-values. The network takes the state as input and outputs Q-values for all possible actions.

3. **Experience Replay:**
   - DQN uses a technique called experience replay, where the agent's experiences (state, action, reward, next state) are stored in a replay buffer. During training, random samples from this buffer are used to update the network, which helps break the correlation between consecutive experiences and improves learning stability.

4. **Target Network:**
   - DQN employs a separate target network to stabilize training. The target network is a copy of the main network but is updated less frequently. This helps to reduce oscillations and divergence during training.

### How Deep Q-learning Works

1. **Initialize Networks:**
   - Initialize the main Q-network and the target Q-network with random weights.

2. **Experience Collection:**
   - The agent interacts with the environment and collects experiences (state, action, reward, next state).

3. **Store Experiences:**
   - Store these experiences in the replay buffer.

4. **Sample and Train:**
   - Randomly sample a batch of experiences from the replay buffer.
   - For each experience, compute the target Q-value using the target network.
   - Update the main Q-network by minimizing the loss between the predicted Q-values and the target Q-values.

5. **Update Target Network:**
   - Periodically update the target network to match the main Q-network.

### Example Code Snippet

Here is a simplified example of how you might implement DQN using TensorFlow:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import random
from collections import deque

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
state_size = 2  # Example state size
action_size = 4  # Example action size
main_network = create_q_network(state_size, action_size)
target_network = create_q_network(state_size, action_size)
target_network.set_weights(main_network.get_weights())
replay_buffer = deque(maxlen=2000)
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.95  # Discount factor
batch_size = 32

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

# Example of storing experience and training
state = np.array([[0.5, 0.5]])  # Example initial state
action = choose_action(state)
next_state = np.array([[0.6, 0.6]])  # Example next state
reward = 1  # Example reward
done = False  # Example done flag
replay_buffer.append((state, action, reward, next_state, done))
train_network()
target_network.set_weights(main_network.get_weights())
```

### Summary

Deep Q-learning enhances traditional Q-learning by using deep neural networks to handle complex state spaces, experience replay to improve learning stability, and target networks to stabilize training. This makes it a powerful method for training agents in environments with high-dimensional state spaces, such as video games or robotic control tasks.

Feel free to ask if you have more questions or need further clarification!

# So I ran this example code, and my questions is, what does it do, how could it visualize its function so I could use it to understand and explain what is actually happening? In other words, how do I create a program that explains the deep-q algorithm and training when I run it?

Great question! Understanding and visualizing the Deep Q-learning (DQN) algorithm can significantly enhance your comprehension and ability to explain it to others. Let's break down what the example code does and how you can visualize its function.

### What the Example Code Does

1. **Initialization:**
   - Creates two neural networks: the main Q-network and the target Q-network.
   - Initializes a replay buffer to store experiences.
   - Sets parameters for epsilon-greedy strategy, learning rate, discount factor, and batch size.

2. **Choosing Actions:**
   - Uses the epsilon-greedy strategy to balance exploration and exploitation when selecting actions.

3. **Storing Experiences:**
   - Stores experiences (state, action, reward, next state, done) in the replay buffer.

4. **Training the Network:**
   - Samples a batch of experiences from the replay buffer.
   - Updates the Q-values using the Bellman equation.
   - Trains the main Q-network with the updated Q-values.
   - Periodically updates the target Q-network to stabilize training.

### Visualizing the DQN Algorithm

To visualize the DQN algorithm and its training process, you can use various techniques such as plotting graphs, displaying the agent's actions in the environment, and showing the evolution of Q-values. Here's an example of how you can modify the code to include visualization using `matplotlib` and `pygame`:

#### Step-by-Step Visualization

1. **Install Required Libraries:**
   - Ensure you have `matplotlib` installed: `pip install matplotlib`.

2. **Modify the Code for Visualization:**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pygame
import random
from collections import deque
import matplotlib.pyplot as plt

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
state_size = 2  # Example state size
action_size = 4  # Example action size
main_network = create_q_network(state_size, action_size)
target_network = create_q_network(state_size, action_size)
target_network.set_weights(main_network.get_weights())
replay_buffer = deque(maxlen=2000)
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.95  # Discount factor
batch_size = 32

# Visualization setup
scores = []
episodes = []

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

while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

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
    pygame.display.flip()
    pygame.time.Clock().tick(30)

    if len(replay_buffer) >= batch_size:
        episode += 1
        episodes.append(episode)
        scores.append(score)
        score = 0

# Plot the scores
plt.plot(episodes, scores)
plt.xlabel('Episodes')
plt.ylabel('Scores')
plt.title('Scores over Episodes')
plt.show()

pygame.quit()
```

### Explanation of Visualization

1. **Score Plotting:**
   - The code collects scores for each episode and plots them using `matplotlib`. This helps visualize the agent's performance over time.

2. **Real-time Display:**
   - The `pygame` window shows the agent moving towards the object, providing a visual representation of the agent's actions and learning process.

3. **Episodes and Scores:**
   - The scores are plotted against the number of episodes to show how the agent's performance improves as it learns.

By running this modified code, you can visually observe how the agent learns to collect objects more efficiently over time, providing a clear and intuitive understanding of the Deep Q-learning algorithm and its training process. Let me know if you have any more questions or need further assistance!

Absolutely! The reward system is a crucial component of reinforcement learning algorithms, including Deep Q-learning. It guides the agent's learning process by providing feedback on its actions. Here's a detailed explanation:

### What is a Reward System?

In reinforcement learning, the reward system is a mechanism that provides feedback to the agent based on the actions it takes in the environment. The goal of the agent is to maximize the cumulative reward over time.

### Key Concepts of the Reward System

1. **Immediate Reward:**
   - The reward received immediately after taking an action. It can be positive (reward) or negative (penalty).

2. **Cumulative Reward:**
   - The total reward accumulated over a sequence of actions. The agent aims to maximize this cumulative reward.

3. **Reward Function:**
   - A function that defines how rewards are assigned based on the agent's actions and the resulting state. It encapsulates the goals of the agent.

### How the Reward System Works

1. **Action and State Transition:**
   - The agent takes an action in the current state, which leads to a transition to a new state.

2. **Reward Assignment:**
   - The environment provides a reward based on the action taken and the new state. This reward is used to update the agent's knowledge.

3. **Learning from Rewards:**
   - The agent uses the received rewards to update its policy or Q-values, guiding future actions to maximize cumulative rewards.

### Example in the Context of the Game

In the game example provided earlier, the reward system is designed to encourage the agent to collect objects. Here's how it works:

1. **Positive Reward:**
   - The agent receives a reward of `1` each time it successfully collects an object. This positive reward reinforces the behavior of moving towards and collecting objects.

2. **No Reward or Penalty:**
   - If the agent does not collect an object, it receives a reward of `0`. This neutral feedback does not reinforce any particular behavior.

### Code Example of the Reward System

Here's a snippet from the previous code that illustrates the reward system:

```python
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
```

### Explanation of the Code

- **Reward Calculation:**
  - The reward is initially set to `0`.
  - If the agent's position overlaps with the object's position, the reward is set to `1`, and the object is repositioned randomly on the screen.

- **Score Update:**
  - The score is incremented each time the agent collects an object, reflecting the cumulative reward.

### Importance of the Reward System

The reward system is essential because it directly influences the agent's learning process. A well-designed reward system ensures that the agent learns the desired behavior efficiently. In more complex environments, the reward function can be more sophisticated, incorporating various factors to guide the agent towards achieving long-term goals.

By understanding and designing an effective reward system, you can significantly improve the performance and learning efficiency of your reinforcement learning agents. If you have any more questions or need further clarification, feel free to ask!

Great to hear that the game window is working! Let's enhance the visualization with `matplotlib` to better explain what's happening during the training process.

### Enhancing Visualization with Matplotlib

The `matplotlib` implementation in the previous code was intended to plot the scores over episodes, but it seems it didn't provide a real-time explanation of the training process. Let's improve this by adding more detailed visualizations and explanations.

#### Steps to Enhance Visualization

1. **Real-time Plotting:**
   - Use `matplotlib` to plot the agent's performance in real-time.

2. **Detailed Explanations:**
   - Add annotations and labels to the plots to explain what is happening.

3. **Combine Pygame and Matplotlib:**
   - Integrate `pygame` and `matplotlib` to provide a comprehensive visualization.

### Updated Code with Enhanced Visualization

Here's an updated version of the code that includes real-time plotting and detailed explanations:

```python
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
state_size = 2  # Example state size
action_size = 4  # Example action size
main_network = create_q_network(state_size, action_size)
target_network = create_q_network(state_size, action_size)
target_network.set_weights(main_network.get_weights())
replay_buffer = deque(maxlen=2000)
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
gamma = 0.95  # Discount factor
batch_size = 32

# Visualization setup
scores = []
episodes = []

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
    global episode, score, agent_x, agent_y, object_x, object_y
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            plt.close()
            return line,

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
    pygame.display.flip()
    pygame.time.Clock().tick(30)

    if len(replay_buffer) >= batch_size:
        episode += 1
        episodes.append(episode)
        scores.append(score)
        score = 0

    line.set_data(episodes, scores)
    ax.set_xlim(0, max(100, episode))
    ax.set_ylim(0, max(10, max(scores) + 1))
    return line,

ani = FuncAnimation(fig, update, init_func=init, blit=True)
plt.show()
```

### Explanation of the Enhanced Visualization

1. **Real-time Plotting:**
   - The `FuncAnimation` from `matplotlib.animation` is used to update the plot in real-time.
   - The `update` function is called repeatedly to update the plot with new data.

2. **Detailed Explanations:**
   - The plot shows the scores over episodes, providing a visual representation of the agent's learning progress.
   - The axes are labeled, and the plot has a title to explain what is being visualized.

3. **Integration with Pygame:**
   - The `update` function also handles the game logic, ensuring that the game runs while the plot is updated.
   - The game window and the plot window are displayed simultaneously, providing a comprehensive visualization.

By running this updated code, you should see both the game window and a real-time plot of the scores over episodes. This will help you understand and explain the Deep Q-learning algorithm and its training process more effectively. Let me know if you have any more questions or need further assistance!


# Stort tack för all kod och förklaring, mycket hjälpsamt. För att nu återgå till skolverkets ämne AI och kanske även ämnet programmering. Hur kan vi använda koden du skapat för att illustrera AI agenter, visa hur vi tränar AI?
Hur kan vi använda koden, låta eleverna få tillgång till den och utifrån det skapa en uppgift kopplat till det centrala innehållet?
Hur skulle en bedömning på E/C/A skalans se ut för detta?

Det är fantastiskt att du vill använda koden för att illustrera AI-agenter och träning av AI i undervisningen! Här är en plan för hur du kan använda koden i en lektion, skapa en uppgift för eleverna och bedöma deras arbete enligt E/C/A-skalan.

### Lektion och Uppgift

#### Syfte och Mål
- **Syfte:** Ge eleverna en praktisk förståelse av hur AI-agenter fungerar och hur de kan tränas med hjälp av maskininlärning.
- **Mål:** Eleverna ska kunna implementera och träna en AI-agent, förstå grundläggande koncept inom förstärkningsinlärning och kunna analysera agentens beteende.

#### Lektionens Struktur

1. **Introduktion (15 minuter):**
   - Förklara vad AI-agenter är och hur de används i olika sammanhang.
   - Introducera grundläggande koncept inom förstärkningsinlärning, såsom belöningssystem, Q-learning och Deep Q-learning.

2. **Genomgång av Koden (20 minuter):**
   - Gå igenom den tillhandahållna koden steg för steg.
   - Förklara hur agenten tränas och hur belöningssystemet fungerar.
   - Visa hur `matplotlib` används för att visualisera agentens inlärning.

3. **Praktisk Övning (45 minuter):**
   - Låt eleverna köra koden på sina datorer.
   - Uppmuntra dem att experimentera med olika parametrar (t.ex. inlärningshastighet, diskonteringsfaktor, epsilon) för att se hur det påverkar agentens beteende.
   - Be eleverna att dokumentera sina observationer och resultat.

4. **Diskussion och Reflektion (20 minuter):**
   - Diskutera elevernas resultat och erfarenheter.
   - Reflektera över hur AI-agenter kan förbättras och vilka utmaningar som finns.

#### Uppgift

**Uppgift:** Implementera och träna en AI-agent som samlar objekt i en spelmiljö. Analysera agentens beteende och dokumentera dina resultat.

**Instruktioner:**
1. **Kör den tillhandahållna koden och observera agentens beteende.**
2. **Experimentera med olika parametrar och dokumentera hur de påverkar agentens prestanda.**
3. **Försök att förbättra agentens prestanda genom att justera belöningssystemet eller använda mer avancerade algoritmer.**
4. **Skriv en rapport där du beskriver dina experiment, resultat och slutsatser.**

### Bedömning enligt E/C/A-skalan

**E-nivå:**
- Eleven kan med viss säkerhet köra den tillhandahållna koden och observera agentens beteende.
- Eleven dokumenterar sina observationer och gör enkla justeringar av parametrar.
- Rapporten innehåller grundläggande beskrivningar av experiment och resultat.

**C-nivå:**
- Eleven kan med säkerhet köra och modifiera koden för att förbättra agentens prestanda.
- Eleven dokumenterar sina experiment noggrant och analyserar hur olika parametrar påverkar agentens beteende.
- Rapporten innehåller detaljerade beskrivningar och analyser av experiment och resultat.

**A-nivå:**
- Eleven kan med god säkerhet implementera och träna en AI-agent, samt göra avancerade justeringar för att optimera agentens prestanda.
- Eleven dokumenterar sina experiment mycket noggrant och gör djupgående analyser av agentens beteende och prestanda.
- Rapporten innehåller välstrukturerade och insiktsfulla beskrivningar, analyser och slutsatser, samt förslag på vidare förbättringar.

### Sammanfattning

Genom att använda den tillhandahållna koden kan eleverna få en praktisk och djupgående förståelse av AI-agenter och förstärkningsinlärning. Uppgiften och bedömningskriterierna hjälper till att strukturera deras lärande och säkerställa att de uppnår de mål som anges i kursplanen.

Låt mig veta om du behöver ytterligare hjälp eller justeringar!

För att hjälpa eleverna att förstå och experimentera med AI-agentens beteende, kan vi fokusera på att justera några viktiga parametrar i koden. Här är några parametrar som är särskilt viktiga och hur de påverkar agentens beteende:

### Viktiga Parametrar att Justera

1. **Inlärningshastighet (Learning Rate, α):**
   - **Beskrivning:** Bestämmer hur mycket nya informationer påverkar de befintliga Q-värdena.
   - **Parameter i koden:** `learning_rate`
   - **Effekt:** En högre inlärningshastighet gör att agenten lär sig snabbare, men kan också leda till instabil inlärning. En lägre inlärningshastighet gör inlärningen stabilare men långsammare.

2. **Diskonteringsfaktor (Discount Factor, γ):**
   - **Beskrivning:** Bestämmer hur mycket framtida belöningar värderas jämfört med omedelbara belöningar.
   - **Parameter i koden:** `gamma`
   - **Effekt:** En hög diskonteringsfaktor gör att agenten värderar framtida belöningar högre, vilket kan leda till mer långsiktigt planering. En låg diskonteringsfaktor gör att agenten fokuserar mer på omedelbara belöningar.

3. **Utforskningsgrad (Exploration Rate, ε):**
   - **Beskrivning:** Bestämmer hur ofta agenten väljer en slumpmässig åtgärd istället för den bästa kända åtgärden.
   - **Parameter i koden:** `epsilon`
   - **Effekt:** En hög utforskningsgrad gör att agenten utforskar mer, vilket kan hjälpa den att upptäcka bättre strategier. En låg utforskningsgrad gör att agenten utnyttjar sin befintliga kunskap mer.

4. **Epsilon-minskning (Epsilon Decay):**
   - **Beskrivning:** Bestämmer hur snabbt utforskningsgraden minskar över tid.
   - **Parameter i koden:** `epsilon_decay`
   - **Effekt:** En snabbare minskning gör att agenten snabbare övergår till att utnyttja sin kunskap, medan en långsammare minskning gör att agenten utforskar längre.

5. **Storlek på Replay Buffer:**
   - **Beskrivning:** Bestämmer hur många erfarenheter som lagras för träning.
   - **Parameter i koden:** `replay_buffer`
   - **Effekt:** En större buffer gör att agenten kan lära sig från ett bredare spektrum av erfarenheter, vilket kan förbättra inlärningen.

6. **Batchstorlek (Batch Size):**
   - **Beskrivning:** Bestämmer hur många erfarenheter som används för att uppdatera nätverket vid varje träningssteg.
   - **Parameter i koden:** `batch_size`
   - **Effekt:** En större batchstorlek kan göra inlärningen stabilare men långsammare, medan en mindre batchstorlek kan göra inlärningen snabbare men mer instabil.

### Exempel på Justeringar i Koden

Här är ett exempel på hur du kan justera dessa parametrar i koden:

```python
# Justera parametrar
learning_rate = 0.01  # Ändra inlärningshastigheten
gamma = 0.95  # Ändra diskonteringsfaktorn
epsilon = 1.0  # Startvärde för utforskningsgraden
epsilon_min = 0.01  # Minimumvärde för utforskningsgraden
epsilon_decay = 0.995  # Hastighet för minskning av utforskningsgraden
replay_buffer = deque(maxlen=5000)  # Ändra storleken på replay buffer
batch_size = 64  # Ändra batchstorleken
```

### Uppgift för Eleverna

**Uppgift:** Experimentera med olika parametrar för att se hur de påverkar AI-agentens beteende och prestanda. Dokumentera dina resultat och analysera hur varje parameter påverkar agentens inlärning.

**Instruktioner:**
1. **Kör den tillhandahållna koden med standardparametrarna och observera agentens beteende.**
2. **Ändra en parameter i taget och kör koden igen för att se hur agentens beteende förändras.**
3. **Dokumentera dina observationer och resultat för varje parameterändring.**
4. **Analysera och diskutera hur varje parameter påverkar agentens inlärning och prestanda.**

### Bedömning enligt E/C/A-skalan

**E-nivå:**
- Eleven kan med viss säkerhet justera parametrar och observera förändringar i agentens beteende.
- Eleven dokumenterar sina observationer och gör enkla analyser av resultaten.

**C-nivå:**
- Eleven kan med säkerhet justera flera parametrar och analysera hur de påverkar agentens beteende och prestanda.
- Eleven dokumenterar sina experiment noggrant och gör detaljerade analyser av resultaten.

**A-nivå:**
- Eleven kan med god säkerhet justera parametrar och optimera agentens prestanda genom avancerade justeringar.
- Eleven dokumenterar sina experiment mycket noggrant och gör djupgående analyser av resultaten, samt föreslår vidare förbättringar.

Genom att låta eleverna experimentera med dessa parametrar får de en djupare förståelse för hur AI-agenter tränas och hur olika faktorer påverkar inlärningen. Detta ger dem praktisk erfarenhet och insikt i AI och maskininlärning. Låt mig veta om du behöver ytterligare hjälp eller justeringar!