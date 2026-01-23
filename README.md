# Nvolution
Evolution creature neural network python game

Watch creatures evolve after generations!

Creatures:
1. Have 1 layer neural network brains (16 neurons total)
2. Have 13 segment vision segments
3. Comunicate using scent
4. Rare mutations
5. Only the fitest survive
6. (Deafult neuron ammount is 16, you can set it to 64 and more, 64 is recomended for better devices)
Gameplay:
1. Only right click to make food right now

Other:
1. A lot of settings
2. Saves best neural network to newest.json each generation
3. Saves generation logs to simulation_log.json

Setting example:
```
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 650 # Window size
CREATURE_COUNT = 30 
CREATURE_LIMIT = 100
FOOD_COUNT = 300
REPRODUCE_ENERGY = 170 # Minimal energy needed for 2 random creatures to reproduce
START_ENERGY = 140 # Energy the creature starts with
SCENT_GRID_SIZE = 20
SCENT_DECAY = 0.95
SCENT_STRENGTH = 17.0
VISION_SEGMENTS = 36
VISION_RANGE = 200
GENERATION_STEPS = 1000  # cycles per generation
SCENT_SPREAD_RADIUS = 6
SCENT_FALLOFF = 1.5
SHOW_SCENT = True
LAYER_NEURONS = 16 # number of neurons for the hidden layer, before changing delete the old model!

MODEL_SAVE_PATH = "newest"
```
# Example photos
<img width="997" height="679" alt="image" src="https://github.com/user-attachments/assets/8b690143-55cf-4528-afa1-4196578fb4ff" />
<img width="995" height="677" alt="image" src="https://github.com/user-attachments/assets/0fb21a0e-f19e-4b00-b86d-bb070afa14eb" />
<img width="399" height="306" alt="image" src="https://github.com/user-attachments/assets/79ef62c3-955b-46c5-b736-1e6e8bac0052" /> <img width="784" height="653" alt="image" src="https://github.com/user-attachments/assets/44045698-5c6f-4bc0-8f96-de28dec86cd2" />

