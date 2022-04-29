
# Common Libraries
import random

# Custom Libraries

# Sources

def shuffle_list(list):
    random_state = random.randint(0, 100)
    random.seed(random_state)
    random.shuffle(list)
    return list
