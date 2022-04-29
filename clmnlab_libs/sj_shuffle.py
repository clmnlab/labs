
import random

def shuffle_list(list):
    random_state = random.randint(0, 100)
    random.seed(random_state)
    random.shuffle(list)
    return list
