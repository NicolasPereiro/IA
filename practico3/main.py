import gymnasium
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from ucs_agent import UCSAgent
from a_star_agent import AStarAgent
from model import Model
import time
import traceback

envs = [
    (
        "4x4",
        gymnasium.make(
            "FrozenLake-v1",
            desc=generate_random_map(size=4),
            is_slippery=False,
            render_mode="rgb_array",
        ),
    ),
    (
        "8x8",
        gymnasium.make(
            "FrozenLake-v1",
            desc=generate_random_map(size=8),
            is_slippery=False,
            render_mode="rgb_array",
        ),
    ),
    (
        "16x16",
        gymnasium.make(
            "FrozenLake-v1",
            desc=generate_random_map(size=16),
            is_slippery=False,
            render_mode="rgb_array",
        ),
    ),
]

agents = [
    # Ejemplo: ("UCS", UCSAgent),
]


def main():
    #Ejecutar los agentes con distintos entornos y comparar performance 
    raise NotImplementedError





if __name__ == "__main__":
    main()
