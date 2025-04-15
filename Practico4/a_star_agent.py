import numpy as np
import itertools
from queue import PriorityQueue

class AStarAgent:
    def __init__(self, model, heuristic_type="combined"):
        self.model = model
        self.heuristic_type = heuristic_type
        self.action_list = []
        self.expanded_nodes = 0
        self.counter = itertools.count()  # Para desempatar en el heap

    def loop(self, env):
        start_state_flatten = env.reset()
        done = False
        step_counter = 0
        all_rewards = 0
        env.render()

        while not done:
            action = self.next_action(start_state_flatten.reshape(3, 3))
            obs, reward, done_env, _ = env.step(action)
            all_rewards += reward
            done = done_env
            env.render()
            step_counter += 1

        return all_rewards, step_counter

    def next_action(self, state):
        if self.action_list:
            return self.action_list.pop(0)

        self.a_star(state)
        return self.action_list.pop(0)

    def heuristic(self, state):
        goal = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0]).reshape(3, 3)

        if self.heuristic_type == "manhattan":
            return self._manhattan_distance(state, goal)

        if self.heuristic_type == "misplaced":
            return self._misplaced_tiles(state, goal)

        if self.heuristic_type == "combined":
            return (
                self._manhattan_distance(state, goal) +
                self._misplaced_tiles(state, goal)
            )

        raise ValueError("Heurística inválida.")

    def _manhattan_distance(self, state, goal):
        total = 0
        for num in range(1, 9):
            curr = np.argwhere(state == num)[0]
            goal_pos = np.argwhere(goal == num)[0]
            total += abs(curr[0] - goal_pos[0]) + abs(curr[1] - goal_pos[1])
        return total

    def _misplaced_tiles(self, state, goal):
        return np.sum((state != goal) & (state != 0))

    def is_goal(self, state):
        return np.array_equal(state.flatten(), [1, 2, 3, 4, 5, 6, 7, 8, 0])

    def a_star(self, start_state):
        self.action_list.clear()
        self.expanded_nodes = 0
        visited = set()
        open_set = PriorityQueue()

        h = self.heuristic(start_state)
        g = 0
        f = g + h
        count = next(self.counter)
        open_set.put((f, count, g, start_state, []))

        while not open_set.empty():
            f, _, g, current_state, path = open_set.get()
            self.expanded_nodes += 1

            state_id = self._state_to_tuple(current_state)
            if state_id in visited:
                continue
            visited.add(state_id)

            if self.is_goal(current_state):
                self.action_list = path
                return

            for action in range(4):
                next_state = self.model.get_next_state(current_state, action)

                if np.array_equal(current_state, next_state):
                    continue  # Acción inválida (no cambia el estado)

                next_id = self._state_to_tuple(next_state)
                if next_id in visited:
                    continue

                new_path = path + [action]
                next_g = g + 1
                next_h = self.heuristic(next_state)
                next_f = next_g + next_h
                count = next(self.counter)
                open_set.put((next_f, count, next_g, next_state, new_path))

        # Si no encontró solución
        self.action_list = []

    def _state_to_tuple(self, state):
        return tuple(state.flatten())
