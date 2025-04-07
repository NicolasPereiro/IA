import numpy as np
from priority_queue import PriorityQueue


class AStarAgent:
    def __init__(self, model):
        self.model = model
        self.action_list = []

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
        if self.action_list != []:
            action = self.action_list.pop(0)
            return action
        self.a_star(state)
        return self.action_list.pop(0)
    
    def heuristic(self, state):
        # Manhattan distance heuristic
        goal_state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0])
        distance = np.sum(np.abs(state.flatten() - goal_state))
        return distance

    def is_goal(self, state):
        return np.array_equal(state.flatten(), np.array([1, 2, 3, 4, 5, 6, 7, 8, 0]))

    def a_star(self, start_state):
        minCost = []
        for i in range(4):
            next_state = self.model.get_next_state(start_state, i)
            if self.is_goal(next_state):
                self.action_list.append(i)
                return
            cost_estimated = self.heuristic(next_state)
            if np.array_equal(start_state, next_state):
                continue
            minCost.append((cost_estimated, next_state, i))
        # Sort by estimated cost
        minCost.sort(key=lambda x: x[0])
        next_action = minCost[0][2]
        self.action_list.append(next_action)