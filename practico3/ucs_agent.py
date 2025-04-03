from search_agent import SearchAgent
from priority_queue import PriorityQueue


class UCSAgent(SearchAgent):

    def __init__(self, env, initial_state, end_state, model):
        super().__init__(env, initial_state, end_state, model)
        self.action_list = []

    def _next_action(self):
       #Usar action_list o llamar a ucs para actualizar self.action_list
       action = self.action_list.pop(0) if self.action_list else None
       return action

    def ucs(self):
        #Algoritmo de ucs que retorna la lista de acciones para llegar al destino
        map = self.model.graph
        position = self.initial_state
        print("Initial position: ", position)
        visited = set()
        previous = {}
        queue = PriorityQueue()
        visited.add(position)
        queue.push(position, 0, None)  # Initial cost is 0, and no previous node
        while not queue.is_empty():
            current_position = queue.pop()[0]
            print("Current position: ", current_position)
            if current_position == self.end_state:
                print("Goal reached: ", current_position)
                return self.actionsReconstruction(previous, current_position)
            possible_actions = map[current_position]
            print("Possible actions: ", possible_actions)
            for action in possible_actions:
                print("Action: ", action)
                if map[current_position][action] not in visited:
                    print("Adding to queue: ", map[current_position][action])
                    visited.add(map[current_position][action])
                    queue.push(action, 1, current_position)  # Assuming uniform cost of 1 for each action
                    previous[map[current_position][action]] = current_position
                    
    def actionsReconstruction(self, previous, current_position):
        actions = []
        while current_position != self.initial_state:
            actions.append(previous[current_position])
            current_position = previous[current_position]
        actions.reverse()
        return actions
            
            
            
