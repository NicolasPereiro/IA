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
            current_node = queue.pop()
            current_position = current_node[0]
            print("Current position: ", current_position)
            possible_actions = map.get(current_position, [])
            visited.add(current_position)
            if current_position == self.end_state:
                self.action_list = self.actionsReconstruction(previous, current_position)
                print(self.action_list)
                break
            for action in possible_actions:
                next_position = possible_actions[action]
                if map.get(next_position) is None:
                    cost = 9223372036854775807 # Infinite cost if the next position is not in the map
                else:
                    cost = current_node[1] + 1
                if next_position not in visited:
                    queue.push(next_position, cost, current_position)
                    previous[next_position] = (current_position, action)
            
                    
    def actionsReconstruction(self, previous, current_position):
        actions = []
        while current_position in previous:
            current_position, action = previous[current_position]
            actions.append(action)
        actions.reverse()
        return actions
            
            
            
