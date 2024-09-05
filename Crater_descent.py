import numpy as np
import math
import plotly.graph_objects as px
import random

class GreedySearch:
    def __init__(self, mars_map, scale, height_threshold=2):
        self.mars_map = mars_map
        self.scale = scale
        self.height_threshold = height_threshold

    def get_neighbors(self, row, col):
        movements = [(-1, 0),(1, 0),(0, -1),(0, 1),(-1, -1),(-1, 1),(1, -1),(1, 1)]
        valid_neighbors = []

        for dr, dc in movements:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < len(self.mars_map) 
                and 0 <= new_col < len(self.mars_map[0])
                and abs(self.mars_map[new_row][new_col] -self.mars_map[row][col]) <= self.height_threshold):
                valid_neighbors.append((new_row, new_col))
        return valid_neighbors

    def greedy_search(self, start_position):
        current_position = start_position
        path = [current_position]

        while True:
            neighbors = self.get_neighbors(*current_position)

            if not neighbors:
                break
            #vecino con mayor profunidad
            max_neighbor = max(neighbors,key=lambda neighbor: self.mars_map[neighbor[0]][neighbor[1]]) 
            if self.mars_map[max_neighbor[0]][max_neighbor[1]] <= self.mars_map[current_position[0]][current_position[1]]:
                break
            current_position = max_neighbor
            path.append(current_position)
        return path

def simulated_annealing(mars_map, scale, start_position, temperature=3, cooling_rate=0.99995):
    current_position = start_position
    current_cost = abs(mars_map[current_position[0]][current_position[1]])
    path = [current_position]

    while temperature > 0.005 :
        next_position = random.choice(GreedySearch(mars_map, scale).get_neighbors(*current_position))
        new_cost = abs(mars_map[next_position[0]][next_position[1]])

        if new_cost < current_cost:
            current_position = next_position
            current_cost = new_cost
        else:
            # Calculate probability of accepting the neighbor
            p = math.exp(-(new_cost - current_cost) / temperature)
            if p >= random.random():
                current_position = next_position
                current_cost = new_cost

        path.append(current_position)
        temperature *= cooling_rate

        print((temperature, current_cost))
    return path

# mapa del crater
crater_map = np.load('/Users/claudiogonzalezarriaga/Documents/Progra_Tec/CuartoSemestre/Agentes inteligentes/Descenso crater/mars_map_crater.npy')
nr , nc = crater_map.shape
# Posición inicial
start_x, start_y = 2591,4640 

# Escala
scale = 10.045

# posición inicial escalada
row_ini = nr - round(start_y / scale)
col_ini = round(start_x / scale)

#  GreedySearch
greedy_search_instance = GreedySearch(crater_map, scale)
path_greedy = greedy_search_instance.greedy_search((row_ini, col_ini))
print("Greedy Search Path:", path_greedy)

# Recocido de pollo
path_simulated_annealing = simulated_annealing(crater_map, scale, (row_ini, col_ini))
print("Simulated Annealing Path:", path_simulated_annealing)

## Generando mapa
if path_simulated_annealing != None:
    path_x = []
    path_y = []
    path_z = []
    prev_state = []
    distance = 0
    for i, state in enumerate(path_simulated_annealing):    
        path_x.append(state[1] * scale)            
        path_y.append((nr - state[0]) * scale)
        path_z.append(crater_map[state[0]][state[1]] + 1)
        
        if len(prev_state) > 0:
            distance += math.sqrt(
            scale * scale * (state[0] - prev_state[0]) ** 2 + scale * scale * (state[1] - prev_state[1]) ** 2 + (
                crater_map[state[0], state[1]] - crater_map[prev_state[0], prev_state[1]]) ** 2)
        prev_state = state

    print("Total distance", distance)

else:
    print("Unable to find a path between that connect the specified points")

## Plot results
if path_simulated_annealing != None: 

    x = scale*np.arange(crater_map.shape[1])
    y = scale*np.arange(crater_map.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = px.Figure(data = [px.Surface(x=X, y=Y, z=np.flipud(crater_map), colorscale='Hot', cmin = 0, 
                                        lighting = dict(ambient = 0.0, diffuse = 0.8, fresnel = 0.02, roughness = 0.4, specular = 0.2),
                                        lightposition=dict(x=0, y=nr/2, z=2*crater_map.max())),
                        
                            px.Scatter3d(x = path_x, y = path_y, z = path_z, name='path', mode='lines+markers',
                                            marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Bluered", size=4))],
                
                    layout = px.Layout(scene_aspectmode='manual', 
                                        scene_aspectratio=dict(x=1, y=nr/nc, z=max(crater_map.max()/x.max(), 0.2)), 
                                        scene_zaxis_range = [0,crater_map.max()])
                    )
    fig.show() 
