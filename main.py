import heapq
import networkx as nx
import matplotlib.pyplot as plt


class Prim:

    def __init__(self, vertex_count):
        self.vertex_count = vertex_count
        self.graph = [[0 for _ in range(vertex_count)] for _ in range(vertex_count)]
        self.vertex_data = [''] * vertex_count

    def add_vertex_data(self, vertex, data):
        if 0 <= vertex < self.vertex_count:
            self.vertex_data[vertex] = data

    def add_edge(self, u, v, weight):
        if 0 <= u < self.vertex_count and 0 <= v < self.vertex_count:
            self.graph[u][v] = weight
            self.graph[v][u] = weight

    def showTree(self, vertex_parent):
        print("Edge \tWeight")
        for i in range(1, self.vertex_count):
            print(vertex_parent[i], "-", i, "\t", self.graph[i][vertex_parent[i]])

    def findMinKey(self, key, mstIncluded):
        min_val = float('inf')
        min_index = None
        for v in range(self.vertex_count):
            if key[v] < min_val and not mstIncluded[v]:
                min_val = key[v]
                min_index = v
        return min_index

    def createMST_array(self):
        key = [float('inf')] * self.vertex_count
        vertex_parent = [-1] * self.vertex_count
        key[0] = 0
        mstIncluded = [False] * self.vertex_count
        for _ in range(self.vertex_count):
            u = self.findMinKey(key, mstIncluded)
            mstIncluded[u] = True
            for v in range(self.vertex_count):
                if 0 < self.graph[u][v] < key[v] and not mstIncluded[v]:
                    key[v] = self.graph[u][v]
                    vertex_parent[v] = u

        return vertex_parent

    def createMST(self):
        key = [float('inf')] * self.vertex_count
        parent = [-1] * self.vertex_count
        key[0] = 0
        mstIncluded = [False] * self.vertex_count
        # parent[0] = -1  # to record the path of the MST
        pq = []
        for v in range(self.vertex_count):
            heapq.heappush(pq, (key[v], v))  # pushes key and vertex

        while pq:
            u = heapq.heappop(pq)[1]  # Extract the vertex with the minimum key value,[1] is the indx of vrtx
            mstIncluded[u] = True  # included
            for v in range(self.vertex_count):
                if 0 < self.graph[u][v] < key[v] and not mstIncluded[v]:  # for all vrtx less than corresponding key
                    key[v] = self.graph[u][v]  # update key with the smaller weigh
                    parent[v] = u
                    heapq.heappush(pq, (key[v], v))  # push the new key to the heap

        return parent

    def drawTree(self, parent):
        # Visualizes the whole graph using the NetworkX library
        G = nx.Graph()
        for i in range(self.vertex_count):
            for j in range(i, self.vertex_count):
                if self.graph[i][j] != 0:
                    G.add_edge(self.vertex_data[i], self.vertex_data[j], weight=self.graph[i][j])

        pos = nx.circular_layout(G)

        nx.draw_networkx_nodes(G, pos, node_size=700)
        nx.draw_networkx_edges(G, pos, width=6, edge_color='grey')

        # Highlight the MST edges
        for i in range(1, self.vertex_count):
            nx.draw_networkx_edges(G, pos, edgelist=[(self.vertex_data[parent[i]], self.vertex_data[i])],
                                   width=6, edge_color='r')

        nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

        edge_labels = dict([((u, v,), d['weight']) for u, v, d in G.edges(data=True)])
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.axis('off')
        plt.show()


class Dijkstra:

    def __init__(self, vertex_count):
        self.vertex_count = vertex_count
        self.graph = [[0 for _ in range(vertex_count)] for _ in range(vertex_count)]
        self.vertex_data = [''] * vertex_count

    def add_edge(self, u, v, weight):
        if 0 <= u < self.vertex_count and 0 <= v < self.vertex_count:
            self.graph[u][v] = weight  # For directed graph

    def add_vertex_data(self, vertex, data):
        if 0 <= vertex < self.vertex_count:
            self.vertex_data[vertex] = data

    def shortestPath(self, start_data):
        start_vertex = self.vertex_data.index(start_data)  # Find the index of the start vertex
        distances = [float('inf')] * self.vertex_count
        visited = [False] * self.vertex_count
        predecessors = [-1] * self.vertex_count
        distances[start_vertex] = 0
        pq = []
        for v in range(self.vertex_count):
            heapq.heappush(pq, (distances[v], v))

        while pq:  # While the priority queue is not empty
            u = heapq.heappop(pq)[1]
            visited[u] = True
            for v in range(self.vertex_count):
                if distances[v] > distances[u] + self.graph[u][v] > 0 != self.graph[u][v] and not visited[v]:  # If the distance is shorter
                    distances[v] = distances[u] + self.graph[u][v]  # Update the distance
                    heapq.heappush(pq, (distances[v], v))  # Push the new distance to the priority queue
                    predecessors[v] = u

        return distances, predecessors

    def shortestPath_array(self, start_vertex_data):
        start_vertex = self.vertex_data.index(start_vertex_data)
        distances = [float('inf')] * self.vertex_count
        distances[start_vertex] = 0
        visited = [False] * self.vertex_count
        predecessors = [-1] * self.vertex_count

        for _ in range(self.vertex_count):
            min_distance = float('inf')
            u = None
            for i in range(self.vertex_count):
                if not visited[i] and distances[i] < min_distance:
                    min_distance = distances[i]
                    u = i

            if u is None:
                break

            visited[u] = True

            for v in range(self.vertex_count):
                if self.graph[u][v] != 0 and not visited[v]:
                    alt = distances[u] + self.graph[u][v]
                    if alt < distances[v]:
                        distances[v] = alt
                        predecessors[v] = u  # Update the predecessor of v

        return distances, predecessors  # Return both distances and predecessors

    def visualizeShortestPath(self, start_vertex_data, end_vertex_data):
        distances, predecessors = self.shortestPath(start_vertex_data)
        start_vertex = self.vertex_data.index(start_vertex_data)
        end_vertex = self.vertex_data.index(end_vertex_data)

        # Construct the shortest path
        path = []
        i = end_vertex
        while i != start_vertex:  # we append to path from destination to source
            path.append(i)
            i = predecessors[i]
        path.append(start_vertex)
        path = path[::-1]  # Reverse the path

        # Calculate the total distance of the shortest path
        total_distance = distances[end_vertex]

        # Visualize the whole graph
        G = nx.DiGraph()  # Directed graph from networkx python libr
        for i in range(self.vertex_count):  # Add the vertices
            for j in range(self.vertex_count):
                if self.graph[i][j] != 0:
                    G.add_edge(self.vertex_data[i], self.vertex_data[j], weight=self.graph[i][j])

        pos = nx.planar_layout(G)  # this lays out the vertices, planar is the best layout I found for this

        nx.draw_networkx_nodes(G, pos, node_size=700)
        nx.draw_networkx_edges(G, pos, width=6, edge_color='grey',
                               connectionstyle='arc3,rad=0.1', arrows=True)  # I made it curve so no overlap occurs

        # Highlight the shortest path edges
        for i in range(len(path) - 1):  # colors the shortest path the shortest path
            nx.draw_networkx_edges(G, pos, edgelist=[(self.vertex_data[path[i]], self.vertex_data[path[i + 1]])],
                                   width=6, edge_color='r', connectionstyle='arc3,rad=0.1',
                                   arrows=True)

        nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
        edge_labels = {}
        f = False
        for u, v, d in G.edges(data=True):
            edge = (u, v,)
            weight = d['weight']
            if G.has_edge(v,
                          u):  # I didn't know how point the arrow to the correct direction when two verteces are pointing to each other
                if f:  # the if statement is to alternate the direction of the arrows, 3ashan tnfa3 bas fl test case bta3etna
                    edge_labels[edge] = str(weight) + "←"
                    edge_labels[(v, u,)] = "→" + str(G[v][u]['weight'])
                    f = False
                else:
                    edge_labels[edge] = str(weight) + "→"
                    edge_labels[(v, u,)] = "←" + str(G[v][u]['weight'])
                    f = True
            else:
                edge_labels[edge] = str(weight)

        # Calculate the mid-point of each edge
        edge_labels_pos = {}
        for edge in G.edges():  # Calculate the mid-point of each edge 3ashan a7ot two labels
            u, v = edge
            edge_labels_pos[edge] = [(pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2]

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3)

        # Display the total distance of the shortest path in the corner of the graph
        plt.text(0.05, 0.95, 'Shortest Path Distance: ' + str(total_distance), transform=plt.gca().transAxes,
                 fontsize=14,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        plt.axis('off')
        plt.show()


prim_graph = Prim(9)
prim_graph.add_vertex_data(0, 'a')
prim_graph.add_vertex_data(1, 'b')
prim_graph.add_vertex_data(2, 'v')
prim_graph.add_vertex_data(3, 'd')
prim_graph.add_vertex_data(4, 'e')
prim_graph.add_vertex_data(5, 'f')
prim_graph.add_vertex_data(6, 'g')
prim_graph.add_vertex_data(7, 'h')
prim_graph.add_vertex_data(8, 'i')

# Add edges between each vertex and the one before
prim_graph.add_edge(1, 0, 4)  # B - A
prim_graph.add_edge(2, 1, 8)  # C - B
prim_graph.add_edge(3, 2, 7)  # D - C
prim_graph.add_edge(4, 3, 9)  # E - D
prim_graph.add_edge(5, 4, 10)  # F - E
prim_graph.add_edge(6, 5, 2)  # G - F
prim_graph.add_edge(7, 6, 1)  # H - G
prim_graph.add_edge(8, 7, 7)  # I - H

prim_graph.add_edge(7, 0, 8)  # H - A
prim_graph.add_edge(1, 7, 11)  # B - H
prim_graph.add_edge(8, 6, 6)  # I - G
prim_graph.add_edge(2, 5, 4)  # C - F
prim_graph.add_edge(2, 8, 2)  # C - I
prim_graph.add_edge(3, 5, 14)  # D - F

vertex_parent = prim_graph.createMST()
prim_graph.drawTree(vertex_parent)

# Test Dijkstra class
dijkstra_graph = Dijkstra(5)

dijkstra_graph.add_vertex_data(0, 's')
dijkstra_graph.add_vertex_data(1, 't')
dijkstra_graph.add_vertex_data(2, 'x')
dijkstra_graph.add_vertex_data(3, 'y')
dijkstra_graph.add_vertex_data(4, 'z')

# Add the edges with their weights
dijkstra_graph.add_edge(0, 1, 10)  # S - T
dijkstra_graph.add_edge(1, 2, 1)  # T - X
dijkstra_graph.add_edge(2, 4, 4)  # X - Z
dijkstra_graph.add_edge(4, 2, 6)  # Z - X
dijkstra_graph.add_edge(3, 4, 2)  # Y - Z
dijkstra_graph.add_edge(0, 3, 5)  # S - Y
dijkstra_graph.add_edge(4, 0, 7)  # Z - S
dijkstra_graph.add_edge(3, 2, 9)  # Y - X
dijkstra_graph.add_edge(1, 3, 2)  # T - Y
dijkstra_graph.add_edge(3, 1, 3)  # Y - T

# Visualize the shortest path from S to all other vertices
dijkstra_graph.visualizeShortestPath('s', 'x')
dijkstra_graph.visualizeShortestPath('s', 'z')

# prim_graph = Prim(10)
# prim_graph.add_vertex_data(0, 'a')
# prim_graph.add_vertex_data(1, 'b')
# prim_graph.add_vertex_data(2, 'c')
# prim_graph.add_vertex_data(3, 'd')
# prim_graph.add_vertex_data(4, 'e')
# prim_graph.add_vertex_data(5, 'f')
# prim_graph.add_vertex_data(6, 'g')
# prim_graph.add_vertex_data(7, 'h')
# prim_graph.add_vertex_data(8, 'i')
# prim_graph.add_vertex_data(9, 'j')
#
# prim_graph.add_edge(0, 1, 4)
# prim_graph.add_edge(0, 7, 8)
# prim_graph.add_edge(1, 2, 8)
# prim_graph.add_edge(1, 7, 11)
# prim_graph.add_edge(2, 3, 7)
# prim_graph.add_edge(2, 8, 2)
# prim_graph.add_edge(2, 5, 4)
# prim_graph.add_edge(3, 4, 9)
# prim_graph.add_edge(3, 5, 14)
# prim_graph.add_edge(4, 5, 10)
# prim_graph.add_edge(5, 6, 2)
# prim_graph.add_edge(6, 7, 1)
# prim_graph.add_edge(6, 8, 6)
# prim_graph.add_edge(7, 8, 7)
# prim_graph.add_edge(8, 9, 5)
# prim_graph.add_edge(9, 4, 3)
#
# vertex_parent = prim_graph.createMST()
# prim_graph.drawTree(vertex_parent)
#
# # Complex test case for Dijkstra class
# dijkstra_graph = Dijkstra(6)
#
# dijkstra_graph.add_vertex_data(0, 's')
# dijkstra_graph.add_vertex_data(1, 't')
# dijkstra_graph.add_vertex_data(2, 'x')
# dijkstra_graph.add_vertex_data(3, 'y')
# dijkstra_graph.add_vertex_data(4, 'z')
# dijkstra_graph.add_vertex_data(5, 'w')
#
# dijkstra_graph.add_edge(0, 1, 10)
# dijkstra_graph.add_edge(0, 3, 5)
# dijkstra_graph.add_edge(1, 2, 1)
# dijkstra_graph.add_edge(1, 3, 2)
# dijkstra_graph.add_edge(2, 4, 4)
# dijkstra_graph.add_edge(3, 1, 3)
# dijkstra_graph.add_edge(3, 2, 9)
# dijkstra_graph.add_edge(3, 4, 2)
# dijkstra_graph.add_edge(3, 5, 7)
# dijkstra_graph.add_edge(4, 0, 7)
# dijkstra_graph.add_edge(4, 5, 6)
# dijkstra_graph.add_edge(5, 4, 1)
#
# dijkstra_graph.visualizeShortestPath('s', 'x')
# dijkstra_graph.visualizeShortestPath('s', 'z')
# dijkstra_graph.visualizeShortestPath('s', 'w')