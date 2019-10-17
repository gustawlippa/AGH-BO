from mpl_toolkits import mplot3d  # noqa: F401 unused import

import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class City:
    def __init__(self, x, y, z, value):
        self.x = x
        self.y = y
        self.z = z
        self.value = value

    def __repr__(self):
        return "X: {}, Y: {}, Z: {:.3}, V: {:.3}".format(self.x, self.y, self.z, self.value)


class World:
    def __init__(self, cities_no, size):
        self.cities_no = cities_no
        self.size = size
        self.height_mesh = self.create_height_mesh()
        self.value_mesh = self.create_value_mesh()
        self.cities = self.populate_world()
        print(self.cities)

        #  l = lambda x, y: 1 if (x, y) in [(c.x, c.y) for c in self.cities] else 0
        #  plot(size, np.vectorize(l))

        self.road_matrix, self.cost_matrix, self.value_matrix = self.create_road_matrices()
        print("No of roads: \n", np.count_nonzero(self.road_matrix))
        print("Roads: \n", self.road_matrix)
        print("Cost: \n", self.cost_matrix)
        print("Value: \n", self.value_matrix)

        sols = generate_solutions(0, cities_no-1, self.road_matrix, self.cost_matrix, self.value_matrix)
        print(sols)

        plt.show()

    def create_height_mesh(self):
        return self.create_somewhat_random_normalized_mesh()

    def create_value_mesh(self):
        return self.create_somewhat_random_normalized_mesh()

    def create_somewhat_random_normalized_mesh(self):
        x = np.arange(0, self.size)
        y = np.arange(0, self.size)
        X, Y = np.meshgrid(x, y)
        Z = (X + Y) * np.random.rand(self.size, self.size)
        max_z = np.amax(Z)
        Z = Z/max_z
        return Z

    def populate_world(self):
        n = self.cities_no
        size = self.size
        coords = random.sample([(x, y) for x in range(size) for y in range(size)], n)
        return [City(x, y, self.height_mesh[x, y], self.value_mesh[x, y]) for (x, y) in coords]

    def create_road_matrices(self):
        n = self.cities_no
        cities = self.cities
        cost_matrix = np.zeros([n, n])
        value_matrix = np.zeros([n, n])
        road_matrix = np.zeros([n, n])

        for x1, c1 in enumerate(cities):
            for y1, c2 in enumerate(cities):
                if c1 != c2:
                    x = x1-1
                    y = y1-1

                    dist_to_centre = ((c1.x - self.size/2) ** 2 + (c1.y - self.size/2) ** 2) ** (1.0 / 2)
                    radius = (dist_to_centre / 2) + (0.35 * self.size)  # the range of a city is ~40% of a total space
                    # print(radius, dist_to_centre, c1.x, c1.y)
                    city_dist = ((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2) ** (1.0 / 2)

                    if city_dist < radius:
                        road_matrix[x][y] = 1
                        cost_matrix[x][y] = self.get_cost(c1, c2)
                        value_matrix[x][y] = self.get_value(c1, c2)
        return road_matrix, cost_matrix, value_matrix

    def get_cost(self, c1, c2):
        dist = ((c1.x-c2.x)**2+(c1.y-c2.y)**2)**(1.0/2)
        cost = dist*(c2.z-c1.z) * random.uniform(0.7, 1.0)
        if cost < 0:
            return cost * (-0.6)
        else:
            return cost

    def get_value(self, c1, c2):
        dist = ((c1.x-c2.x)**2+(c1.y-c2.y)**2)**1.0/2
        value = dist*(c2.value + c1.value) * random.uniform(0.5, 1.0)
        return value


def generate_solutions(start, end, road_matrix, cost_matrix, value_matrix):
    b = bfs(start, end, road_matrix, [], [])
    # b_max = bfs_max(start, end, value_matrix)
    return b


def bfs(start, end, matrix, result, visited):
    Q = []
    Q.append([start])
    while Q:
        path = Q.pop()
        city = path[-1]
        if city == end:
            return path
        elif city not in visited:
            for c in neighbours(city, matrix):
                new_path = list(path)
                new_path.append(c)
                Q.append(new_path)
            visited.append(city)
    return []


def neighbours(start, matrix):
    return [end for end in range(len(matrix[start])) if matrix[start][end] and start!=end]


def plot(size, z):
    # print(Z, type(Z), Z.shape)

    x = np.arange(0, size, 0.1)
    y = np.arange(0, size, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = z(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z,
                       linewidth=0, antialiased=False, cmap=cm.coolwarm)

    # ax.scatter([3], [3], [1], 'o')
    plt.show()


def main(cities_no=10, world_size=10, world_height=100):

    w = World(10, 10)


if __name__=='__main__':
    main()