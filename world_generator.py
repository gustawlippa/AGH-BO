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


class World:
    def __init__(self, cities_no, size, height_max, value_max):
        self.cities_no = cities_no
        self.size = size
        self.height_mesh = self.create_height_mesh(height_max)
        self.value_mesh = self.create_value_mesh(value_max)
        self.cities = self.populate_world()
        print(self.cities)

        l = lambda x, y: 1 if (x, y) in self.cities else 0
        plot(size, np.vectorize(l))

        self.cities_matrix = self.create_cities_matrix()

    def create_height_mesh(self, height_max):
        x = np.arange(0, self.size)
        y = np.arange(0, self.size)
        X, Y = np.meshgrid(x, y)
        print(type(X), X, X.shape)
        # TODO placeholder height function
        Z = (X + Y)
        max_z = np.amax(Z)
        Z = Z*height_max/max_z
        print(Z, type(Z), Z.shape)
        return Z

    def create_value_mesh(self, value_max):
        return self.create_height_mesh(value_max)

    def populate_world(self):
        n = self.cities_no
        size = self.size
        return random.sample([(x, y) for x in range(size) for y in range(size)], n)

    def create_cities_matrix(self):
        return []


def plot(size, z):
    # print(Z, type(Z), Z.shape)

    x = np.arange(0, size)
    y = np.arange(0, size)
    X, Y = np.meshgrid(x, y)
    Z = z(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z,
                       linewidth=0, antialiased=False)
    plt.show()


def main(cities_no=10, world_size=10, world_height=100):

    w = World(10, 10, 10, 10)

    # x = np.arange(0, world_size, 0.1)
    # y = np.arange(0, world_size, 0.1)
    # X, Y = np.meshgrid(x, y)
    # Z = X + Y
    # print(Z)
    # for x in range(13):
    #     for y in range(13):
    #         Z[x][y] = 13
    #
    # surf = ax.plot_surface(X, Y, Z,
    #                    linewidth=0, antialiased=False)
    #
    # # h = plt.contourf(x, y, Z)
    # plt.show()
    #
    # print(random.random())


if __name__=='__main__':
    main()