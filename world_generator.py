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
    def __init__(self, cities_no, size, height_max, value_max):
        self.cities_no = cities_no
        self.size = size
        self.height_mesh = self.create_height_mesh(height_max)
        self.value_mesh = self.create_value_mesh(value_max)
        self.cities = self.populate_world()
        print(self.cities)

        l = lambda x, y: 1 if (x, y) in [(c.x, c.y) for c in self.cities] else 0
        plot(size, np.vectorize(l))

        self.price_matrix, self.value_matrix = self.create_roads_matrics()

        print("Price: ", self.price_matrix,"Value: ", self.value_matrix)

        plt.show()


    def create_height_mesh(self, height_max):
        x = np.arange(0, self.size)
        y = np.arange(0, self.size)
        X, Y = np.meshgrid(x, y)
        # TODO placeholder height function
        Z = (X + Y)
        max_z = np.amax(Z)
        Z = Z*height_max/max_z
        return Z

    def create_value_mesh(self, value_max):
        return self.create_height_mesh(value_max)

    def populate_world(self):
        n = self.cities_no
        size = self.size
        coords = random.sample([(x, y) for x in range(size) for y in range(size)], n)
        return [City(x, y, self.height_mesh[x,y], self.value_mesh[x,y]) for (x,y) in coords]

    def create_roads_matrics(self):
        n = self.cities_no
        cities = self.cities
        price_matrix = np.ones([n, n])
        value_matrix = np.zeros([n, n])

        # d= [maybe_create_road(c1,c2)  for c1 in self.cities for c2 in self.cities if c1!=c2]

        for x1, c1 in enumerate(cities):
            for y1, c2 in enumerate(cities):
                if c1 != c2:
                    x=x1-1
                    y=y1-1
                    # implement logic for choosing when to put a road
                    if random.random() > 0.5:
                        price_matrix[x][y] = self.get_price(c1, c2)
                        value_matrix[x][y] = self.get_value(c1, c2)
        return price_matrix, value_matrix

    def get_price(self, c1, c2):
        dist = ((c1.x-c2.x)**2+(c1.y-c2.y)**2)**1.0/2
        price = dist*(c2.z-c1.z)
        if price <= 0:
            return dist*0.01
        else:
            return price

    def get_value(self, c1, c2):
        dist = ((c1.x-c2.x)**2+(c1.y-c2.y)**2)**1.0/2
        value = dist*abs(c2.value-c1.value)
        return value


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