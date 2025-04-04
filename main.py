import matplotlib.pyplot as plt
from Matrix import Matrix, Vector
from MatMethods import GaussSeidel
from math import cos, pi

tczew = {
    "name": "Tczew - Starogard Gdański",
    "data": [[float(a) for a in x.strip().split(" ")] for x in open("tczew_starogard.txt", "r").readlines()]
    }
everest = {
    "name": "Mount Everest",
    "data": [[float(a) for a in x.strip().split(",")] for x in open("MountEverest.csv", "r").readlines()[1:]]
    }

def linspace(start, stop, n):
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]

def regular_nodes(data, n):
    return [int(a) for a in linspace(0, len(data) - 1, n)]

def chebyshev_nodes(data, n):
    return [int((len(data) - 1) / 2 * (1 + cos((2 * i + 1) / (2 * n) * pi))) for i in range(n)]

def mul(list):
    result = 1
    for x in list:
        result *= x
    return result

def lagrange_mul(x, y, x0):
    return sum([y[i] * mul([(x0 - x[j]) / (x[i] - x[j]) for j in range(len(x)) if i != j]) for i in range(len(x))])

def lagrange(data, original_nodes, interpolated_nodes, node_generator=regular_nodes):
    print("Wykonuję interpolację Lagrange'a", end="")
    if node_generator == chebyshev_nodes:
        print(" z węzłami Czebyszewa", end="")
    print(f" ({original_nodes} węzłów) dla trasy {data['name']}... ", end="")
    x = [a[0] for a in data["data"]]
    y = [a[1] for a in data["data"]]
    indexes_to_interpolate = node_generator(x, original_nodes)
    x_to_interpolate = [x[i] for i in indexes_to_interpolate]
    y_to_interpolate = [y[i] for i in indexes_to_interpolate]
    x_interpolated = linspace(x[0], x[-1], interpolated_nodes)
    y_interpolated = [lagrange_mul(x_to_interpolate, y_to_interpolate, x0) for x0 in x_interpolated]

    plt.plot(x, y, label="Oryginalne dane")
    plt.scatter([x[i] for i in indexes_to_interpolate], [y[i] for i in indexes_to_interpolate], color="red", label="Węzły interpolacji")
    plt.plot(x_interpolated, y_interpolated, label="Interpolacja Lagrange'a")
    plt.xlabel("Dystans [m]")
    plt.ylabel("Wysokość [m]")
    if node_generator == regular_nodes:
        plt.title(f"{data['name']} - interpolacja Lagrange'a, {original_nodes} węzłów")
    elif node_generator == chebyshev_nodes:
        plt.title(f"{data['name']} - interp. Lagrange'a, {original_nodes} węzłów Czebyszewa")
    plt.legend()
    if node_generator == regular_nodes:
        plt.savefig(f"plots\\{data['name']}_lagrange_{original_nodes}.png")
    elif node_generator == chebyshev_nodes:
        plt.savefig(f"plots\\{data['name']}_lagrange+chebyshev_{original_nodes}.png")
    plt.close()

    if node_generator == regular_nodes and original_nodes >= 20:
        plt.plot(x, y, label="Oryginalne dane")
        plt.scatter([x[i] for i in indexes_to_interpolate], [y[i] for i in indexes_to_interpolate], color="red", label="Węzły interpolacji")
        plt.plot(x_interpolated, y_interpolated, label="Interpolacja Lagrange'a")
        plt.xlabel("Dystans [m]")
        plt.ylabel("Wysokość [m]")
        plt.title(f"{data['name']} - interpolacja Lagrange'a, {original_nodes} węzłów")
        plt.legend()
        plt.ylim(min(y) / 1.02, max(y) * 1.02)
        plt.savefig(f"plots\\{data['name']}_lagrange_ylim_{original_nodes}.png")
        plt.close()
        
    print("wykonano")

def cubic_splines(data, original_nodes, interpolated_nodes):
    print(f"Wykonuję interpolację splajnami 3. stopnia ({original_nodes} węzłów) dla trasy {data['name']}... ", end="")
    x = [a[0] for a in data["data"]]
    y = [a[1] for a in data["data"]]
    indexes_to_interpolate = [int(a) for a in linspace(0, len(x) - 1, original_nodes)]
    x_to_interpolate = [x[i] for i in indexes_to_interpolate]
    y_to_interpolate = [y[i] for i in indexes_to_interpolate]
    n = len(x_to_interpolate)
    h = [x_to_interpolate[i] - x_to_interpolate[i - 1] for i in range(1, n)]
    A = Matrix(n, n)
    for i in range(1, n - 1):
        A.vals[i][i - 1] = h[i - 1]
        A.vals[i][i] = 2 * (h[i - 1] + h[i])
        A.vals[i][i + 1] = h[i]
    A.vals[0][0] = 1
    A.vals[n - 1][n - 1] = 1
    B = Vector(n)
    B.vals = [[0]] + [[3 * (y_to_interpolate[i + 1] - y_to_interpolate[i]) / h[i] - 3 * (y_to_interpolate[i] - y_to_interpolate[i - 1]) / h[i - 1]] for i in range(1, n - 1)] + [[0]]
    c = [x[0] for x in GaussSeidel(A, B).vals]
    b = [(y_to_interpolate[i] - y_to_interpolate[i - 1]) / h[i - 1] - h[i - 1] * (c[i] + 2 * c[i - 1]) / 3 for i in range(1, n)] + [0]
    d = [(c[i] - c[i - 1]) / (3 * h[i - 1]) for i in range(1, n)] + [0]
    x_interpolated = linspace(x[0], x[-1], interpolated_nodes)
    y_interpolated = []
    for x0 in x_interpolated:
        for i in range(1, n):
            if x_to_interpolate[i - 1] <= x0 <= x_to_interpolate[i]:
                h = x0 - x_to_interpolate[i - 1]
                y0 = y_to_interpolate[i - 1] + b[i - 1] * h + c[i - 1] * h ** 2 + d[i - 1] * h ** 3
                y_interpolated.append(y0)
                break
            
    plt.plot(x, y, label="Oryginalne dane")
    plt.scatter(x_to_interpolate, y_to_interpolate, color="red", label="Węzły interpolacji")
    plt.plot(x_interpolated, y_interpolated, label="Interpolacja splajnami")
    plt.xlabel("Dystans [m]")
    plt.ylabel("Wysokość [m]")
    plt.title(f"{data['name']} - interpolacja splajnami 3. stopnia, {original_nodes} węzłów")
    plt.legend()
    plt.savefig(f"plots\\{data['name']}_splines_{original_nodes}.png")
    plt.close()
    print("wykonano")

for i in [5, 10, 20, 40]:
    for input in [tczew, everest]:
        lagrange(input, i, 1000)
        lagrange(input, i, 1000, chebyshev_nodes)
        cubic_splines(input, i, 1000)