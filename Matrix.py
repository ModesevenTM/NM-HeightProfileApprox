from copy import deepcopy

class Matrix:
    def __init__(self, cols, rows):
        self.cols = cols
        self.rows = rows
        self.vals = [[0 for _ in range(cols)] for _ in range(rows)]

    def __str__(self):
        str = ""
        for i in range(self.rows):
            for j in range(self.cols):
                str += f"{self.vals[i][j]} "
            str += "\n"
        return str

    def __sub__(self, other):
        if self.cols != other.cols or self.rows != other.rows:
            raise ValueError
        mat = deepcopy(self)
        for i in range(self.rows):
            for j in range(self.cols):
                mat.vals[i][j] -= other.vals[i][j]
        return mat

    def norm(self):
        e = 0
        for i in range(self.rows):
            e += self.vals[i][0] ** 2
        return e ** (1/2)

    def get_column(self, n):
        return [x[n - 1] for x in self.vals]

    def get_row(self, n):
        return self.vals[n - 1]


class BandMatrix(Matrix):
    def __init__(self, N, a1, a2, a3):
        super().__init__(N, N)
        for i in range(N):
            self.vals[i][i] = a1
            if i - 1 >= 0:
                self.vals[i][i - 1] = a2
                if i - 2 >= 0:
                    self.vals[i][i - 2] = a3
            if i + 1 < N:
                self.vals[i][i + 1] = a2
                if i + 2 < N:
                    self.vals[i][i + 2] = a3


class Vector(Matrix):
    def __init__(self, N):
        super().__init__(1, N)

class SpEye(Matrix):
    def __init__(self, N):
        super().__init__(N, N)
        for i in range(N):
            self.vals[i][i] = 1