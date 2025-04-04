from Matrix import Vector
from MatCalc import matmul

def GaussSeidel(a, b):
    if a.cols != b.rows and b.cols != 1:
        raise ValueError
    x = Vector(a.cols)
    for i in range(x.rows):
        x.vals[i][0] = 0
    iters = 0
    norm = 0
    while iters < 100:
        iters += 1
        newX = Vector(a.cols)
        for i in range(newX.rows):
            val = b.vals[i][0]
            for j in range(i):
                val -= a.vals[i][j] * newX.vals[j][0]
            for j in range(i + 1, newX.rows):
                val -= a.vals[i][j] * x.vals[j][0]
            val /= a.vals[i][i]
            newX.vals[i][0] = val
        x = newX
        res = matmul(a, x) - b
        norm = res.norm()
        if norm < 10e-9:
            break
    return x