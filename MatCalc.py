from Matrix import Matrix


def matmul(mt1, mt2):
    if mt1.cols != mt2.rows:
        raise ValueError
    newMt = Matrix(mt2.cols, mt1.rows)
    for i in range(newMt.rows):
        for j in range(newMt.cols):
            for k in range(mt1.cols):
                newMt.vals[i][j] += mt1.vals[i][k] * mt2.vals[k][j]
    return newMt