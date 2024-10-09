import numpy as np


def REF(matrix) -> np.array:
    matrix_ref = matrix.copy()
    rows, cols = matrix_ref.shape
    row = 0
    if matrix_ref.dtype == np.bool_:
        xor = lambda x, y: x ^ y
    else:
        xor = lambda x, y: (x + y) % 2

    for col in range(cols):
        pivot = None
        for r in range(row, rows):
            if matrix_ref[r, col] != 0:
                pivot = r
                break

        if pivot is None:
            continue

        if pivot != row:
            matrix_ref[row, pivot] = matrix_ref[pivot, row]

        for r in range(row + 1, rows):
            if matrix_ref[r, col] != 0:
                matrix_ref[r] = xor(matrix_ref[r], matrix_ref[row])

        row += 1

    return matrix_ref[~np.all(matrix_ref == 0, axis=1)]


def RREF(matrix: np.array, is_ref: bool = False) -> np.array:
    matrix_rref = matrix.copy() if is_ref else REF(matrix)
    rows, cols = matrix_rref.shape

    if matrix_rref.dtype == np.bool_:
        xor = lambda x, y: x ^ y
    else:
        xor = lambda x, y: (x + y) % 2

    for row in range(rows - 1, -1, -1):
        leading_col = None
        for col in range(cols):
            if matrix_rref[row, col] == 1:
                leading_col = col
                break

        if leading_col is None:
            continue

        for r in range(row):
            if matrix_rref[r, leading_col] != 0:
                matrix_rref[r] = xor(matrix_rref[r], matrix_rref[row])

    return matrix_rref


if __name__ == '__main__':
    matrix = np.array(
        [
            [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=np.int32
    )

    print(matrix, REF(matrix), RREF(matrix), sep='\n\n')
