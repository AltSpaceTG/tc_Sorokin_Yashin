import numpy as np
import itertools
class LinearCode:
    def __init__(self, matrix: np.array):
        self.matrix = matrix
        self.matrix_ref = None
    
    def ref(self) -> np.array:
        self.matrix_ref = self.matrix.copy()
        rows, cols = self.matrix.shape
        row = 0
        if self.matrix.dtype == np.bool_:
            self.__xor_func = lambda x, y: np.bitwise_xor(x, y)
        else:
            self.__xor_func = lambda x, y: (x + y) % 2
        
        for col in range(cols):
            pivot = None
            for r in range(row, rows):
                if self.matrix_ref[r, col] != 0:
                    pivot = r
                    break
            
            if pivot is None:
                continue
            
            if pivot !=  row:
                self.matrix_ref[row, pivot] = self.matrix_ref[pivot, row]
                
            for r in range(row + 1,rows):
                if self.matrix_ref[r, col] != 0:
                    self.matrix_ref[r] = self.__xor_func(self.matrix_ref[r], self.matrix_ref[row])
            
            row += 1
        
        self.matrix_ref = self.matrix_ref[~np.all(self.matrix_ref==0, axis=1)]
        self.k, self.n = self.matrix_ref.shape
        return  self.matrix
    
    def get_matrix_ref_size(self):
        return self.k, self.n
    
    def create_checker_matrix(self):
        matrix_rref = self.matrix_ref.copy() #1.3.3.1
        leading_cols = []
        for row in range(self.k - 1, -1, -1):  
            leading_col = None
            for col in range(self.n):
                if matrix_rref[row, col] == 1:
                    leading_col = col
                    #1.3.3.2
                    leading_cols.append(leading_col)
                    break

            if leading_col is None:
                continue  

            
            for r in range(row):
                if matrix_rref[r, leading_col] != 0:
                    matrix_rref[r] = self.__xor_func(matrix_rref[r],matrix_rref[row])

        X = np.delete(matrix_rref, leading_cols, axis=1) # 1.3.3.3
        
        I = np.eye(self.n - len(leading_cols), dtype=int) 
        
        H = np.zeros((self.n, self.n - self.k), dtype=int)
        #1.3.3.4
        row_idx = 0
        for i in range(self.n):
            if i in leading_cols:
                H[i] = X[row_idx]
                row_idx += 1
            else:
                H[i] = I[i - row_idx]
        self.H = H
        return H

    def generate_keys_by_combinations(self):
        # Инициализация с нулевым словом
        final_words = {tuple(np.zeros(self.matrix.shape[1], dtype=int))}

        # Проход по всем возможным комбинациям строк матрицы
        for i in range(1, self.matrix.shape[0] + 1):
            for indices in itertools.combinations(range(self.matrix.shape[0]), i):
                word = np.zeros(self.matrix.shape[1], dtype=int)
                # Сложение строк по выбранным индексам
                word = sum(self.matrix[j, :] for j in indices) % 2
                final_words.add(tuple(word))

        return np.array(list(final_words))

    def generate_keys_by_K(self):

        k, n = self.get_matrix_ref_size()
        # Перебор всех возможных двоичных векторов длины k
        final_code = itertools.product(range(2), repeat=k)
        return [np.dot(code, self.matrix) % 2 for code in final_code]

    
        
            
            
        
        
        
        
if __name__ == "__main__":
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
    #1.3
    lc = LinearCode(matrix)
    #1.3.1
    G = lc.ref()
    print(G)
    #1.3.2
    print(lc.get_matrix_ref_size())
    #1.3.3
    H = lc.create_checker_matrix()
    print(H)

    #1.4.1
    words_from_combinations = lc.generate_keys_by_combinations()
    #1.4.2
    words_from_k = lc.generate_keys_by_K()

    print({tuple(word.tolist()) for word in words_from_combinations} == {tuple(word.tolist()) for word in words_from_k})

    #1.4
    d = np.min(np.sum(G, axis=1))
    t = d - 1


    # Пример значений
    v = np.array([1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0])
    e1 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    # Сложение вектора v с вектором ошибки e1
    s = (v + e1) % 2
    print(f'v + e1 = {s}')



    s2 = (s @ H) % 2
    print(f'(v + e1)@H = {s2} - error')

    # Поиск ошибки с двумя ошибочными битами
    is_find = False
    print(f'v = {v}')

    # Поиск двухбитной ошибки
    for i in range(len(v)):
        if is_find:
            break
        for j in range(i + 1, len(v)):
            e2 = np.zeros(len(v), dtype=int)
            e2[i], e2[j] = 1, 1

            s = (v + e2) % 2
            s2 = (s @ H) % 2
            if np.sum(s2) == 0:
                is_find = True
                break

    print(f'e2 = {e2}')
    print(f'v + e2 = {s}')
    print(f'(v + e2)@H = {s2} - no error')






