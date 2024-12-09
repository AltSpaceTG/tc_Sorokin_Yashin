import numpy as np
from itertools import combinations, product
from math import comb
import random


class ReedMullerCode:
    def __init__(self, order, variables):
        self.order = order
        self.variables = variables
        self.generator_matrix = self._construct_generator_matrix()

    def _generate_basis_vector(self, indices, length):
        if not indices:
            return np.ones(2 ** length, dtype=int)

        idx_array = np.arange(2 ** length)
        binary_representation = (idx_array[:, None] >> np.arange(length)[::-1]) & 1
        product_result = np.prod((binary_representation[:, indices] + 1) % 2, axis=1)
        return product_result

    def _construct_generator_matrix(self):
        total_rows = sum(comb(self.variables, k) for k in range(self.order + 1))
        matrix = np.zeros((total_rows, 2 ** self.variables), dtype=int)
        row = 0

        for subset_size in range(self.order + 1):
            for subset in combinations(range(self.variables), subset_size):
                matrix[row] = self._generate_basis_vector(subset, self.variables)
                row += 1

        return matrix

    def display_matrix(self):
        print("Генераторная матрица Рида-Маллера:")
        print(self.generator_matrix)


class ErrorSimulator:
    def __init__(self, generator_matrix):
        self.G = generator_matrix

    def introduce_errors(self, original_vector, num_errors):
        encoded = self._encode(original_vector)
        print(f"Исходное сообщение: {encoded}")

        error_vector = np.zeros(encoded.size, dtype=int)
        error_positions = random.sample(range(encoded.size), num_errors)
        error_vector[error_positions] = 1
        received = (encoded + error_vector) % 2
        print(f"Сообщение с ошибками: {received}")
        return received

    def _encode(self, message):
        return np.dot(message, self.G) % 2


class Decoder:
    def __init__(self, generator_matrix, matrix_size, reference_level, total_length):
        self.G = generator_matrix
        self.size = matrix_size
        self.ref_level = reference_level
        self.length = total_length

    def _sorted_combinations(self, n, k):
        return list(combinations(range(n), k))

    def _generate_patterns(self, subsets, size):
        patterns = []
        for subset in subsets:
            for w in product([0, 1], repeat=size):
                if np.prod([(w[j] + 1) % 2 for j in subset]) == 1:
                    patterns.append(np.array(w, dtype=int))
        return patterns

    def decode(self, received_word):
        counter = 0
        current_ref = self.ref_level
        corrected_output = np.zeros(self.length, dtype=int)
        max_wt = 2 ** (self.size - current_ref - 1) - 1
        current = received_word.copy()

        while True:
            for subset in self._sorted_combinations(self.size, current_ref):
                max_threshold = 2 ** (self.size - current_ref - 1)
                zero_count = one_count = 0
                possible_matrices = self._generate_patterns([subset], self.size)

                for transformation in possible_matrices:
                    remaining_indices = [j for j in range(self.size) if j not in subset]
                    binary_products = self._compute_binary_products(remaining_indices, self.size, transformation)

                    if not ((current @ binary_products) % 2):
                        zero_count += 1
                    else:
                        one_count += 1

                if zero_count > max_wt and one_count > max_wt:
                    return None

                if zero_count > max_threshold:
                    corrected_output[counter] = 0
                    counter += 1
                if one_count > max_threshold:
                    vector = self._generate_basis_vector(subset, self.size)
                    current = (current + vector) % 2
                    corrected_output[counter] = 1
                    counter += 1

            if current_ref > 0:
                if len(current) < max_wt:
                    for subset in self._sorted_combinations(self.size, self.ref_level + 1):
                        corrected_output[counter] = 0
                        counter += 1
                    break
                current_ref -= 1
            else:
                break

        return corrected_output

    def _compute_binary_products(self, remaining_indices, length, transformation):
        if not remaining_indices:
            return np.ones(2 ** length, dtype=int)
        return [
            np.prod([(w[j] + transformation[j] + 1) % 2 for j in remaining_indices])
            for w in product([0, 1], repeat=length)
        ]

    def _generate_basis_vector(self, indices, length):
        if not indices:
            return np.ones(2 ** length, dtype=int)

        idx_array = np.arange(2 ** length)
        binary_representation = (idx_array[:, None] >> np.arange(length)[::-1]) & 1
        product_result = np.prod((binary_representation[:, indices] + 1) % 2, axis=1)
        return product_result


def main():
    # Инициализация параметров Рида-Маллера
    order = 2
    variables = 4
    reed_muller = ReedMullerCode(order, variables)
    reed_muller.display_matrix()

    # Инициализация симулятора ошибок
    simulator = ErrorSimulator(reed_muller.generator_matrix)

    # Определение исходного сообщения
    original_message = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=int)

    # Введение однократной ошибки
    print("\nВведение однократной ошибки:")
    received_once = simulator.introduce_errors(original_message, 1)

    # Инициализация декодера
    decoder = Decoder(
        generator_matrix=reed_muller.generator_matrix,
        matrix_size=variables,
        reference_level=order,
        total_length=reed_muller.generator_matrix.shape[0]
    )

    # Декодирование однократной ошибки
    decoded_once = decoder.decode(received_once)
    if decoded_once is None:
        print("Ошибка не исправлена")
    else:
        corrected_message = (decoded_once @ reed_muller.generator_matrix) % 2
        print(f"v*G: {corrected_message}\nИсправленное сообщение: {decoded_once}")

    # Введение двукратной ошибки
    print("\nВведение двукратной ошибки:")
    received_twice = simulator.introduce_errors(original_message, 2)

    # Декодирование двукратной ошибки
    decoded_twice = decoder.decode(received_twice)
    if decoded_twice is None:
        print("Ошибка не исправлена")
    else:
        corrected_message = (decoded_twice @ reed_muller.generator_matrix) % 2
        print(f"v*G: {corrected_message}\nИсправленное сообщение: {decoded_twice}")


if __name__ == "__main__":
    main()
