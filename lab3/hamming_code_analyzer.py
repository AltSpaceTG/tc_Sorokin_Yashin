import random


class MatrixOperations:
    @staticmethod
    def identity(size):
        return [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    @staticmethod
    def horizontal_concat(matrix_a, matrix_b):
        return [row_a + row_b for row_a, row_b in zip(matrix_a, matrix_b)]

    @staticmethod
    def vertical_concat(matrix_a, matrix_b):
        return matrix_a + matrix_b

    @staticmethod
    def vector_matrix_product(vector, matrix):
        return [sum(v * m for v, m in zip(vector, col)) % 2 for col in zip(*matrix)]


class HammingCode:
    def __init__(self, redundant_bits, is_extended=False):
        self.redundant_bits = redundant_bits
        self.is_extended = is_extended
        self.num_code_bits = 2 ** redundant_bits - 1
        self.num_data_bits = self.num_code_bits - redundant_bits
        self.generator_matrix = self._create_generator_matrix()
        self.parity_check_matrix = self._create_parity_check_matrix()
        if is_extended:
            self._extend_matrices()

    def _create_generator_matrix(self):
        return MatrixOperations.horizontal_concat(
            MatrixOperations.identity(self.num_data_bits), self._generate_basis_vectors()
        )

    def _create_parity_check_matrix(self):
        return MatrixOperations.vertical_concat(
            self._generate_basis_vectors(), MatrixOperations.identity(self.num_code_bits - self.num_data_bits)
        )

    def _generate_basis_vectors(self):
        vectors = []
        current_vector = [0] * (self.num_code_bits - self.num_data_bits)
        for _ in range(self.num_data_bits):
            for i in reversed(range(len(current_vector))):
                if current_vector[i] == 0:
                    current_vector[i] = 1
                    vectors.append(current_vector[:])
                    break
                else:
                    current_vector[i] = 0
        return vectors

    def _extend_matrices(self):
        for row in self.generator_matrix:
            row.append(sum(row) % 2)
        extra_row = [0] * len(self.parity_check_matrix[0])
        self.parity_check_matrix.append(extra_row)
        for row in self.parity_check_matrix:
            row.append(1)

    def display_matrices(self):
        print("\nПорождающая матрица (G):")
        for row in self.generator_matrix:
            print(row)
        print("\nПроверочная матрица (H):")
        for row in self.parity_check_matrix:
            print(row)


class SyndromeDecoder:
    def __init__(self, parity_check_matrix):
        self.syndrome_table = self._create_syndrome_table(parity_check_matrix)

    def _create_syndrome_table(self, parity_check_matrix):
        syndrome_table = {}
        for bit_position in range(len(parity_check_matrix[0])):
            error_vector = [0] * len(parity_check_matrix[0])
            error_vector[bit_position] = 1
            syndrome = MatrixOperations.vector_matrix_product(error_vector, parity_check_matrix)
            syndrome_table[tuple(syndrome)] = error_vector
        return syndrome_table

    def decode(self, received_vector, parity_check_matrix):
        syndrome = MatrixOperations.vector_matrix_product(received_vector, parity_check_matrix)
        if tuple(syndrome) in self.syndrome_table:
            error_vector = self.syndrome_table[tuple(syndrome)]
            corrected_vector = [(bit + err) % 2 for bit, err in zip(received_vector, error_vector)]
            return corrected_vector, syndrome
        return received_vector, syndrome


class HammingAnalyzer:
    def __init__(self, hamming_code, decoder):
        self.hamming_code = hamming_code
        self.decoder = decoder

    def generate_code_word(self):
        u_vectors = [list(col) for col in zip(*self.hamming_code.generator_matrix)]
        return u_vectors[random.randint(0, len(u_vectors) - 1)]

    def introduce_errors(self, code_word, num_errors):
        error_vector = [0] * len(code_word)
        error_positions = random.sample(range(len(code_word)), num_errors)
        for pos in error_positions:
            error_vector[pos] = 1
        return [(bit + err) % 2 for bit, err in zip(code_word, error_vector)], error_vector

    def analyze(self, max_errors):
        code_word = self.generate_code_word()
        print("\n*** Анализ кода Хэмминга ***")
        print("Исходное кодовое слово (без ошибок):", code_word)

        for error_count in range(1, max_errors + 1):
            if error_count > len(code_word):
                break
            print(f"\n--- Тестирование с количеством ошибок: {error_count} ---")

            # Генерация и вывод ошибки
            received_word, error_vector = self.introduce_errors(code_word, error_count)
            print("Внесенные ошибки:", error_vector)
            print("Принятое слово с ошибками:", received_word)

            # Исправление ошибки и вывод синдрома
            corrected_word, syndrome = self.decoder.decode(received_word, self.hamming_code.parity_check_matrix)
            print("Рассчитанный синдром ошибки:", syndrome)

            # Результат исправления
            if syndrome == [0] * len(syndrome):
                print("Ошибок не обнаружено.")
            else:
                print("Исправленное слово:", corrected_word)
                final_syndrome = MatrixOperations.vector_matrix_product(corrected_word,
                                                                        self.hamming_code.parity_check_matrix)
                if final_syndrome == [0] * len(final_syndrome):
                    print("Коррекция успешна, ошибок не осталось (синдром после коррекции: [0,...,0])")
                else:
                    print("Коррекция завершена, но остались ошибки (синдром после коррекции:", final_syndrome, ")")

        print("\n*** Завершение анализа ***\n")


# Примеры использования кода

# Анализ стандартного кода Хэмминга для 1, 2 и 3 ошибок
for r in [2, 3, 4]:
    hamming_code = HammingCode(r)
    decoder = SyndromeDecoder(hamming_code.parity_check_matrix)
    analyzer = HammingAnalyzer(hamming_code, decoder)
    hamming_code.display_matrices()
    analyzer.analyze(max_errors=3)

# Анализ расширенного кода Хэмминга для 1, 2, 3 и 4 ошибок
for r in [2, 3, 4]:
    extended_hamming_code = HammingCode(r, is_extended=True)
    extended_decoder = SyndromeDecoder(extended_hamming_code.parity_check_matrix)
    extended_analyzer = HammingAnalyzer(extended_hamming_code, extended_decoder)
    extended_hamming_code.display_matrices()
    extended_analyzer.analyze(max_errors=4)
