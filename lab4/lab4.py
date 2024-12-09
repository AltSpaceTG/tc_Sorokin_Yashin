import numpy as np
from itertools import product

class GolayHandler:
    def __init__(self):
        self._B, self._G, self._H = self._build_golay_matrices()

    def _build_golay_matrices(self):
        base_matrix = np.array([
            [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
            [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
            [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        ])
        generator = np.hstack((np.eye(len(base_matrix), dtype=int), base_matrix))
        parity_check = np.vstack((np.eye(len(base_matrix), dtype=int), base_matrix))
        return base_matrix, generator, parity_check

    def inject_noise(self, vector, num_flips):
        distorted = vector.copy()
        flip_positions = np.random.choice(len(vector), num_flips, replace=False)
        for pos in flip_positions:
            distorted[pos] = 1 - distorted[pos]
        return distorted

    def compute_syndrome(self, received):
        return (received @ self._H) % 2

    def attempt_correction(self, received):
        s = self.compute_syndrome(received)
        if np.sum(s) <= 3:
            return np.hstack((s, np.zeros(self._B.shape[0], dtype=int)))
        else:
            for i in range(self._B.shape[0]):
                candidate = (s + self._B[i]) % 2
                if np.sum(candidate) <= 2:
                    err_vec = np.zeros(self._B.shape[0], dtype=int)
                    err_vec[i] = 1
                    return np.hstack((candidate, err_vec))

        s2 = (s @ self._B) % 2
        if np.sum(s2) <= 3:
            return np.hstack((np.zeros(self._B.shape[0], dtype=int), s2))
        else:
            for i in range(self._B.shape[0]):
                candidate = (s2 + self._B[i]) % 2
                if np.sum(candidate) <= 2:
                    err_vec = np.zeros(self._B.shape[0], dtype=int)
                    err_vec[i] = 1
                    return np.hstack((err_vec, candidate))

        return None

    def demonstrate_recovery(self):
        original_msg = np.random.randint(2, size=12)
        coded = (original_msg @ self._G) % 2
        symbol_space = [[0, 1] for _ in range(12)]
        all_msgs = np.array(list(product(*symbol_space)))
        msg_map = {np.array_str((m @ self._G) % 2): m for m in all_msgs}
        print("Изначальное сообщение:", original_msg)
        print("Закодированное слово:", coded)
        for errs in range(1, 5):
            distorted = self.inject_noise(coded, errs)
            print(f"\nОшибок добавлено: {errs}")
            print("Нарушенное слово:", distorted)
            correction = self.attempt_correction(distorted)
            if correction is not None:
                corrected = (distorted + correction) % 2
                key = np.array_str(corrected)
                if key in msg_map:
                    print("Декодировано сообщение:", msg_map[key])
                else:
                    print("Восстановить корректное сообщение не удалось.")
            else:
                print("Коррекция не сработала.")


class RMCodeHandler:
    def __init__(self, order_r, dimension_m):
        self.r = order_r
        self.m = dimension_m
        self._baseH = np.array([[1, 1], [1, -1]])
        self._Hs = self._construct_hadamards()
        self._codebook, self._encoded_book = self._generate_codebook()

    def _construct_hadamards(self):
        matrices = []
        for lvl in range(1, self.m + 1):
            segment = np.kron(np.kron(np.eye(2**(self.m - lvl), dtype=int), self._baseH),
                              np.eye(2**(lvl - 1), dtype=int))
            matrices.append(segment)
        return matrices

    def rm_generator(self, r=None, m=None):
        if r is None:
            r = self.r
        if m is None:
            m = self.m
        if r == 0:
            return np.ones((1, 2**m), dtype=int)
        if r == m:
            upper = self.rm_generator(m - 1, m)
            lower = np.array([0]*(2**m - 1) + [1])
            return np.vstack((upper, lower))
        top = self.rm_generator(r, m - 1)
        bottom = self.rm_generator(r - 1, m - 1)
        return np.vstack((np.hstack((top, top)),
                          np.hstack((np.zeros((bottom.shape[0], top.shape[1]), dtype=int), bottom))))

    def _generate_codebook(self):
        bitspace = [[0, 1] for _ in range(self.m + 1)]
        all_inputs = np.array(list(product(*bitspace)))
        codewords = np.array([(inp @ self.rm_generator()) % 2 for inp in all_inputs])
        return all_inputs, codewords

    def decode_by_min_distance(self, received_word):
        minimal_dist = np.inf
        best_matches = []
        for cw in self._encoded_book:
            dist = np.sum((received_word + cw) % 2)
            if dist < minimal_dist:
                minimal_dist = dist
                best_matches = [cw]
            elif dist == minimal_dist:
                best_matches.append(cw)
        return best_matches

    def refine_decision(self, candidate):
        temp = candidate.copy()
        temp[temp == 0] = -1
        for Hpart in self._Hs:
            temp = temp @ Hpart
        index = np.argmax(np.abs(temp))
        binary_idx = bin(index)[2:].rjust(self.m, '0')
        reversed_code = binary_idx[::-1]
        prefix = '1' if temp[index] > 0 else '0'
        return np.array(list(prefix + reversed_code), dtype=int)

    def illustrate_decoding(self, input_msg, flips=None):
        codeword = (input_msg @ self.rm_generator()) % 2
        if flips:
            for f in flips:
                codeword[f] = 1 - codeword[f]
        print("\n=== Рида-Маллера Демо ===")
        print("Исходный вектор:", input_msg)
        print("Полученный код:", codeword)
        candidates = self.decode_by_min_distance(codeword)
        if len(candidates) == 1:
            refined = self.refine_decision(candidates[0])
            print("Сведённый к однозначному решению результат:", refined)
        else:
            print("Неоднозначность или ошибка.")

if __name__ == "__main__":
    golay = GolayHandler()
    golay.demonstrate_recovery()
    rm = RMCodeHandler(1, 3)
    test_msg = np.array([1, 1, 0, 0])
    distorted = test_msg @ rm.rm_generator() % 2
    distorted[0] = (distorted[0] + 1) % 2
    rm.illustrate_decoding(test_msg, flips=[0])
    distorted = test_msg @ rm.rm_generator() % 2
    distorted[0] = (distorted[0] + 1) % 2
    distorted[1] = (distorted[1] + 1) % 2
    rm.illustrate_decoding(test_msg, flips=[0, 1])
    rm_big = RMCodeHandler(1, 4)
    new_msg = np.array([1, 1, 1, 0, 0])
    distorted = (new_msg @ rm_big.rm_generator()) % 2
    distorted[0] = (distorted[0] + 1) % 2
    rm_big.illustrate_decoding(new_msg, flips=[0])
    distorted = (new_msg @ rm_big.rm_generator()) % 2
    distorted[0] = (distorted[0] + 1) % 2
    distorted[1] = (distorted[1] + 1) % 2
    rm_big.illustrate_decoding(new_msg, flips=[0, 1])
    distorted = (new_msg @ rm_big.rm_generator()) % 2
    distorted[0] = (distorted[0] + 1) % 2
    distorted[1] = (distorted[1] + 1) % 2
    distorted[2] = (distorted[2] + 1) % 2
    rm_big.illustrate_decoding(new_msg, flips=[0, 1, 2])
    distorted = (new_msg @ rm_big.rm_generator()) % 2
    distorted[0] = (distorted[0] + 1) % 2
    distorted[1] = (distorted[1] + 1) % 2
    distorted[2] = (distorted[2] + 1) % 2
    distorted[3] = (distorted[3] + 1) % 2
    rm_big.illustrate_decoding(new_msg, flips=[0, 1, 2, 3])
