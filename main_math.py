import math
import numpy as np
import matplotlib.pyplot as plt
import random
import decimal


# CONSTANT
base_matrix = np.array([2, 3])
lst_seeds = []
[lst_seeds.append((random.randrange(60, 110), random.randrange(4, 10))) for _ in range(1, 1000)]


class generateMatrix(object):
    def __init__(self, seed):
        self.seed = seed

    def generate_arguments(self, seed):

        decimal.getcontext().prec = 10000

        A = 2 ** seed[0] + 5
        B = math.ceil(abs(math.log(682*(seed[0]**2) + 32*seed[0]+2)))
        C = math.floor(float(4.5) * (3*seed[0] + 17))
        m = abs(math.floor(11 * seed[0] * math.sin(2*seed[0])))

        set_args = (A, B, C, m)
        return set_args


    def generate_An(self, n):
        An = float(43/9) * (4)**n + float(41/9) - float(1/3) * float(n + 1) - (1/2) * float(n + 1) * float(n + 2)
        return An

    def generate_Fn(self, An):
        # Definitly meilleure facon de faire ca
        A = self.generate_arguments(i)[0]
        B = self.generate_arguments(i)[1]
        C = self.generate_arguments(i)[2]
        m = self.generate_arguments(i)[3]

        Fn = (A * (An ** B) + C) % m
        return Fn

    # VOUDOU MAGIE NOIR TKT
    def divide_chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]


    def vector_generator(self, seed):
        num_elements_to_generate = seed[1] ** 2
        lst_elements_generated = []
        for i in range(num_elements_to_generate):
            lst_elements_generated.append(self.generate_Fn(i))

        sub_lists = list(self.divide_chunks(lst_elements_generated, seed[1]))
        first_matrix = np.array([np.array(xi)
                                for xi in sub_lists])
        det_matrix = int(abs(np.linalg.det(first_matrix)))

        return len(str(det_matrix))

    def final_vector(self, seed):
        final_lst = []
        for i in range(self.vector_generator(seed), self.vector_generator(seed) + 4):
            final_lst.append(self.generate_Fn(self.generate_An(i)))

        sub_lists = list(self.divide_chunks(final_lst, 2))
        final_matrix = np.array([np.array(xi) for xi in sub_lists])

        return np.dot(final_matrix, base_matrix)


lst_matrices = []

for i in lst_seeds:
    # lst_matrices.append(render_matrix(i).final_vector(i))
    lst_matrices.append((generateMatrix(i).final_vector(i)).tolist())


data = np.asarray(lst_matrices)
plt.scatter(data[:, 0], data[:, 1])
plt.show()