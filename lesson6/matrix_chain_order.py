# -*-coding:utf-8-*-#
import numpy as np

matrix_shapes = np.array([[30, 35], [35, 15], [15, 5], [5, 10], [10, 20], [20, 25]])

length = len(matrix_shapes)
m = np.full((length, length), np.inf)
s = np.zeros((length, length), dtype=np.int32)

def maxtrix_chain_order(shapes, start, end):
	if start == end:
		m[start, end] = 0
	else:
		q = 0
		matrix_start = shapes[start]
		matrix_end = shapes[end]
		q = 0
		for i in range(start, end):
			matrix_k = shapes[i]
			q = maxtrix_chain_order(shapes, start, i) + maxtrix_chain_order(shapes, i + 1, end) + matrix_start[0] * matrix_k[1] * matrix_end[1]
			if m[start, end] > q:
				m[start, end] = q
				s[start, end] = i + 1
	return m[start, end]

maxtrix_chain_order(matrix_shapes, 0, length - 1)
print("m: \n", m)
print("s: \n", s)

def print_optimal_parens(s, i, j):
	if i == j:
		print("A_{%d}" % (i + 1), end="")
	else:
		print("(", end="")
		print_optimal_parens(s, i, s[i, j] - 1)
		print_optimal_parens(s, s[i, j], j)
		print(")", end="")

print("optimal seq:")
print_optimal_parens(s, 0, length - 1)
print()
