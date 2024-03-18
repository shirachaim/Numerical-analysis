from colors import bcolors
from matrix_utility import row_addition_elementary_matrix, scalar_multiplication_elementary_matrix
import numpy as np

"""
Function that find the inverse of non-singular matrix
The function performs elementary row operations to transform it into the identity matrix. 
The resulting identity matrix will be the inverse of the input matrix if it is non-singular.
 If the input matrix is singular (i.e., its diagonal elements become zero during row operations), it raises an error.
"""


def inverse(matrix):
    print(bcolors.OKBLUE, f"=================== Finding the inverse of a non-singular matrix using elementary "
                          f"row operations ===================\n {matrix}\n", bcolors.ENDC)

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square.")

    n = matrix.shape[0]
    identity = np.identity(n)

    # Perform row operations to transform the input matrix into the identity matrix
    for i in range(n):
        if matrix[i, i] == 0:
            raise ValueError("Matrix is singular, cannot find its inverse.")

        if matrix[i, i] != 1:
            # Scale the current row to make the diagonal element 1
            scalar = 1.0 / matrix[i, i]
            elementary_matrix = scalar_multiplication_elementary_matrix(n, i, scalar)
            print(f"elementary matrix to make the diagonal element 1: \n {elementary_matrix} \n")
            matrix = np.dot(elementary_matrix, matrix)
            print(f"The matrix after elementary operation: \n {matrix}")
            print(bcolors.OKGREEN,
                  "----------------------------------------------------------------------------------------------------"
                  "--------------",
                  bcolors.ENDC)
            identity = np.dot(elementary_matrix, identity)

        # Zero out the elements above and below the diagonal
        for j in range(n):
            if i != j:
                scalar = -matrix[j, i]
                elementary_matrix = row_addition_elementary_matrix(n, j, i, scalar)
                print(f"elementary matrix for R{j + 1} = R{j + 1} + ({scalar}R{i + 1}): \n {elementary_matrix} \n")
                matrix = np.dot(elementary_matrix, matrix)
                print(f"The matrix after elementary operation: \n {matrix}")
                print(bcolors.OKGREEN,
                      "------------------------------------------------------------------------------------------------"
                      "------------------",
                      bcolors.ENDC)
                identity = np.dot(elementary_matrix, identity)

    return identity

#Date: 18/3/24
#Group: Rina Biniashvili  214398794 , Tamar Nahum 021983812 ,Shira Chaimchaim 215217456
#Name:shira chaim 215217456
if __name__ == '__main__':

    A = np.array([[-1, 1, 3, -3, 1],
                  [3, -3, -4, 2, 3],
                  [2, 1, -5, -3, 5],
                  [-5, -6, 4, 1, 3],
                  [3, -2, -2, -3, 5]])
    B = np.array([3, 8, 2, 14, 6])

    try:
        A_inverse = inverse(A)
        X = np.dot(A_inverse, B)
        print(X)
        print(bcolors.OKBLUE, "\nInverse of matrix A: \n", A_inverse)
        print(
            "=========================================================================================================="
            "===========",
            bcolors.ENDC)

    except ValueError as e:
        print(str(e))

