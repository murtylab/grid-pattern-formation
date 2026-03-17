import numpy as np

def get_2d_sort(x1,x2):
    """
    Reshapes x1 and x2 into square arrays, and then sorts
    them such that x1 increases downward and x2 increases
    rightward. Returns the order.
    """
    n = int(np.round(np.sqrt(len(x1))))
    total_order = x1.argsort()
    total_order = total_order.reshape(n,n)
    for i in range(n):
        row_order = x2[total_order.ravel()].reshape(n,n)[i].argsort()
        total_order[i] = total_order[i,row_order]
    total_order = total_order.ravel()
    return total_order
