import numpy as np

a = np.array([[2, 4],
              [1, 3],
              [0, 0],
              [0, 0]])

U, s, VT = np.linalg.svd(a, full_matrices=True)

s = np.diag(s)
s = np.concatenate([s, np.zeros([2,2])])

print("U:\n {}".format(U))
print("s:\n {}".format(s))
print("VT:\n {}".format(VT.T))
print(U.dot(s).dot(VT.T))