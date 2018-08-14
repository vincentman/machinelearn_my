import numpy as np
from scipy.sparse import csr_matrix, random
import implicit
from implicit.als import AlternatingLeastSquares

eps = 1.0e-6

# cosine similarity
# eps: avoid dividing by 0
def cosSim(inA, inB):
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return float(inA * inB.T) / (denom + eps)


# Matrix of user to item
# A = np.mat([[5, 5, 3, 0, 5, 5], [5, 0, 4, 0, 4, 4], [0, 3, 0, 5, 4, 5], [5, 4, 3, 3, 5, 5]])
A = np.mat([[5, 5, 3, 0, 5, 5], [5, 0, 4, 0, 4, 4], [0, 3, 0, 5, 4, 5], [5, 4, 3, 3, 5, 5], [5, 5, 0, 0, 0, 5]])
# Vector of new user to item
new = np.mat([[5, 5, 0, 0, 0, 5]])

# csr_matrix is a class
# CSR matrix: Compressed Sparse Row matrix
user_items = csr_matrix(A, dtype=np.float64)

# initialize a model
model = implicit.als.AlternatingLeastSquares(factors=50)

# train the model on a sparse matrix of item/user/confidence weights
model.fit(user_items.T)

item_factors, user_factors = model.item_factors, model.user_factors

print('item_factors.shape = ', item_factors.shape)
print('user_factors.shape = ', user_factors.shape)

recs = model.recommend(4, user_items, N=3)

print(recs)