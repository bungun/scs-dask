# import numpy as np
# import dask.array as da

# class JacobiPrecond(object):
#     """ Build a Jacobi diagonal preconditioner P = inv(diag(A))
#     """
#     def __init__(self, A_symmetric):
#         self.d = np.reciprocal(np.diag(A_symmetric))
#     def dot(self, v):
#         return self.d * v

# def test_cg_npy(A, b):
#     m, n = shape(A)
#     Asymm_unregularized = A.T.dot(A)
#     Asymm_weakreg = A.T.dot(A) + 0.1 * np.eye(n)
#     Asymm = A.T.dot(A) + np.eye(n)

#     print("A'A")
#     xcg, iters, res = dcg(Asymm_unregularized, b, verbose=2)
#     print("ITERS: {:4}".format(iters))
#     print("||(rho * I + A'A)x - b||_2 / sqrt(n): {}".format(res / n**0.5))

#     print("0.1 *I + A'A")
#     xcg, iters, res = dcg(Asymm_weakreg, b)
#     print("ITERS: {:4}".format(iters))
#     print("||(rho * I + A'A)x - b||_2 / sqrt(n): {}".format(res / n**0.5))

#     print("I + A'A")
#     xcg, iters, res = dcg(Asymm, b, verbose=1)
#     print("ITERS: {:4}".format(iters))
#     print("||(rho * I + A'A)x - b||_2 / sqrt(n): {}".format(res / n**0.5))

# def test_pcg(A, b):
#     m, n = A.shape
#     Asymm = A.T.dot(A) + np.eye(n)

#     print("IDENTITY PRECONDITIONER")
#     xcg, iters, res = dcg(Asymm, b, preconditioner=np.eye(n))
#     print("ITERS: {:4}".format(iters))
#     print("||(rho * I + A'A)x - b||_2 / sqrt(n): {}".format(res / n**0.5))

#     print("JACOBI PRECONDITIONER")
#     xcg, iters, res = dcg(Asymm, b, preconditioner=JacobiPrecond(Asymm))
#     print("ITERS: {:4}".format(iters))
#     print("||(rho * I + A'A)x - b||_2 / sqrt(n): {}".format(res / n**0.5))

# def test_pcg_warmstart(A, b):
#     m, n = A.shape
#     Asymm = A.T.dot(A) + np.eye(n)

#     xcg, iters, res = dcg(Asymm, b, verbose=0)

#     pct_perturb = 5.
#     perturb = np.random.random(n) - 0.5
#     perturb = (perturb / np.linalg.norm(perturb)) * (pct_perturb / 100.) * np.linalg.norm(b)
#     b_new = b + perturb

#     print("PRECONDITIONER + WARM START")
#     xwarm, iters, res = dcg(Asymm, b_new, x0=xcg, preconditioner=JacobiPrecond(Asymm))
#     print("ITERS: {:4}".format(iters))
#     print("||(rho * I + A'A)x - b||_2 / sqrt(n): {}".format(res / n**0.5))

#     print("PRECONDITIONER + COLD START")
#     xcold, iters, res = dcg(Asymm, b_new, preconditioner=JacobiPrecond(Asymm))
#     print("ITERS: {:4}".format(iters))
#     print("||(rho * I + A'A)x - b||_2 / sqrt(n): {}".format(res / n**0.5))

#     print("WARM START, no preconditioner")
#     xwarm_noprecon, iters, res = dcg(Asymm, b_new, x0=xcg)
#     print("ITERS: {:4}".format(iters))
#     print("||(rho * I + A'A)x - b||_2 / sqrt(n): {}".format(res / n**0.5))

#     print("COLD START, no preconditioner")
#     xcold_noprecon, iters, res = dcg(Asymm, b_new)
#     print("ITERS: {:4}".format(iters))
#     print("||(rho * I + A'A)x - b||_2 / sqrt(n): {}".format(res / n**0.5))

# def test_cg_dask(A, b, chunk_by=1):
#     m, n = A.shape
#     Asymm = A.T.dot(A) + np.eye(n)
#     Ad = da.from_array(Asymm, chunks=n/int(chunk_by))

#     print("NUMPY: I + A'A")
#     xcg, iters, res = dcg(Asymm, b, verbose=1)
#     print("ITERS: {:4}".format(iters))
#     print("||(rho * I + A'A)x - b||_2 / sqrt(n): {}".format(res / n**0.5))

#     print("DASK: I + A'A")
#     xcg, iters, res = dcg(Ad, b, verbose=1)
#     print("ITERS: {:4}".format(iters))
#     print("||(rho * I + A'A)x - b||_2 / sqrt(n): {}".format(res / n**0.5))

# def test_cgd_large(m, n, chunk_by=2):
#     c = int(chunk_by)
#     mlg = 40000
#     nlg = 30000
#     Alg = da.random.random((mlg, nlg), chunks=(mlg/c, nlg/c))
#     Aslg = Alg.T.dot(Alg) + da.eye(nlg, chunks=nlg/c)
#     blg = da.random.random(nlg, chunks=nlg/c)

#     print("DASK: I + A'A ({})".format(Aslg.shape))
#     xcg, iters, res = dcg(Ad, b, verbose=1)
#     print("ITERS: {:4}".format(iters))
#     print("||(rho * I + A'A)x - b||_2 / sqrt(n): {}".format(res / n**0.5))

# if __name__ == "__main__":
#     m = 400
#     n = 300
#     A = np.random.random((m, n))
#     b = np.random.random(n)
#     test_cg_npy(A, b)
#     test_pcg_npy(A, b)
#     test_pcg_warmstart(A, b)
#     test_cg_dask(A, b)
#     test_cgd_large(m * 100, n * 100, 2)

