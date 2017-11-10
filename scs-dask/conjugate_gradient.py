"""Solve a linear system with conjugate gradient."""

import dask.array as da

def conjugate_gradient_solve(A, b, x_init, tol=1e-8, name=None):
	"""Solve linear equation `Ax = b`, using conjugate gradient."""
	delta = tol * da.linalg.norm(b)
	
	def iterate(x, k, r_norm_sq, r, p):
		Ap = A.dot(p)
		alpha = r_norm_sq / da.dot(p, Ap)
		x = x + alpha * p
		r = r - alpha * p
		r_norm_sq_prev = r_norm_sq
		r_norm_sq = da.dot(r, r)
		beta = r_norm_sq / r_norm_sq_prev
		p = r + beta * p
		return (x, k + 1, r_norm_sq, r, p)
	
	def cond(r_norm_sq):
		return da.sqrt(r_norm_sq) > delta

	r = b - A.dot(x_init)
	x = x_init
	k = 0
	while cond(da.dot(r, r)):
		x, k, r_norm_sq, r, p = iterate(x, k, r_norm_sq, r, p)
	return x
