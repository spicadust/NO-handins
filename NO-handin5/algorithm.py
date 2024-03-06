import numpy as np
from typing import Callable, Tuple
from scipy.optimize import minimize
from scipy.linalg import block_diag


def backtracking_line_search(f, df, x, p, alpha):
    c1 = 0.05
    rho = 0.5
    iters = 0
    max_iters = 200
    while f(x + alpha * p) > f(x) + c1 * alpha * np.dot(df(x), p):
        alpha *= rho
        iters += 1
        if iters > max_iters:
            raise ValueError(
                f"Backtracking line search did not converge within {max_iters} iterations"
            )
    return alpha


def constrained_steepest_descent(
    f: Callable,
    df: Callable,
    Hf: Callable,
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = 0.5
    beta = 1.0
    M = np.eye(len(x0)) - A.T @ np.linalg.inv(A @ A.T) @ A
    xs = [x0]
    grad_norms = [np.linalg.norm(df(x0))]
    alphas = [beta]
    x = x0
    k = 0

    # iterate until stopping criterion is met or max_iter is reached
    while grad_norms[-1] > tol and k < max_iter:
        # calculate gradient to find p_k
        p = -M @ df(x)
        # find alpha_k using backtracking line search
        try:
            alpha = backtracking_line_search(f, df, x, p, beta)
            alphas.append(alpha)
        except ValueError as _:
            return np.array(xs), np.array(grad_norms), np.array(alphas)

        # calculate x_k+1 and append to xs
        x = x + alpha * p
        xs.append(x)
        # calculate gradient norm and append to grad_norms
        grad_norms.append(np.linalg.norm(df(x)))
        # update beta_k+1
        beta = alpha / rho
        # update k
        k += 1

    return np.array(xs), np.array(grad_norms), np.array(alphas)


def constrained_newton(
    f: Callable,
    df: Callable,
    hf: Callable,
    A: np.ndarray,
    b: np.ndarray,
    x0: np.ndarray,
    tol: float,
    max_iter: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    xs = [x0]
    grad_norms = [np.linalg.norm(df(x0))]
    x = x0
    k = 0
    alphas = [1]

    # iterate until stopping criterion is met or max_iter is reached
    while grad_norms[-1] > tol and k < max_iter:
        # calculate hessian at x_k
        hf_x = hf(x)

        # Check if hessian is positive definite
        if np.all(np.linalg.eigvals(hf_x) > 0):
            B = hf_x
        else:
            # Compute eigenvectors and eigenvalues
            eigvals, eigvecs = np.linalg.eig(hf_x)
            # Modify the Hessian to be positive definite
            B = sum(
                abs(lam) * np.outer(v, v) for lam, v in zip(eigvals, eigvecs))

        # Solve the KKT system for p_k and lambda*
        KKT_matrix = np.block([[B, A.T],
                               [A, np.zeros((A.shape[0], A.shape[0]))]])
        KKT_rhs = np.block([-df(x), np.zeros(b.shape[0])])

        solution = np.linalg.solve(KKT_matrix, KKT_rhs)
        p_k = solution[:x0.size]
        lambda_star = solution[x0.size:]

        # find alpha_k using backtracking line search
        try:
            alpha_k = backtracking_line_search(f, df, x, p_k, 1.0)
            alphas.append(alpha_k)
        except ValueError as _:
            return np.array(xs), np.array(grad_norms), np.array(alphas)

        # calculate x_k+1 and append to xs
        x = x + alpha_k * p_k
        xs.append(x)

        # calculate gradient norm and append to grad_norms
        grad_norms.append(np.linalg.norm(df(x)))

        # update k
        k = k + 1

    return np.array(xs), np.array(grad_norms), np.array(alphas)


def constrained_scipy(f, df, Hf, A, b, x0, tol, max_iter):

    def constraint(x):
        return A @ x + b

    # Convert the constraint to the form expected by scipy
    con = {'type': 'eq', 'fun': constraint}

    result = minimize(f, x0, jac=df, constraints=con, method='SLSQP')

    # Check if the optimization was successful
    if result.success:
        # The optimal value under the constraint is found
        optimal_x = result.x
        optimal_value = result.fun
        # print('Optimal value:', optimal_value)
        # print('Optimal x:', optimal_x)
        return optimal_x, optimal_value, np.zeros(1)
    else:
        # Optimization failed
        # print('Optimization was not successful. Message:', result.message)
        raise ValueError(f'Optimization was not successful: {result.message}')


def generate_Abx(m, n):
    # Generate a random matrix A of size m*n
    A = np.random.rand(m, n)

    # Use SVD to decompose A, and then reconstruct it with the desired rank of m
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    S = np.diag(S)
    A_rank_m = U @ S @ VT

    # Check the rank of the generated matrix A
    rank_A = np.linalg.matrix_rank(A_rank_m)

    # Generate a random vector b of length m
    b = np.random.rand(m)

    # To find x such that Ax + b = 0, we solve the linear system Ax = -b
    # Since A may not be square, we use the least squares solution
    x, residuals, rank, s = np.linalg.lstsq(A_rank_m, -b, rcond=None)
    return A, b, x
