import numpy as np
from typing import Callable, Tuple


# the function for algorithm 12
def min_quad(Q, b, Delta, epsilon, max_iter=1000):

    def p(lmbda, Q, b):
        n = len(b)
        return -np.linalg.inv(Q + lmbda * np.eye(n)) @ b

    # Check if the unconstrained solution is feasible and optimal
    lmbda_1 = np.min(np.linalg.eigvals(Q))  # smallest eigenvalue of Q
    p_0 = p(0, Q, b)
    if lmbda_1 > 0 and np.linalg.norm(p_0) <= Delta:
        return p_0

    # Initialize l and u for the algorithm
    l = max(0, -lmbda_1)
    u = l + 1

    # Expansion phase to find an upper bound 'u' such that norm(p(u)) > Delta
    iter = 0
    while np.linalg.norm(p(u, Q, b)) > Delta and iter < max_iter:
        l = u
        u = 2 * u
        iter += 1
    if iter >= max_iter:
        raise ValueError(
            "The algorithm did not converge in the expansion phase")

    # Bisection phase to find the optimal lambda within the interval [l, u]
    iter = 0
    while True and iter < max_iter:
        lmbda_prime = 0.5 * (l + u)
        p_prime = p(lmbda_prime, Q, b)
        if np.linalg.norm(p_prime) <= Delta and np.linalg.norm(
                np.linalg.norm(p_prime) - Delta) < epsilon:
            return p_prime  # Found the optimal lambda, return the corresponding p
        if np.linalg.norm(p_prime) > Delta:
            l = lmbda_prime
        else:
            u = lmbda_prime
        iter += 1
    if iter >= max_iter:
        raise ValueError(
            "The algorithm did not converge in the bisection phase")


def generate_pos_Q_b(n: int) -> Tuple[np.ndarray, np.ndarray]:
    Q = np.random.rand(n, n)
    Q = Q.T @ Q
    b = np.random.rand(n)
    return Q, b


def generate_neg_Q_b(n: int) -> Tuple[np.ndarray, np.ndarray]:
    Q = np.random.rand(n, n)
    Q = -Q.T @ Q
    b = np.random.rand(n)
    return Q, b


def check_kkt(x_star, Q, b, Delta, atol):

    def df(x, Q, b):
        return np.dot(Q, x) + b

    def norm_constraint(x, Delta):
        return np.dot(x, x) - Delta**2

    def grad_constraint(x):
        return 2 * x

    def calc_lambda_star(x_star, Q, b, Delta):
        grad_f_at_x_star = df(x_star, Q, b)
        grad_h_at_x_star = grad_constraint(x_star)
        if np.isclose(np.linalg.norm(x_star), Delta):
            lambda_star = -np.dot(grad_f_at_x_star, grad_h_at_x_star) / np.dot(
                grad_h_at_x_star, grad_h_at_x_star)
        else:
            lambda_star = 0
        return lambda_star

    lambda_star = calc_lambda_star(x_star, Q, b, Delta)
    stationarity = np.allclose(df(x_star, Q, b) +
                               lambda_star * grad_constraint(x_star),
                               0,
                               atol=atol)
    primal_feasibility = np.linalg.norm(x_star) <= Delta
    dual_feasibility = lambda_star >= -atol
    complementary_slackness = np.isclose(lambda_star *
                                         norm_constraint(x_star, Delta),
                                         0,
                                         atol=atol)
    if not stationarity:
        print(
            f"f'(x_star) + lambda_star * grad_constraint(x_star) = {df(x_star, Q, b) + lambda_star * grad_constraint(x_star)}"
        )
    if not primal_feasibility:
        print(f"norm(x_star) = {np.linalg.norm(x_star)}")
    if not dual_feasibility:
        print(f"lambda_star = {lambda_star}")
    if not complementary_slackness:
        print(
            f"lambda_star * norm_constraint(x_star, Delta) = {lambda_star * norm_constraint(x_star, Delta)}"
        )
    else:
        return True
