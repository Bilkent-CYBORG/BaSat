"""Private additive-GP numerical routines, derived from Dragonfly 0.1.4.

Copyright (c) 2018 Kirthevasan Kandasamy. MIT-licensed source adaptations;
see ``_additive_numerics.LICENSE.txt`` for the complete upstream license.
Original authors include Kirthevasan Kandasamy and Karun Raju Vysyaraju.
The slice/Metropolis source identifies its adaptation from PyMC3; the PyMC
developers' Apache 2.0 license and copyright notices are retained alongside
this module in ``_additive_numerics.PYMC_LICENSE.txt``.

Basat modifications: selected numerical definitions only (no external runner),
context-local RNG and caches, explicit NumPy-2 scalar conversions, source-
equivalent PDOO fallback and Basat-owned jitter warnings. Original operation
order is preserved and tested against the pinned source.

This module contains no external optimizer, experiment runner or benchmark.
NumPy/SciPy implement numerical primitives. Every stochastic operation uses
the calling Basat algorithm's local RandomState, never NumPy's global RNG.
Only the continuous Matérn additive policy is supported by the owning helper.
Source file hashes below identify the exact definitions adapted here.
"""

from __future__ import annotations

# The retained upstream numerical definitions predate type annotations.
# Their typed Basat policy boundary is checked; numerical parity is executable.
# mypy: check-untyped-defs=False
# Source arithmetic and control order are deliberately retained for parity.
# These exemptions concern historical style/dead optional branches, not names.
# ruff: noqa: B007, B023, B026, B904, B905, E731, E741, F841, RUF005, SIM101, SIM108, SIM114, UP004, UP008, UP031, UP034
import math
import queue as Qu
import sys
import warnings
from argparse import Namespace
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import total_ordering
from itertools import product as itertools_product
from typing import Any

import numpy as _numpy
from scipy.linalg import solve_triangular

from .Exception.exception_ADD_GP_UCB import ADD_GP_UCBWarning

_active_rng: ContextVar[_numpy.random.RandomState] = ContextVar("basat_additive_rng")
_active_cache: ContextVar[Any] = ContextVar("basat_additive_kernel_cache", default=None)


class _RandomFacade:
    """Dispatch draws to a context-local, checkpointable MT19937 stream."""

    RandomState = _numpy.random.RandomState

    def __getattr__(self, name):
        return getattr(_active_rng.get(), name)


class _NumpyFacade:
    """Keep source NumPy expressions without changing process-global state."""

    random = _RandomFacade()
    math = math
    int = int
    float = float

    @staticmethod
    def asscalar(value):
        return _numpy.asarray(value).item()

    def __getattr__(self, name):
        return getattr(_numpy, name)


np = _NumpyFacade()
nr = np.random


@contextmanager
def local_random_state(rng, cache=None):
    """Use one learner-owned RNG without interfering with another learner."""
    token = _active_rng.set(rng)
    cache_token = _active_cache.set(cache)
    try:
        yield
    finally:
        _active_rng.reset(token)
        _active_cache.reset(cache_token)


class _SilentReporter:
    def write(self, message):
        pass

    def writeln(self, message):
        pass


def get_reporter(reporter=None):
    return reporter if isinstance(reporter, _SilentReporter) else _SilentReporter()


def get_option_specs(name, required=False, default=None, help_str="", **kwargs):
    return {"name": name, "default": default}


def load_options(options, descr=None, reporter=None, cmd_line=False, partial_options=None):
    """Resolve internal constants only; no command-line/framework dependency."""
    if cmd_line:
        raise ValueError("The internal additive fitter does not parse command lines.")
    values = {option["name"]: option["default"] for option in options}
    if partial_options is not None:
        values.update(
            partial_options if isinstance(partial_options, dict) else vars(partial_options)
        )
    return Namespace(**values)


def _unsupported(*args, **kwargs):
    raise NotImplementedError("Only the continuous Matérn additive policy is supported.")


class _UnsupportedKernel:
    def __init__(self, *args, **kwargs):
        _unsupported()


def direct_ft_maximise(objective, bounds, max_evals):
    """Historical pure-Python DIRECT fallback, used only in audit dimensions.

    E2 has 100 dimensions and chooses PDOO explicitly. The source's absent
    optional Fortran extension falls back to this same PDOO implementation.
    The native policy never loads a separate optimizer package or extension.
    """
    return pdoo_maximise(objective, bounds, max_evals)


LOOP_ERR_MSG = "max slicer iters %d exceeded"


# Source: utils/general_utils.py SHA256 8e67c24c01bafdb8f3c0ea2ea9812ac0b67984396b016a28d6508a133b57fa27


def map_to_cube(pts, bounds):
    """Maps bounds to [0,1]^d and returns the representation in the cube."""
    return (pts - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def map_to_bounds(pts, bounds):
    """Given a point in [0,1]^d, returns the representation in the original space."""
    return pts * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]


def get_sublist_from_indices(orig_list, idxs):
    """Returns a sublist from the indices. orig_list can be anthing that can be
    indexed and idxs are a list of indices."""
    return [orig_list[idx] for idx in idxs]


def get_idxs_from_list_of_lists(list_of_lists, idx):
    """Returns a list of objects corresponding to index idx from a list of lists."""
    return [elem[idx] for elem in list_of_lists]


def dist_squared(X1, X2):
    """If X1 is n1xd and X2 is n2xd, this returns an n1xn2 matrix where the (i,j)th
    entry is the squared distance between X1(i,:) and X2(j,:).
    """
    n1, dim1 = X1.shape
    n2, dim2 = X2.shape
    if dim1 != dim2:
        raise ValueError("Second dimension of X1 and X2 should be equal.")
    dist_sq = (
        np.outer(np.ones(n1), (X2**2).sum(axis=1))
        + np.outer((X1**2).sum(axis=1), np.ones(n2))
        - 2 * X1.dot(X2.T)
    )
    dist_sq = np.clip(dist_sq, 0.0, np.inf)
    return dist_sq


def project_symmetric_to_psd_cone(M, is_symmetric=True, epsilon=0):
    """Projects the symmetric matrix M to the PSD cone."""
    if is_symmetric:
        try:
            eigvals, eigvecs = np.linalg.eigh(M)
        except np.linalg.LinAlgError:
            print("LinAlgError encountered with eigh. Defaulting to eig.")
            eigvals, eigvecs = np.linalg.eig(M)
            eigvals = np.real(eigvals)
            eigvecs = np.real(eigvecs)
    else:
        eigvals, eigvecs = np.linalg.eig(M)
    clipped_eigvals = np.clip(eigvals, epsilon, np.inf)
    return (eigvecs * clipped_eigvals).dot(eigvecs.T)


def stable_cholesky(M, add_to_diag_till_psd=True):
    """Returns L, a 'stable' cholesky decomposition of M. L is lower triangular and
    satisfies L*L' = M.
    Sometimes nominally psd matrices are not psd due to numerical issues. By adding a
    small value to the diagonal we can make it psd. This is what this function does.
    Use this iff you know that K should be psd. We do not check for errors.
    """
    if M.size == 0:
        return M  # if you pass an empty array then just return it.
    try:
        # First try taking the Cholesky decomposition.
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError as e:
        # If it doesn't work, then try adding diagonal noise.
        if not add_to_diag_till_psd:
            raise e
        diag_noise_power = -11
        max_M = np.diag(M).max()
        diag_noise = np.diag(M).max() * 1e-11
        chol_decomp_succ = False
        while not chol_decomp_succ:
            try:
                diag_noise = (10**diag_noise_power) * max_M
                warnings.warn(
                    f"Basat ADD_GP_UCB added jitter={diag_noise:.12g} to the "
                    f"{M.shape[0]}x{M.shape[1]} GP covariance diagonal during "
                    "Cholesky factorization after the unmodified matrix failed.",
                    ADD_GP_UCBWarning,
                    stacklevel=2,
                )
                L = np.linalg.cholesky(M + diag_noise * np.eye(M.shape[0]))
                chol_decomp_succ = True
            except np.linalg.LinAlgError:
                diag_noise_power += 1
            if diag_noise_power >= 5:
                raise ValueError(
                    (
                        "Could not compute Cholesky decomposition despite adding"
                        + " %0.4f to the diagonal. This is likely because the M is not "
                        + "positive semi-definite or has infinities/nans."
                    )
                    % (diag_noise)
                )
    return L


def _solve_triangular_common(A, b, lower):
    """Solves Ax=b when A is a triangular matrix."""
    if A.size == 0 and b.shape[0] == 0:
        return np.zeros((b.shape))
    else:
        return solve_triangular(A, b, lower=lower)


def solve_lower_triangular(A, b):
    """Solves Ax=b when A is lower triangular."""
    return _solve_triangular_common(A, b, lower=True)


def solve_upper_triangular(A, b):
    """Solves Ax=b when A is upper triangular."""
    return _solve_triangular_common(A, b, lower=False)


def draw_gaussian_samples(num_samples, mu, K):
    """Draws num_samples samples from a Gaussian distribution with mean mu and
    covariance K.
    """
    num_pts = len(mu)
    L = stable_cholesky(K)
    U = np.random.normal(size=(num_pts, num_samples))
    V = L.dot(U).T + mu
    return V


# Source: utils/ancillary_utils.py SHA256 945b14863f3c1c4075faf06023f8e3112a0ea9d7dfd463db94a58a2e5d29c9c2


def get_list_of_floats_as_str(float_list, round_to_decimals=3):
    """Rounds the list and returns a string representation."""
    return str(list(np.round(float_list, round_to_decimals)))


# Source: utils/doo.py SHA256 448c5bf689b63aa88c10d98a65aaafba3426fe8fb551689a82600b35727e5541


class DOOFunction(object):
    """This just creates a wrapper to call the function by appropriately creating bounds
    and querying appropriately."""

    def __init__(self, func, domain_bounds, vectorised=False):
        """Constructor.
        func: takes an argument x which is a point in the domain.
        domain_bounds: are the bounds of the domain space
        vectorised: If True it means func can take multiple inputs and produce multiple
                    outputs. If False, the functions can take only single inputs in
                    'column' form.
        """
        self.func = func
        self.domain_bounds = np.array(domain_bounds)
        self.domain_dim = len(domain_bounds)
        self.vectorised = vectorised

    def get_unnormalised_coords(self, X):
        """Maps points in the cube to the original space."""
        ret_X = None if X is None else map_to_bounds(X, self.domain_bounds)
        return ret_X

    def eval_single_point(self, X):
        """Evaluates X at the given Z at a single point."""
        if not self.vectorised:
            return float(np.asarray(self.func(X)).item())
        else:
            X = np.array(X).reshape((1, self.domain_dim))
            return float(np.asarray(self.func(X)).item())

    def eval_single_point_normalised(self, X):
        """Evaluates X at the given Z at a single point using normalised coordinates."""
        X = self.get_unnormalised_coords(X)
        return self.eval_single_point(X)

    def eval_cost_single_point_normalised(self, Z):
        """Evaluates the cost function at a single point using normalised coordinates."""
        # pylint: disable=unused-argument
        return 1.0


@total_ordering
class Node(object):
    """Base object for the node of a tree."""

    def __init__(self, cell, value, fidel, upp_bound, height, dimension):
        """cell: tuple"""
        self.cell = cell
        self.value = value
        self.fidelity = fidel
        self.upp_bound = upp_bound
        self.height = height
        self.dimension = dimension

    def __cmp__(self, other):
        return (other.upp_bound > self.upp_bound) - (other.upp_bound < self.upp_bound)

    def __lt__(self, other):
        return other.upp_bound < self.upp_bound

    def __eq__(self, other):
        return other.upp_bound == self.upp_bound


class OptTree(object):
    """Class for Tree search based optimisation."""

    def __init__(
        self, doo_obj, nu_max, rho_max, total_budget, K, C_init, tol=1e-3, Randomize=False
    ):
        self.doo_obj = doo_obj
        self.nu_max = nu_max
        self.rho_max = rho_max
        self.total_budget = total_budget
        self.K = K
        self.C = C_init
        self.tol = tol
        self.value_dict = {}
        self.qcount = 0
        self.last_z_hist = []
        self.Randomize = Randomize
        self.query_vals = []
        self.query_pts = []

    def get_value(self, cell, fidel):
        """cell: tuple"""
        # pylint: disable=unused-argument
        self.qcount = self.qcount + 1
        x = np.array([(s[0] + s[1]) / 2.0 for s in list(cell)])
        y = self.doo_obj.eval_single_point_normalised(x)
        if len(self.query_vals) <= self.total_budget:
            self.query_pts.append(x)
            self.query_vals.append(y)
        return y

    def get_queried_pts(self):
        """Returns queried points and corresponding values."""
        return self.query_pts, self.query_vals

    def querie(self, cell, height, rho, nu, dimension, option=1):
        """Query."""
        diam = nu * (rho**height)
        if option == 1:
            z = min(max(1 - diam / self.C, self.tol), 1.0)
        else:
            z = 1.0
        if cell in self.value_dict:
            current = self.value_dict[cell]
            if abs(current.fidelity - z) <= self.tol:
                value = current.value
                cost = 0
            else:
                value = self.get_value(cell, z)
                self.last_z_hist = self.last_z_hist + [z]
                if abs(value - current.value) > self.C * abs(current.fidelity - z):
                    self.C = 2.0 * self.C
                current.value = value
                current.fidelity = z
                self.value_dict[cell] = current
                cost = self.doo_obj.eval_cost_single_point_normalised([z])
        else:
            value = self.get_value(cell, z)
            self.last_z_hist = self.last_z_hist + [z]
            bhi = diam + self.C * (1.0 - z) + value
            self.value_dict[cell] = Node(cell, value, z, bhi, height, dimension)
            cost = self.doo_obj.eval_cost_single_point_normalised([z])

        bhi = diam + self.C * (1.0 - z) + value
        current_object = Node(cell, value, z, bhi, height, dimension)
        return current_object, cost

    def split_children(self, current, rho, nu, option=1):
        """Split children."""
        pcell = list(current.cell)
        span = [abs(pcell[i][1] - pcell[i][0]) for i in range(len(pcell))]
        if self.Randomize:
            dimension = np.random.choice(range(len(pcell)))
        else:
            dimension = np.argmax(span)
        dd = len(pcell)
        if dimension == current.dimension:
            dimension = (current.dimension - 1) % dd
        cost = 0
        h = current.height + 1
        l = np.linspace(pcell[dimension][0], pcell[dimension][1], self.K + 1)
        children = []
        for i in range(len(l) - 1):
            cell = []
            for j, _ in enumerate(pcell):
                if j != dimension:
                    cell = cell + [pcell[j]]
                else:
                    cell = cell + [(l[i], l[i + 1])]
            child, c = self.querie(tuple(cell), h, rho, nu, dimension, option)
            children = children + [child]
            cost = cost + c

        return children, cost

    def run_DOO(self, budget, nu, rho):
        """Runs DOO optimisation."""
        leaf_Q = Qu.PriorityQueue()
        d = self.doo_obj.domain_dim
        cell = tuple([tuple([0, 1]) for _ in range(d)])
        height = 0
        cost = 0
        current, c = self.querie(cell, height, rho, nu, 0)
        cost = cost + c
        leaf_Q.put(current)
        dict_of_points = {}
        while cost <= budget:
            current = leaf_Q.get()
            dict_of_points[current.cell] = {
                "val": current.value,
                "fidel": current.fidelity,
                "height": current.height,
            }
            children, curr_cost = self.split_children(current, rho, nu)
            if current.cell == children[0].cell:
                break
            cost = cost + curr_cost
            for child in children:
                leaf_Q.put(child)
        while not leaf_Q.empty():
            c = leaf_Q.get()
            dict_of_points[c.cell] = {"val": c.value, "fidel": c.fidelity, "height": c.height}

        # maxi = float(-sys.maxint - 1)
        maxi = float("-inf")
        point = 0
        maxh = 0
        val = 0
        fidel = 0
        for key in dict_of_points:
            c = dict_of_points[key]
            if c["val"] - self.C * (1.0 - c["fidel"]) > maxi:  # - nu*(rho**c.height) > maxi:
                maxi = c["val"] - self.C * (1.0 - c["fidel"])  # - nu*(rho**c.height)
                val = c["val"]
                fidel = c["fidel"]
                point = np.array([(s[0] + s[1]) / 2 for s in key])
                maxh = c["height"]

        return val, fidel, point, cost, maxh

    def run_PDOO(self, mult=0.5):
        """Runs DOO optimisation iteratively based on the budget."""
        Dm = int(np.log(self.K) / np.log(1 / self.rho_max))
        n = self.total_budget / self.doo_obj.eval_cost_single_point_normalised(np.array([1.0]))
        N = int(mult * Dm * np.log(n / np.log(n)))
        budget = self.total_budget / float(N)
        nu = self.nu_max
        total_cost = 0.0
        results = []
        for i in range(N):
            rho = (self.rho_max) ** (float(N) / (N - i))
            est, fidel, point, cost, h = self.run_DOO(budget, nu, rho)
            results = results + [(est, fidel, point, cost, h)]
            total_cost = total_cost + cost
        temp = [s[0] - self.C * (1 - s[1]) for s in results]
        index = np.argmax(temp)
        self.last_results = (temp[index], results[index][2], total_cost)

        return results, index


def pdoo_wrap(
    doo_obj,
    total_budget,
    nu_max=1.0,
    rho_max=0.9,
    K=2,
    C_init=0.8,
    tol=1e-3,
    POO_mult=0.5,
    Randomize=False,
    return_history=False,
):
    """Wrapper for running PDOO optimisation."""
    # pylint: disable=too-many-locals
    total_budget = total_budget * doo_obj.eval_cost_single_point_normalised([1.0])
    opt_tree = OptTree(doo_obj, nu_max, rho_max, total_budget, K, C_init, tol, Randomize)
    results, index = opt_tree.run_PDOO(POO_mult)
    max_pt = doo_obj.get_unnormalised_coords(results[index][2])
    # IF not return history
    if not return_history:
        return results[index][0], max_pt, None
    history = Namespace()
    max_iter = int(total_budget)
    query_pts, query_vals = opt_tree.get_queried_pts()
    max_val = max(query_vals)
    history.query_step_idxs = [i for i in range(max_iter)]
    history.query_send_times = list(range(0, max_iter))
    history.query_receive_times = list(range(1, max_iter + 1))
    history.query_points = [doo_obj.get_unnormalised_coords(x) for x in query_pts]
    history.query_vals = query_vals
    history.query_true_vals = query_vals
    history.curr_opt_vals = []
    history.curr_opt_points = []
    curr_max = -np.inf
    for idx, qv in enumerate(history.query_vals):
        if qv >= curr_max:
            curr_max = qv
            curr_opt_point = history.query_points[idx]
        history.curr_opt_vals.append(curr_max)
        history.curr_opt_points.append(curr_opt_point)
    history.query_eval_times = [1 for _ in range(max_iter)]
    history.curr_true_opt_vals = history.curr_opt_vals
    return max_val, max_pt, history


# Source: utils/oper_utils.py SHA256 aab8aa7a0ce88f9657c0a5b5f8f103258bc6b04ed7e85601bb1a49200cff2869


def random_sample(obj, bounds, max_evals, vectorised=True):
    """Optimises a function by randomly sampling and choosing its maximum."""
    dim = len(bounds)
    rand_pts = map_to_bounds(np.random.random((int(max_evals), dim)), bounds)
    if vectorised:
        obj_vals = obj(rand_pts)
    else:
        obj_vals = np.array([obj(x) for x in rand_pts])
    return rand_pts, obj_vals


def random_maximise(obj, bounds, max_evals, return_history=False, vectorised=True):
    """Optimises a function by randomly sampling and choosing its maximum."""
    rand_pts, obj_vals = random_sample(obj, bounds, max_evals, vectorised)
    max_idx = obj_vals.argmax()
    max_val = obj_vals[max_idx]
    max_pt = rand_pts[max_idx]
    if return_history:
        history = Namespace(query_vals=obj_vals, query_points=rand_pts)
    else:
        history = None
    return max_val, max_pt, history


def pdoo_maximise(obj, bounds, max_evals):
    """
    A wrapper for maximising a function which calls doo_wrap. See arguments
    under utils.doo_wrap for more details.
    """
    K = 2
    tol = 1e-3
    nu_max = 1.0
    C_init = 0.8
    rho_max = 0.9
    POO_mult = 0.5
    doo_obj = DOOFunction(obj, bounds)
    max_val, max_pt, history = pdoo_wrap(
        doo_obj, max_evals, nu_max, rho_max, K, C_init, tol, POO_mult
    )
    return max_val, max_pt, history


def latin_hc_indices(dim, num_samples):
    """Obtains indices for Latin Hyper-cube sampling."""
    index_set = [list(range(num_samples))] * dim
    lhs_indices = []
    for i in range(num_samples):
        curr_idx_idx = np.random.randint(num_samples - i, size=dim)
        curr_idx = [index_set[j][curr_idx_idx[j]] for j in range(dim)]
        index_set = [
            index_set[j][: curr_idx_idx[j]] + index_set[j][curr_idx_idx[j] + 1 :]
            for j in range(dim)
        ]
        lhs_indices.append(curr_idx)
    return lhs_indices


def latin_hc_sampling(dim, num_samples):
    """Latin Hyper-cube sampling in the unit hyper-cube."""
    if num_samples == 0:
        return np.zeros((0, dim))
    elif num_samples == 1:
        return 0.5 * np.ones((1, dim))
    lhs_lower_boundaries = (np.linspace(0, 1, num_samples + 1)[:num_samples]).reshape(1, -1)
    width = lhs_lower_boundaries[0][1] - lhs_lower_boundaries[0][0]
    lhs_lower_boundaries = np.repeat(lhs_lower_boundaries, dim, axis=0).T
    lhs_indices = latin_hc_indices(dim, num_samples)
    lhs_sample_boundaries = []
    for i in range(num_samples):
        curr_idx = lhs_indices[i]
        curr_sample_boundaries = [lhs_lower_boundaries[curr_idx[j]][j] for j in range(dim)]
        lhs_sample_boundaries.append(curr_sample_boundaries)
    lhs_sample_boundaries = np.array(lhs_sample_boundaries)
    uni_random_width = width * np.random.random((num_samples, dim))
    lhs_samples = lhs_sample_boundaries + uni_random_width
    return lhs_samples


def random_sample_from_discrete_domain(dscr_vals, num_points=None):
    """Samples from a discrete domain."""

    def _sample_single_point():
        """Samples a single point."""
        return [np.random.choice(categ) for categ in dscr_vals]

    # Now draw num_points of them
    num_points_to_sample = 1 if num_points is None else num_points
    if len(dscr_vals) == 0:
        ret = [[]] * num_points_to_sample
    else:
        ret = [_sample_single_point() for _ in range(num_points_to_sample)]
    if num_points is None:
        return ret[0]
    else:
        return ret


def random_sample_from_prod_discrete_domain(list_of_list_of_vals, num_samples):
    """Sample from a product of discrete domains."""
    return random_sample_from_discrete_domain(list_of_list_of_vals, num_samples)


def random_sample_cts_dscr(obj, cts_bounds, dscr_vals, max_evals, vectorised=True):
    """Sample from a joint continuous and discrete space."""
    dim = len(cts_bounds)
    cts_rand_pts = map_to_bounds(np.random.random((int(max_evals), dim)), cts_bounds)
    dscr_rand_pts = random_sample_from_discrete_domain(dscr_vals, max_evals)
    if vectorised:
        obj_vals = obj(cts_rand_pts, dscr_rand_pts)
    else:
        obj_vals = np.array([obj(cx, dx) for (cx, dx) in zip(cts_rand_pts, dscr_rand_pts)])
    return cts_rand_pts, dscr_rand_pts, obj_vals


# Source: sampling/slice.py SHA256 c19639defd0fc22e66b98be281edc47ccfe8478ae2cd7ca572b6189b08eacdbb


class Slice(object):
    """
    Class for Slice sampler.

    Parameters
    ----------
    model : Distribution from which sampling has done
    w     : Initial width for slice
    tune  : Flag for tuning
    """

    def __init__(self, model, w=1.0, tune=True, iter_limit=np.inf):
        """
        Constructor for slice sampler.
        """
        self.model = model
        self.w = w
        self.tune = tune
        self.n_tunes = 0.0
        self.iter_limit = iter_limit

    def _sample(self, q0):
        """
        Helper function which implements slice sampler.
        """
        self.w = np.resize(self.w, len(q0))  # this is a repmat
        q = np.copy(q0)
        ql = np.copy(q0)  # l for left boundary
        qr = np.copy(q0)  # r for right boudary
        for i, _ in enumerate(q0):
            # uniformly sample from 0 to p(q), but in log space
            y = self.model.logp(q) - nr.standard_exponential()
            ql[i] = q[i] - nr.uniform(0, self.w[i])
            qr[i] = q[i] + self.w[i]

            # Stepping out procedure
            cnt = 0
            while y < self.model.logp(ql):
                ql[i] -= self.w[i]
                cnt += 1
                if cnt > self.iter_limit:
                    raise RuntimeError(LOOP_ERR_MSG % self.iter_limit)
            cnt = 0
            while y < self.model.logp(qr):
                qr[i] += self.w[i]
                cnt += 1
                if cnt > self.iter_limit:
                    raise RuntimeError(LOOP_ERR_MSG % self.iter_limit)

            cnt = 0
            q[i] = (qr[i] - ql[i]) * nr.rand() + ql[i]
            while self.model.logp(q) < y:
                # Sample uniformly from slice
                if q[i] > q0[i]:
                    qr[i] = q[i]
                elif q[i] < q0[i]:
                    ql[i] = q[i]
                q[i] = (qr[i] - ql[i]) * nr.rand() + ql[i]
                cnt += 1
                if cnt > self.iter_limit:
                    raise RuntimeError(LOOP_ERR_MSG % self.iter_limit)

            if self.tune:
                self.w[i] = self.w[i] * (self.n_tunes / (self.n_tunes + 1)) + (qr[i] - ql[i]) / (
                    self.n_tunes + 1
                )  # same as before
                qr[i] = q[i]
                ql[i] = q[i]

        if self.tune:
            self.n_tunes += 1

        return q

    def sample(self, q0, num_samples=1, burn=100):
        """
        sample -- function which populates samples
        by calling function _sample iteratively
        """
        if num_samples is None:
            num_samples = 1
        if not hasattr(q0, "__len__"):
            q0 = np.array([q0])

        # Burn in samples
        for i in range(burn):
            q0 = self._sample(q0)

        samples = np.zeros([num_samples, len(q0)])
        for i in range(num_samples):
            samples[i] = self._sample(q0)
            q0 = samples[i]

        return samples


# Source: sampling/metropolis.py SHA256 be524e9c0214a6abee4cfed6273e5da6474d546f28c5a0b4b076fd99d914dded


class Proposal(object):
    """Base class for Proposal Distribution."""

    def __init__(self, s):
        self.s = s


class NormalProposal(Proposal):
    """Normal Proposal Distribution."""

    def __call__(self):
        return nr.normal(scale=self.s)


class Metropolis(object):
    """
    Metropolis-Hastings sampling step

    Parameters
    ----------
    S : standard deviation or covariance matrix
      Some measure of variance to parameterize proposal distribution
    proposal_dist : function
      Function that returns zero-mean deviates when parameterized with
      S (and n). Defaults to normal.
    scaling : scalar or array
      Initial scale factor for proposal. Defaults to 1.
    tune : bool
      Flag for tuning. Defaults to True.
    tune_interval : int
      The frequency of tuning. Defaults to 100 iterations.
    """

    def __init__(
        self,
        model,
        discrete=False,
        S=None,
        proposal_dist=None,
        scaling=1.0,
        tune=True,
        tune_interval=100,
    ):

        if S is None:
            S = np.ones(model.get_dim())

        if proposal_dist is not None:
            self.proposal_dist = proposal_dist(S)
        elif S.ndim == 1:
            self.proposal_dist = NormalProposal(S)
        elif S.ndim == 2:
            self.proposal_dist = _unsupported()
        else:
            raise ValueError("Invalid rank for variance: %s" % S.ndim)

        self.model = model
        self.scaling = np.atleast_1d(scaling).astype("d")
        self.tune = tune
        self.tune_interval = tune_interval
        self.steps_until_tune = tune_interval
        self.accepted = 0
        self.discrete = discrete

        super(Metropolis, self).__init__()

    def _sample(self, q0):
        """
        Helper function which implements metropolis sampler.
        """
        if not self.steps_until_tune and self.tune:
            # Tune scaling parameter
            self.scaling = tune_params(self.scaling, self.accepted / float(self.tune_interval))
            # Reset counter
            self.steps_until_tune = self.tune_interval
            self.accepted = 0

        delta = self.proposal_dist() * self.scaling
        if self.discrete:
            delta = np.round(delta, 0).astype("int64")
            q0 = q0.astype("int64")
            q = (q0 + delta).astype("int64")
        else:
            q = q0 + delta

        accept = delta_logp(self.model, q, q0)
        q_new, accepted = metrop_select(accept, q, q0)
        self.accepted += accepted

        self.steps_until_tune -= 1

        return q_new

    def sample(self, q0, num_samples=1):
        """
        sample -- function which populates samples
        by calling function _sample iteratively
        """
        if num_samples is None:
            num_samples = 1
        if not hasattr(q0, "__len__"):
            q0 = np.array([q0])

        samples = np.zeros([num_samples, len(q0)])
        for i in range(num_samples):
            samples[i] = self._sample(q0)
            q0 = samples[i]

        return samples


def metrop_select(mr, q, q0):
    """Perform rejection/acceptance step for Metropolis class samplers.

    Returns the new sample q if a uniform random number is less than the
    metropolis acceptance rate (`mr`), and the old sample otherwise, along
    with a boolean indicating whether the sample was accepted.

    Parameters
    ----------
    mr : float, Metropolis acceptance rate
    q : proposed sample
    q0 : current sample

    Returns
    -------
    q or q0
    """
    # Compare acceptance ratio to uniform random number
    if np.isfinite(mr) and nr.uniform() < mr:
        return q, True
    else:
        return q0, False


def tune_params(scale, acc_rate):
    """
    Tunes the scaling parameter for the proposal distribution
    according to the acceptance rate over the last tune_interval:

    Rate  Variance adaptation
    ----  -------------------
    <0.001    x 0.1
    <0.05     x 0.5
    <0.2      x 0.9
    >0.5      x 1.1
    >0.75     x 2
    >0.95     x 10

    """

    # Switch statement
    if acc_rate < 0.001:
        # reduce by 90 percent
        scale *= 0.1
    elif acc_rate < 0.05:
        # reduce by 50 percent
        scale *= 0.5
    elif acc_rate < 0.2:
        # reduce by ten percent
        scale *= 0.9
    elif acc_rate > 0.95:
        # increase by factor of ten
        scale *= 10.0
    elif acc_rate > 0.75:
        # increase by double
        scale *= 2.0
    elif acc_rate > 0.5:
        # increase by ten percent
        scale *= 1.1

    return scale


def delta_logp(model, q, q0):
    """Function for calculating acceptance criteria"""
    val = model.logp(q) - model.logp(q0)
    val = min(1, np.exp(val))

    return val


# Source: distributions/distribution.py SHA256 2463527740fff9b808f272b4d9bae90acd914ce81c9a80a01e67269e3316d961


class Distribution(object):
    """An abstract class which implements probability distributions"""

    def __init__(self):
        """Constructor."""
        self.domain = None
        self.dim = None
        super(Distribution, self).__init__()

    def draw_samples(self, method, *args):
        """Returns random samples drawn from distribution."""
        raise NotImplementedError("Implement in a child class.")

    def draw_samples_random(self, method, *args):
        """Returns random samples drawn from distribution."""
        raise NotImplementedError("Implement in a child class.")

    def get_mean(self):
        """Returns the mean of the distribution."""
        raise NotImplementedError("Implement in a child class.")

    def get_variance(self):
        """Returns the variance of the distribution."""
        raise NotImplementedError("Implement in a child class.")

    def get_domain(self):
        """Returns domain"""
        return self.domain

    def get_dim(self):
        """Returns dimension"""
        return self.dim


class Discrete(Distribution):
    """Base class for Discrete distributions"""

    def __init__(self):
        """Constructor."""
        super(Discrete, self).__init__()

    def pmf(self, x):
        """Returns pmf of distribution at x"""
        raise NotImplementedError("Implement in a child class.")

    def draw_samples(self, method, size=None, init_sample=None, *args):
        """Returns samples drawn from distribution."""
        if method == "random":
            return self.draw_samples_random(size)
        elif method == "metropolis":
            return self.draw_samples_metropolis(size, init_sample, *args)
        elif method == "binarymetropolis":
            return self.draw_samples_binary_metropolis(size, init_sample, *args)

    def draw_samples_metropolis(self, size, init_sample, *args):
        """Returns samples drawn from distribution using Metropolis sampler"""
        sampler = Metropolis(self, True, *args)
        return sampler.sample(init_sample, size)

    def draw_samples_binary_metropolis(self, size, init_sample, *args):
        """Returns samples drawn from distribution using Binary Metropolis sampler"""
        sampler = _unsupported()
        return sampler.sample(init_sample, size)


class Continuous(Distribution):
    """Base class for Continuous distributions"""

    def __init__(self):
        """Constructor."""
        super(Continuous, self).__init__()

    def pdf(self, x):
        """Returns pdf of distribution at x"""
        raise NotImplementedError("Implement in a child class.")

    def draw_samples(self, method, size=None, init_sample=None, *args):
        """Returns samples drawn from distribution."""
        if method == "random":
            return self.draw_samples_random(size)
        elif method == "nuts":
            return self.draw_samples_nuts(size, init_sample, *args)
        elif method == "slice":
            return self.draw_samples_slice(size, init_sample, *args)
        elif method == "metropolis":
            return self.draw_samples_metropolis(size, init_sample, *args)

    def draw_samples_slice(self, size, init_sample, *args):
        """Returns samples drawn from distribution using Slice sampler"""
        sampler = Slice(self, *args)
        return sampler.sample(init_sample, size)

    def draw_samples_nuts(self, size, init_sample, *args):
        """Returns samples drawn from distribution using NUTS sampler"""
        sampler = _unsupported()
        return sampler.sample(init_sample, size, *args)

    def draw_samples_metropolis(self, size, init_sample, *args):
        """Returns samples drawn from distribution using Metropolis sampler"""
        sampler = Metropolis(self, *args)
        return sampler.sample(init_sample, size)


# Source: distributions/continuous.py SHA256 56b95190938af986f02d068b786b6d3afa00c34a3b76ea8a66429a0c8053089e


class ContinuousUniform(Continuous):
    """Continuous Uniform Distribution"""

    def __init__(self, lower, upper):
        """Constructor."""
        super(ContinuousUniform, self).__init__()
        self.lower = float(lower)
        self.upper = float(upper)
        self.dim = 1
        self.domain = None

    def pdf(self, x):
        """Returns value of pdf at x"""
        if x < self.lower or x > self.upper:
            return 0
        return np.asscalar(1 / (self.upper - self.lower))

    def draw_samples_random(self, size=None):
        """Returns random samples drawn from distribution."""
        return np.random.uniform(self.lower, self.upper, size)

    def logp(self, x):
        """Returns the log pdf at x"""
        if x < self.lower or x > self.upper:
            return -np.inf
        return -np.log(self.upper - self.lower)

    def grad_logp(self, x):
        """Returns gradient of log pdf at x"""
        return 0

    def get_parameters(self):
        """Returns parameters"""
        return self.lower, self.upper

    def get_mean(self):
        """Returns mean"""
        return (self.lower + self.upper) / 2

    def get_variance(self):
        """Returns variance"""
        return (np.power((self.upper - self.lower), 2)) / 12

    def __str__(self):
        """Returns a string representation."""
        return "Continuous Uniform: lower=%0.3f, upper=%.3f" % (self.lower, self.upper)


# Source: distributions/discrete.py SHA256 863cdf40818977bb861c886f29ff7773c8f2873dd9218959ed90341832e10193


class Categorical(Discrete):
    """Categorical Distribution"""

    def __init__(self, categories, p):
        """Constructor"""
        super(Categorical, self).__init__()
        self.cat = list(categories)
        self.k = len(categories)
        self.p = np.array(p)
        self.dim = 1
        self.domain = None

    def pmf(self, i):
        """Returns pmf of distribution at x."""
        if i < 0 or i >= self.k:
            return 0
        return np.asscalar(self.p[i])

    def draw_samples_random(self, size=None):
        """Returns random samples drawn from distribution."""
        samples = np.random.multinomial(1, self.p, size)
        return np.argmax(samples, len(samples.shape) - 1)

    def get_category(self, i):
        """Returns category value."""
        if i < 0 or i >= self.k:
            return None
        return self.cat[int(i)]

    def get_parameters(self):
        """Returns parameters"""
        return self.cat, self.p

    def logp(self, value):
        """Returns log probability of category"""
        if hasattr(value, "__len__"):
            if len(value) == 1:
                value = np.asscalar(value)
            else:
                raise ValueError("Input dimension should be 1.")

        if value < 0 or value >= self.k:
            return -np.inf
        return np.log(self.p[value])

    def get_id(self, category):
        """Returns id for the category."""
        if category is None or np.isnan(category):
            return -1
        return self.cat.index(category)

    def __str__(self):
        """Returns a string representation."""
        return "Categorical: k=%0.3f, p=%0.3f" % (self.k, self.p)


# Source: distributions/model.py SHA256 5df8c521d257d9cb8cf965a3d24aa940950a47df7cf3403153473d8444c751e5


class Model(Distribution):
    """Class for abstract distributions"""

    def __init__(self, pdf, logp, grad_logp):
        super(Model, self).__init__()
        self._pdf = pdf
        self._logp = logp
        self._grad_logp = grad_logp
        self.dim = None

    def pdf(self, x):
        """Returns pdf of distribution at x"""
        return np.asscalar(self._pdf(x))

    def logp(self, x):
        """Returns log of pdf at x"""
        return self._logp(x)

    def grad_logp(self, x):
        """Returns gradient of log pdf at x"""
        return self._grad_logp(x)

    def draw_samples(self, method, size=None, init_sample=None, *args):
        """Returns samples drawn from distribution."""
        if method == "nuts":
            return self.draw_samples_nuts(size, init_sample, *args)
        elif method == "slice":
            return self.draw_samples_slice(size, init_sample, *args)
        elif method == "metropolis":
            return self.draw_samples_metropolis(size, init_sample, *args)
        elif method == "binarymetropolis":
            return self.draw_samples_binary_metropolis(size, init_sample, *args)

    def draw_samples_slice(self, size, init_sample, *args):
        """Returns samples drawn from distribution using Slice sampler"""
        sampler = Slice(self)
        return sampler.sample(init_sample, size, *args)

    def draw_samples_nuts(self, size, init_sample, *args):
        """Returns samples drawn from distribution using NUTS sampler"""
        sampler = _unsupported()
        return sampler.sample(init_sample, size, *args)

    def draw_samples_metropolis(self, size, init_sample, *args):
        """Returns samples drawn from distribution using Metropolis sampler"""
        if hasattr(init_sample, "__len__"):
            self.dim = len(init_sample)
        else:
            self.dim = 1
        sampler = Metropolis(self, True, *args)
        return sampler.sample(init_sample, size)

    def draw_samples_binary_metropolis(self, size, init_sample, *args):
        """Returns samples drawn from distribution using Binary Metropolis sampler"""
        sampler = _unsupported()
        return sampler.sample(init_sample, size)


continuous = Namespace(ContinuousUniform=ContinuousUniform)
discrete = Namespace(Categorical=Categorical)


# Source: gp/kernel.py SHA256 e45d9df6a6e43416270db0e51a4c4f825a9359ea1d3a18a2f81180e879f1eb3e


def _get_se_matern_scale_bw_strs(kern):
    """Gets descriptions of the bandwidths and scales for the SE and matern kernels."""
    if kern.dim > 6:
        bw_str = "avg-bw: %0.4f" % (kern.hyperparams["dim_bandwidths"].mean())
    else:
        bw_str = (
            "bws:["
            + " ".join(["%0.2f" % (dbw) for dbw in kern.hyperparams["dim_bandwidths"]])
            + "]"
        )
    scale_str = "sc:%0.4f" % (kern.hyperparams["scale"])
    return scale_str, bw_str


class Kernel(object):
    """A kernel class."""

    def __init__(self):
        """Constructor."""
        super(Kernel, self).__init__()
        self.hyperparams = {}

    def is_guaranteed_psd(self):
        """The child class should implement this method to indicate whether it is
        guaranteed to be PSD."""
        raise NotImplementedError("Implement in a child class.")

    def __call__(self, X1, X2=None):
        """Evaluates the kernel by calling evaluate."""
        return self.evaluate(X1, X2)

    def evaluate(self, X1, X2=None):
        """Evaluates kernel values between X1 and X2 and returns an n1xn2 kernel matrix.
        This is a wrapper for _child_evaluate.
        """
        X2 = X1 if X2 is None else X2
        if len(X1) == 0 or len(X2) == 0:
            return np.zeros((len(X1), len(X2)))
        return self._child_evaluate(X1, X2)

    def _child_evaluate(self, X1, X2):
        """Evaluates kernel values between X1 and X2 and returns an n1xn2 kernel matrix.
        This is to be implemented in a child kernel.
        """
        raise NotImplementedError("Implement in a child class.")

    def evaluate_from_dists(self, dists):
        """Evaluates the kernel from pairwise distances."""
        raise NotImplementedError("Implement in a child class.")

    def set_hyperparams(self, **kwargs):
        """Set hyperparameters here."""
        self.hyperparams = kwargs

    def add_hyperparams(self, **kwargs):
        """Set additional hyperparameters here."""
        for key, value in kwargs.items():
            self.hyperparams[key] = value

    def get_effective_norm(self, X, order=None, *args, **kwargs):
        """Gets the effective norm scaled by bandwidths."""
        raise NotImplementedError("Implement in a child class.")

    def compute_std_slack(self, X1, X2):
        """Computes a bound on the maximum standard deviation between X1 and X2."""
        raise NotImplementedError("Implement in a child class.")

    def change_smoothness(self, factor):
        """Decreases smoothness by the factor given."""
        raise NotImplementedError("Implement in a child class.")

    def gradient(self, param, X1, X2=None, *args):
        """Computes gradient of kernel w.r.t the param"""
        if len(X1) == 0 or len(X2) == 0:
            return np.zeros((len(X1), len(X2)))
        X2 = X1 if X2 is None else X2
        return self._child_gradient(param, X1, X2, *args)

    def _child_gradient(self, param, X1, X2, param_num=None):
        """Computes gradient of kernel w.r.t the param"""
        raise NotImplementedError("Implement in a child class.")

    def __str__(self):
        """Returns a string representation of the kernel."""
        return "%s:: %s" % (type(self), str(self.hyperparams))


class SEKernel(Kernel):
    """Squared exponential kernel."""

    def __init__(self, dim, scale=None, dim_bandwidths=None):
        """Constructor. dim is the dimension."""
        super(SEKernel, self).__init__()
        self.dim = dim
        self.set_se_hyperparams(scale, dim_bandwidths)

    def is_guaranteed_psd(self):
        """The child class should implement this method to indicate whether it is
        guaranteed to be PSD."""
        return True

    def set_dim_bandwidths(self, dim_bandwidths):
        """Sets the bandwidth for each dimension."""
        if dim_bandwidths is not None:
            if len(dim_bandwidths) != self.dim:
                raise ValueError("Dimension of dim_bandwidths should be the same as dimension.")
            dim_bandwidths = np.array(dim_bandwidths).T
        self.add_hyperparams(dim_bandwidths=dim_bandwidths)

    def set_single_bandwidth(self, bandwidth):
        """Sets the bandwidth of all dimensions to be the same value."""
        dim_bandwidths = None if bandwidth is None else [bandwidth] * self.dim
        self.set_dim_bandwidths(dim_bandwidths)

    def set_scale(self, scale):
        """Sets the scale parameter for the kernel."""
        self.add_hyperparams(scale=scale)

    def set_se_hyperparams(self, scale, dim_bandwidths):
        """Sets both the scale and the dimension bandwidths for the SE kernel."""
        self.set_scale(scale)
        if hasattr(dim_bandwidths, "__len__"):
            self.set_dim_bandwidths(dim_bandwidths)
        else:
            self.set_single_bandwidth(dim_bandwidths)

    def _child_evaluate(self, X1, X2):
        """Evaluates the SE kernel between X1 and X2 and returns the gram matrix."""
        scaled_X1 = self.get_scaled_repr(X1)
        scaled_X2 = self.get_scaled_repr(X2)
        dist_sq = dist_squared(scaled_X1, scaled_X2)
        K = self.hyperparams["scale"] * np.exp(-dist_sq / 2)
        return K

    def get_scaled_repr(self, X):
        """Returns the scaled version of an input by the bandwidths."""
        return X / self.hyperparams["dim_bandwidths"]

    def get_effective_norm(self, X, order=None, is_single=True):
        """Gets the effective norm. That is the norm of X scaled by bandwidths."""
        # pylint: disable=arguments-differ
        scaled_X = self.get_scaled_repr(X)
        if is_single:
            return np.linalg.norm(scaled_X, ord=order)
        else:
            return np.array([np.linalg.norm(sx, ord=order) for sx in scaled_X])

    def compute_std_slack(self, X1, X2):
        """Computes a bound on the maximum standard deviation diff between X1 and X2."""
        k_12 = np.array(
            [
                float(self.evaluate(X1[i].reshape(1, -1), X2[i].reshape(1, -1)))
                for i in range(len(X1))
            ]
        )
        return np.sqrt(self.hyperparams["scale"] - k_12)

    def change_smoothness(self, factor):
        """Decreases smoothness by the given factor."""
        self.hyperparams["dim_bandwidths"] *= factor

    def _child_gradient(self, param, X1, X2, param_num=None):
        """Computes gradient of kernel w.r.t the param"""
        scaled_X1 = self.get_scaled_repr(X1)
        scaled_X2 = self.get_scaled_repr(X2)
        dist_sq = dist_squared(scaled_X1, scaled_X2)
        if param == "scale":
            return self.hyperparams["scale"] * np.exp(-dist_sq / 2)
        elif param == "same_dim_bandwidths":
            dist_sq_dv = dist_sq / (self.hyperparams["dim_bandwidths"][0, 0])
            return self.hyperparams["scale"] * (np.multiply(dist_sq_dv, np.exp(-dist_sq / 2)))
        else:
            dim_X1 = np.expand_dims(scaled_X1[:, param_num], axis=1)
            dim_X2 = np.expand_dims(scaled_X2[:, param_num], axis=1)
            dim_sq = dist_squared(dim_X1, dim_X2)
            dim_sq = dim_sq / (self.hyperparams["dim_bandwidths"][0, param_num])
            return self.hyperparams["scale"] * (np.multiply(dim_sq, np.exp(-dist_sq / 2)))

    def __str__(self):
        """Returns a string representation of the kernel."""
        scale_str, bw_str = _get_se_matern_scale_bw_strs(self)
        return "SE: " + scale_str + " " + bw_str


class MaternKernel(Kernel):
    """The Matern class of kernels."""

    # pylint: disable=abstract-method

    def __init__(self, dim, nu=None, scale=None, dim_bandwidths=None):
        """Constructor. dim is the dimension."""
        super(MaternKernel, self).__init__()
        self.dim = dim
        self.p = None
        self.norm_constant = None
        self.set_matern_hyperparams(nu, scale, dim_bandwidths)

    def is_guaranteed_psd(self):
        """The child class should implement this method to indicate whether it is
        guaranteed to be PSD."""
        return True

    def set_matern_hyperparams(self, nu, scale, dim_bandwidths):
        """Sets the parameters of the matern kernel."""
        if nu % 1 != 0.5:
            raise ValueError("Matern kernel: nu has to be p + 0.5 where p is an integer.")
        self.add_hyperparams(nu=nu)
        self.add_hyperparams(scale=scale)
        dim_bandwidths = (
            dim_bandwidths if hasattr(dim_bandwidths, "__len__") else [dim_bandwidths] * self.dim
        )
        dim_bandwidths = np.array(dim_bandwidths).T
        self.add_hyperparams(dim_bandwidths=dim_bandwidths)
        self.p = int(nu)
        self.norm_constant = 1.0 / self._eval_kernel_values_unnormalised(0)

    def get_scaled_repr(self, X):
        """Returns scaled versions of the input by the bandwidths."""
        return X / self.hyperparams["dim_bandwidths"]

    def _eval_kernel_values_unnormalised(self, dist):
        """Given the distances, evaluates the unnormalised kernel value."""
        unnorm_K = np.zeros(dist.shape) if isinstance(dist, np.ndarray) else 0
        for i in range(self.p + 1):
            coeff = np.math.factorial(self.p + i) / (
                np.math.factorial(i) * np.math.factorial(self.p - i)
            )
            mult_mat = np.sqrt(8 * self.hyperparams["nu"]) * dist
            unnorm_K += coeff * (mult_mat) ** (self.p - i)
        # Finally multiply by the leading exponential
        unnorm_K *= (
            np.math.gamma(self.p + 1)
            / np.math.gamma(2 * self.p + 1)
            * np.exp(-np.sqrt(2 * self.hyperparams["nu"]) * dist)
        )
        return unnorm_K

    def _eval_grad_kernel_values_unnormalised(self, dist, dist_dv):
        """Given the distances, evaluates the gradient of unnormalised kernel value."""
        unnorm_K = np.zeros(dist.shape) if isinstance(dist, np.ndarray) else 0
        unnorm_K_dv = np.zeros(dist_dv.shape) if isinstance(dist_dv, np.ndarray) else 0
        for i in range(self.p + 1):
            coeff = np.math.factorial(self.p + i) / (
                np.math.factorial(i) * np.math.factorial(self.p - i)
            )
            mult_mat = np.sqrt(8 * self.hyperparams["nu"]) * dist
            unnorm_K += coeff * (mult_mat) ** (self.p - i)
            if self.p - i > 0:
                unnorm_K_dv += (
                    (np.sqrt(8 * self.hyperparams["nu"]))
                    * (self.p - i)
                    * coeff
                    * ((mult_mat) ** (self.p - i - 1))
                    * dist_dv
                )
        # Finally multiply by the leading exponential
        unnorm_K_dv *= (
            np.math.gamma(self.p + 1)
            / np.math.gamma(2 * self.p + 1)
            * np.exp(-np.sqrt(2 * self.hyperparams["nu"]) * dist)
        )
        unnorm_K *= (
            np.math.gamma(self.p + 1)
            / np.math.gamma(2 * self.p + 1)
            * np.exp(-np.sqrt(2 * self.hyperparams["nu"]) * dist)
            * (-np.sqrt(2 * self.hyperparams["nu"]) * dist_dv)
        )
        return unnorm_K + unnorm_K_dv

    def _child_evaluate(self, X1, X2):
        """Evaluates the SE kernel between X1 and X2 and returns the gram matrix."""
        scaled_X1 = self.get_scaled_repr(X1)
        scaled_X2 = self.get_scaled_repr(X2)
        dist = np.sqrt(dist_squared(scaled_X1, scaled_X2))
        unnorm_kernel_vals = self._eval_kernel_values_unnormalised(dist)
        K = self.hyperparams["scale"] * self.norm_constant * unnorm_kernel_vals
        return K

    def _child_gradient(self, param, X1, X2, param_num=None):
        """Computes gradient of kernel w.r.t the param."""
        scaled_X1 = self.get_scaled_repr(X1)
        scaled_X2 = self.get_scaled_repr(X2)
        dist = np.sqrt(dist_squared(scaled_X1, scaled_X2))
        if param == "scale":
            unnorm_kernel_vals = self._eval_kernel_values_unnormalised(dist)
            return self.hyperparams["scale"] * self.norm_constant * unnorm_kernel_vals
        elif param == "same_dim_bandwidths":
            dist_dv = -(dist / (self.hyperparams["dim_bandwidths"][0, 0]))
            unnorm_kernel_vals = self._eval_grad_kernel_values_unnormalised(dist, dist_dv)
            return self.hyperparams["scale"] * self.norm_constant * unnorm_kernel_vals
        else:
            dim_X1 = np.expand_dims(scaled_X1[:, param_num], axis=1)
            dim_X2 = np.expand_dims(scaled_X2[:, param_num], axis=1)
            np.fill_diagonal(dist, 1.0)
            dist_dv = 1 / dist
            np.fill_diagonal(dist, 0.0)
            dim_sq = dist_squared(dim_X1, dim_X2)
            dist_dv *= -(dim_sq / (self.hyperparams["dim_bandwidths"][0, param_num]))
            unnorm_kernel_vals = self._eval_grad_kernel_values_unnormalised(dist, dist_dv)
            return self.hyperparams["scale"] * self.norm_constant * unnorm_kernel_vals

    def __str__(self):
        """Returns a string representation of the kernel."""
        scale_str, bw_str = _get_se_matern_scale_bw_strs(self)
        nu_str = "nu=%0.1f" % (self.hyperparams["nu"])
        return "Matern: " + nu_str + " " + scale_str + " " + bw_str


class AdditiveKernel(Kernel):
    """Implements an Additive kernel on Euclidean Spaces with non-overlapping groups."""

    # pylint: disable=abstract-method

    def __init__(self, scale, kernel_list, groupings):
        """Constructor:
        kernel_list is a list of kernel objects.
        groupings is a list of list first dimension is the group second dimension is the
        set of dimensions in the group.
        """
        if len(kernel_list) != len(groupings):
            raise ValueError("number of kernels do not correspond to number of groups.")
        super(AdditiveKernel, self).__init__()
        self.kernel_list = kernel_list
        self.groupings = groupings
        self.add_hyperparams(scale=scale)
        self.dim = sum([kern.dim for kern in self.kernel_list])

    def is_guaranteed_psd(self):
        """The child class should implement this method to indicate whether it is
        guaranteed to be PSD. Returns True if each kernel is is_guaranteed_psd()"""
        return all([kern.is_guaranteed_psd() for kern in self.kernel_list])

    def _child_evaluate(self, X1, X2):
        "computes the gram matrix"
        # Convert them to arrays so that we can index the groups out.
        X1 = np.array(X1)
        X2 = np.array(X2)
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        result = np.zeros((n1, n2))
        for kernel, group in zip(self.kernel_list, self.groupings):
            result += kernel(X1[:, group], X2[:, group])
        return self.hyperparams["scale"] * result

    def __str__(self):
        """Returns a string representation of the kernel."""
        kernels_str_list = [
            "%s(%s)" % (grp, kern) for (grp, kern) in zip(self.groupings, self.kernel_list)
        ]
        kernels_str = ", ".join(kernels_str_list)
        return "ADD scale=%0.2f, " % (self.hyperparams["scale"]) + kernels_str


gp_kernel = Namespace(
    SEKernel=SEKernel,
    MaternKernel=MaternKernel,
    AdditiveKernel=AdditiveKernel,
    PolyKernel=_UnsupportedKernel,
    ESPKernelSE=_UnsupportedKernel,
    ESPKernelMatern=_UnsupportedKernel,
    ExpDecayKernel=_UnsupportedKernel,
)


# Source: gp/gp_core.py SHA256 052c5dd6043714e7beae058f3f46a10dc8c023371aab9f25d4efbaac639026cf

mandatory_gp_args = [
    get_option_specs(
        "hp_tune_criterion",
        False,
        "ml",
        "Which criterion to use when tuning hyper-parameters. Other "
        + "options are post_sampling and post_mean.",
    ),
    get_option_specs(
        "hp_tune_probs",
        False,
        "uniform",
        "With what probability should we choose each strategy given in hp_tune_criterion."
        + 'If "uniform" we we will use uniform probabilities and if "adaptive" we will use '
        + "adaptive probabilities which weight acquisitions according to how well they do.",
    ),
    get_option_specs(
        "ml_hp_tune_opt",
        False,
        "default",
        "Which optimiser to use when maximising the tuning criterion.",
    ),
    get_option_specs(
        "hp_tune_max_evals",
        False,
        -1,
        "How many evaluations to use when maximising the tuning criterion.",
    ),
    get_option_specs(
        "handle_non_psd_kernels", False, "guaranteed_psd", "How to handle kernels that are non-psd."
    ),
    # The mean and noise variance of the GP
    get_option_specs(
        "mean_func",
        False,
        None,
        ("The mean function. If not None, will use this instead of the" + "other options below"),
    ),
    get_option_specs(
        "mean_func_type",
        False,
        "tune",
        (
            "Specify the type of mean function. Should be mean, median, const ",
            "zero, or tune. If const, specifcy value in mean-func-const.",
        ),
    ),
    get_option_specs(
        "mean_func_const", False, 0.0, "The constant value to use if mean_func_type is const."
    ),
    get_option_specs(
        "noise_var_type",
        False,
        "tune",
        (
            "Specify how to obtain the noise variance. Should be tune, label or value. "
            "Specify appropriate value in noise_var_label or noise_var_value"
        ),
    ),
    get_option_specs(
        "noise_var_label", False, 0.05, "The fraction of label variance to use as noise variance."
    ),
    get_option_specs(
        "noise_var_value", False, 0.1, "The (absolute) value to use as noise variance."
    ),
    get_option_specs(
        "post_hp_tune_method",
        False,
        "slice",
        "Which sampling to use when maximising the tuning criterion. Other " + "option is nuts.",
    ),
    get_option_specs(
        "post_hp_tune_burn", False, -1, "How many initial samples to ignore during sampling."
    ),
    get_option_specs(
        "post_hp_tune_offset", False, 25, "How many samples to ignore between samples."
    ),
]


def _check_feature_label_lengths_and_format(X, Y):
    """Checks if the length of X and Y are the same."""
    if len(X) != len(Y):
        raise ValueError("Length of X (" + len(X) + ") and Y (" + len(Y) + ") do not match.")


def _get_default_ml_hp_tune_opt(num_hps):
    """Return default optimisation method for maximum likelihood."""
    if num_hps > 60:
        return "pdoo"
    else:
        return "direct"


class GP(object):
    """Base class for Gaussian processes."""

    # pylint: disable=attribute-defined-outside-init

    def __init__(
        self,
        X,
        Y,
        kernel,
        mean_func,
        noise_var,
        build_posterior=True,
        reporter=None,
        handle_non_psd_kernels="guaranteed_psd",
    ):
        """Constructor."""
        super(GP, self).__init__()
        _check_feature_label_lengths_and_format(X, Y)
        self.set_data(X, Y, build_posterior=False)
        self.kernel = kernel
        self.mean_func = mean_func
        self.noise_var = noise_var
        self.reporter = get_reporter(reporter)
        self.handle_non_psd_kernels = handle_non_psd_kernels
        # Some derived attribues.
        self.num_tr_data = len(self.Y)
        # Initialise other attributes we will need.
        self.L = None
        self.alpha = None
        self.K_trtr_wo_noise = None
        self._set_up()
        # Build posterior if necessary
        if build_posterior:
            self.build_posterior()

    def _set_up(self):
        """Additional set up."""
        # Test that if the kernel is not guaranteed to be PSD, then have have a way to
        # handle it.
        if not self.kernel.is_guaranteed_psd():
            assert self.handle_non_psd_kernels in ["project_first", "try_before_project"]

    def _write_message(self, msg):
        """Writes a message via the reporter or the std out."""
        if self.reporter:
            self.reporter.write(msg)
        else:
            sys.stdout.write(msg)

    def set_data(self, X, Y, build_posterior=True):
        """Sets the data to X and Y."""
        self.X = list(X)
        self.Y = list(Y)
        self.num_tr_data = len(self.Y)
        if build_posterior:
            self.build_posterior()

    def add_data_single(self, x_new, y_new, *args, **kwargs):
        """Adds a single data point."""
        self.add_data_multiple([x_new], [y_new], *args, **kwargs)

    def add_data_multiple(self, X_new, Y_new, build_posterior=True):
        """Adds new data to the GP. If rebuild is true it rebuilds the posterior."""
        _check_feature_label_lengths_and_format(X_new, Y_new)
        self.X.extend(X_new)
        self.Y.extend(Y_new)
        self.num_tr_data = len(self.Y)
        if build_posterior:
            self.build_posterior()

    # Methods for computing the posterior ----------------------------------------------
    def _get_training_kernel_matrix(self):
        """Returns the training kernel matrix. Writing this as a separate method in case,
        the kernel computation can be done efficiently for a child class.
        """
        return self.kernel(self.X, self.X)

    def build_posterior(self):
        """Builds the posterior GP by computing the mean and covariance."""
        K_trtr_wo_noise = self._get_training_kernel_matrix()
        self.K_trtr_wo_noise = K_trtr_wo_noise
        self.L = _get_cholesky_decomp(K_trtr_wo_noise, self.noise_var, self.handle_non_psd_kernels)
        Y_centred = self.Y - self.mean_func(self.X)
        self.alpha = solve_upper_triangular(self.L.T, solve_lower_triangular(self.L, Y_centred))

    def eval(self, X_test, uncert_form="none"):
        """Evaluates the GP on X_test. If uncert_form is
        covar: returns the entire covariance on X_test (nxn matrix)
        std: returns the standard deviations on the test set (n vector)
        none: returns nothing (default).
        """
        # Compute the posterior mean.
        test_mean = self.mean_func(X_test)
        K_tetr = self.kernel(X_test, self.X)
        pred_mean = test_mean + K_tetr.dot(self.alpha)
        # Compute the posterior variance or standard deviation as required.
        if uncert_form == "none":
            uncert = None
        else:
            K_tete = self.kernel(X_test, X_test)
            V = solve_lower_triangular(self.L, K_tetr.T)
            post_covar = K_tete - V.T.dot(V)
            post_covar = get_post_covar_from_raw_covar(
                post_covar, self.noise_var, self.kernel.is_guaranteed_psd()
            )
            if uncert_form == "covar":
                uncert = post_covar
            elif uncert_form == "std":
                uncert = np.sqrt(np.diag(post_covar))
            else:
                raise ValueError("uncert_form should be none, covar or std.")
        return pred_mean, uncert

    def eval_with_hallucinated_observations(self, X_test, X_halluc, uncert_form="none"):
        """Evaluates the GP with additional hallucinated observations in the
        kernel matrix."""
        pred_mean, _ = self.eval(X_test, uncert_form="none")  # Just compute the means.
        if uncert_form == "none":
            uncert = None
        else:
            # Computed the augmented kernel matrix and its cholesky decomposition.
            X_aug = self.X + X_halluc
            K_haha = self.kernel(X_halluc, X_halluc)  # kernel for hallucinated data
            K_trha = self.kernel(self.X, X_halluc)  # kernel with training and hallucinated data
            aug_K_trtr_wo_noise = np.vstack(
                (np.hstack((self.K_trtr_wo_noise, K_trha)), np.hstack((K_trha.T, K_haha)))
            )
            aug_L = _get_cholesky_decomp(
                aug_K_trtr_wo_noise, self.noise_var, self.handle_non_psd_kernels
            )
            # Augmented kernel matrices for the test data
            aug_K_tete = self.kernel(X_test, X_test)
            aug_K_tetr = self.kernel(X_test, X_aug)
            aug_V = solve_lower_triangular(aug_L, aug_K_tetr.T)
            aug_post_covar = aug_K_tete - aug_V.T.dot(aug_V)
            aug_post_covar = get_post_covar_from_raw_covar(
                aug_post_covar, self.noise_var, self.kernel.is_guaranteed_psd()
            )
            if uncert_form == "covar":
                uncert = aug_post_covar
            elif uncert_form == "std":
                uncert = np.sqrt(np.diag(aug_post_covar))
            else:
                raise ValueError("uncert_form should be none, covar or std.")
        return (pred_mean, uncert)

    def compute_log_marginal_likelihood(self):
        """Computes the log marginal likelihood."""
        Y_centred = self.Y - self.mean_func(self.X)
        ret = (
            -0.5 * Y_centred.T.dot(self.alpha)
            - (np.log(np.diag(self.L))).sum()
            - 0.5 * self.num_tr_data * np.log(2 * np.pi)
        )
        return ret

    def compute_grad_log_marginal_likelihood(self, param, *args):
        """Computes the gradient of log marginal likelihood."""
        alpha = np.expand_dims(self.alpha, axis=0)
        if param == "noise_var":
            grad_m = self.noise_var * np.identity(len(self.X))
        elif param == "noise_mean":
            return np.asscalar(np.matmul(alpha, np.ones((len(self.Y), 1))))
        else:
            grad_m = self.kernel.gradient(param, self.X, self.X, *args)
        grad_m = np.matmul(alpha.T, np.matmul(alpha, grad_m)) - solve_upper_triangular(
            self.L.T, solve_lower_triangular(self.L, grad_m)
        )
        return 0.5 * np.trace(grad_m)

    def __str__(self):
        """Returns a string representation of the GP."""
        return "%s, noise-var=%0.3f (n=%d)" % (self._child_str(), self.noise_var, len(self.Y))

    def _child_str(self):
        """String representation for child GP."""
        raise NotImplementedError("Implement in child class. !")

    def draw_samples(self, num_samples, X_test=None, mean_vals=None, covar=None):
        """Draws num_samples samples at returns their values at X_test."""
        if X_test is not None:
            mean_vals, covar = self.eval(X_test, "covar")
        return draw_gaussian_samples(num_samples, mean_vals, covar)

    def draw_samples_with_hallucinated_observations(self, num_samples, X_test, X_halluc):
        """Draws samples with hallucinated observations."""
        mean_vals, aug_covar = self.eval_with_hallucinated_observations(
            X_test, X_halluc, uncert_form="covar"
        )
        return draw_gaussian_samples(num_samples, mean_vals, aug_covar)

    def visualise(
        self, file_name=None, boundary=None, true_func=None, num_samples=20, conf_width=3
    ):
        """Visualises the GP."""
        # pylint: disable=unused-variable
        # pylint: disable=too-many-locals
        if hasattr(self.kernel, "dim") and self.kernel.dim != 1:
            self._write_message("Cannot visualise only in 1 dimension.\n")
        else:
            import matplotlib.pyplot as plt

            fig = plt.figure()
            N = 400
            leg_handles = []
            leg_labels = []
            if not boundary:
                boundary = [min(self.X), max(self.X)]
            grid = np.linspace(boundary[0], boundary[1], N).reshape((N, 1))
            (pred_vals, pred_stds) = self.eval(grid, "std")
            # Shade a high confidence region
            conf_band_up = pred_vals + conf_width * pred_stds
            conf_band_down = pred_vals - conf_width * pred_stds
            leg_conf = plt.fill_between(
                grid.ravel(), conf_band_up, conf_band_down, color=[0.9, 0.9, 0.9]
            )
            # Plot the samples
            gp_samples = self.draw_samples(num_samples, grid)
            plt.plot(grid, gp_samples.T, "--", linewidth=0.5)
            # plot the true function if available.
            if true_func:
                leg_true = plt.plot(
                    grid, true_func(grid), "b--", linewidth=3, label="true function"
                )
                leg_handles.append(leg_true)
            # Plot the posterior mean
            leg_post_mean = plt.plot(grid, pred_vals, "k-", linewidth=4, label="post mean")
            # Finally plot the training data.
            leg_data = plt.plot(self.X, self.Y, "kx", mew=4, markersize=10, label="data")
            # Finally either plot or show the figure
            if file_name is None:
                plt.show()
            else:
                fig.savefig(file_name)


class GPFitter(object):
    """Class for fitting Gaussian processes."""

    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=no-value-for-parameter

    def __init__(self, X, Y, options, reporter="default"):
        """Constructor."""
        super(GPFitter, self).__init__()
        assert len(X) == len(Y)
        self.reporter = get_reporter(reporter)
        options = load_options(mandatory_gp_args, partial_options=options)
        self.options = options
        self.X = X
        self.Y = Y
        self.num_data = len(X)
        self._set_up()

    def _set_up(self):
        """Sets up a bunch of ancillary parameters."""
        # Set place-holders for hp_bounds or hp_vals depending on whether we are using ml or
        # post_sampling/mean
        # -----------------------------------------------------------------------------
        self.cts_hp_bounds = []  # The bounds for each hyper parameter should be a
        # num_cts_hps x 2 array where the 1st/2nd columns are the
        # lowe/upper bounds.
        self.dscr_hp_vals = []  # A list of lists where each list contains a list of values
        # for each discrete hyper-parameter.
        self.param_order = []
        # Set up for common things - including mean and noise variance hyper-parameters
        # -----------------------------------------------------------------------------
        epsilon = 0.0001
        self.Y_var = np.array(self.Y).std() ** 2 + epsilon if len(self.Y) > 0 else epsilon
        self._set_up_mean_and_noise_variance_bounds()
        # Set up hyper-parameters for the child.
        # -----------------------------------------------------------------------------
        self._child_set_up()
        self._hp_tune_method_set_up()
        self.cts_hp_bounds = np.array(self.cts_hp_bounds)
        # Some post child set up
        if "ml" in self.methods_to_use:
            # The number of hyper parameters
            self.num_hps = len(self.cts_hp_bounds) + len(self.dscr_hp_vals)
            self._set_up_ml_hp_tune()
        if "post_sampling" in self.methods_to_use:
            self._set_up_post_sampling_hp_tune()
            self.num_hps = len(self.hp_priors)  # The number of hyper parameters
        if "post_mean" in self.methods_to_use:
            self._set_up_post_mean_hp_tune()
            self.num_hps = len(self.hp_priors)  # The number of hyper parameters

    def _hp_tune_method_set_up(self):
        """Sets up probabilities for hp tune methods."""
        self.methods_to_use = [elem.lower() for elem in self.options.hp_tune_criterion.split("-")]
        for method in self.methods_to_use:
            if method not in ["ml", "post_sampling", "post_mean"]:
                raise ValueError("hp_tune_criterion should be ml or post_sampling.")
        self.methods_to_use_counter = {key: 0 for key in self.methods_to_use}
        if self.options.hp_tune_probs == "uniform":
            self.hp_tune_probs = np.ones(len(self.methods_to_use)) / float(len(self.methods_to_use))
        elif self.options.hp_tune_probs == "adaptive":
            self.hp_tune_uniform_sampling_prob = 0.05
            self.hp_tune_sampling_weights = {key: 1.0 for key in self.methods_to_use}
            self.hp_tune_probs = self._get_adaptive_hp_tune_probs()
        else:
            self.hp_tune_probs = np.array([float(x) for x in self.options.hp_tune_probs.split("-")])
            if len(self.hp_tune_probs) != len(self.methods_to_use):
                self.hp_tune_probs = np.ones(len(self.methods_to_use)) / float(
                    len(self.methods_to_use)
                )
        self.hp_tune_probs = self.hp_tune_probs / self.hp_tune_probs.sum()

    def _get_adaptive_hp_tune_probs(self):
        """Computes the adaptive hyper parameter tuning method probs."""
        num_methods = len(self.methods_to_use)
        uniform_sampling_probs = (
            self.hp_tune_uniform_sampling_prob * np.ones((num_methods,)) / num_methods
        )
        hp_tune_succ_counter = np.array(
            [self.hp_tune_sampling_weights[key] for key in self.methods_to_use]
        )
        hp_tune_use_counter = np.array(
            [self.methods_to_use_counter[key] for key in self.methods_to_use]
        )
        hp_tune_weights = hp_tune_succ_counter / np.sqrt(1 + hp_tune_use_counter)
        hp_tune_norm_weights = hp_tune_weights / hp_tune_weights.sum()
        adaptive_sampling_probs = (1 - self.hp_tune_uniform_sampling_prob) * hp_tune_norm_weights
        ret = uniform_sampling_probs + adaptive_sampling_probs
        return ret / ret.sum()

    def _set_up_mean_and_noise_variance_bounds(self):
        """Sets up bounds for the mean value and the noise."""
        # 1. The prior mean for the GP.
        if (
            not (hasattr(self.options, "mean_func") and self.options.mean_func is not None)
            and self.options.mean_func_type == "tune"
        ):
            Y_std = np.sqrt(self.Y_var)
            if len(self.Y) > 0:
                Y_median = np.median(self.Y)
                Y_half_range = 0.5 * (max(self.Y) - min(self.Y)) if len(self.Y) > 0 else 1.0
            else:
                Y_median = 0.0
                Y_half_range = 1.0
            Y_width = 0.5 * (Y_half_range + Y_std)
            self.mean_func_bounds = [Y_median - 3 * Y_width, Y_median + 3 * Y_width]
            self.cts_hp_bounds.append(self.mean_func_bounds)
            self.param_order.append(["noise_mean", "cts"])
        # 2. The noise variance
        if self.options.noise_var_type == "tune":
            self.noise_var_log_bounds = [np.log(0.005 * self.Y_var), np.log(0.2 * self.Y_var)]
            self.cts_hp_bounds.append(self.noise_var_log_bounds)
            self.param_order.append(["noise_var", "cts"])

    def _child_set_up(self):
        """Here you should set up parameters for the child, such as the bounds for the
        optimiser etc."""
        raise NotImplementedError("Implement _child_set_up in a child method.")

    def _set_up_ml_hp_tune(self):
        """Sets up optimiser for ml hp tune."""

        # pylint: disable=too-many-branches
        # define the following internal functions to abstract things out more.
        def _direct_wrap(*args):
            """A wrapper so as to only return the optimal point."""
            opt_val, opt_pt, _ = direct_ft_maximise(*args)
            return opt_val, opt_pt, None

        def _pdoo_wrap(*args):
            """A wrapper so as to only return the optimal point."""
            opt_val, opt_pt, _ = pdoo_maximise(*args)
            return opt_val, opt_pt, None

        def _rand_wrap(*args):
            """A wrapper so as to only return the optimal point."""
            opt_val, opt_pt, _ = random_maximise(*args, vectorised=False)
            return opt_val, opt_pt, None

        def _rand_exp_sampling_wrap(*args):
            """A wrapper so as to only return the optimal point."""
            sample_cts_hps, sample_dscr_hps, lml_vals = random_sample_cts_dscr(
                *args, vectorised=False
            )
            sample_probs = np.exp(lml_vals - max(lml_vals))
            sample_probs = sample_probs / sample_probs.sum()
            return sample_cts_hps, sample_dscr_hps, sample_probs

        # Set some parameters
        if self.options.ml_hp_tune_opt == "default":
            self.ml_hp_tune_opt_method = _get_default_ml_hp_tune_opt(self.num_hps)
        else:
            self.ml_hp_tune_opt_method = self.options.ml_hp_tune_opt
        if (
            hasattr(self.options, "hp_tune_max_evals")
            and self.options.hp_tune_max_evals is not None
            and self.options.hp_tune_max_evals > 0
        ):
            self.hp_tune_max_evals = self.options.hp_tune_max_evals
        else:
            if self.ml_hp_tune_opt_method in ["direct", "pdoo"]:
                self.hp_tune_max_evals = min(1e4, max(500, self.num_hps * 50))
            elif self.ml_hp_tune_opt_method == "rand":
                self.hp_tune_max_evals = min(1e4, max(500, self.num_hps * 200))
            elif self.ml_hp_tune_opt_method == "rand_exp_sampling":
                self.hp_tune_max_evals = min(1e5, max(500, self.num_hps * 400))
        # Now set up the function that will be used for optimising
        if self.ml_hp_tune_opt_method == "direct":
            self.cts_hp_optimise = lambda obj, max_evals: _direct_wrap(
                obj, self.cts_hp_bounds, max_evals
            )
        elif self.ml_hp_tune_opt_method == "pdoo":
            self.cts_hp_optimise = lambda obj, max_evals: _pdoo_wrap(
                obj, self.cts_hp_bounds, max_evals
            )
        elif self.ml_hp_tune_opt_method == "rand":
            self.cts_hp_optimise = lambda obj, max_evals: _rand_wrap(
                obj, self.cts_hp_bounds, max_evals
            )
        elif self.ml_hp_tune_opt_method == "rand_exp_sampling":
            self.hp_sampler = lambda obj, max_evals: _rand_exp_sampling_wrap(
                obj, self.cts_hp_bounds, self.dscr_hp_vals, max_evals
            )

    def _set_up_post_sampling_hp_tune(self):
        """Sets up posterior sampling for tuning the parameters of the GP."""
        if self.options.post_hp_tune_method == "slice":
            self.hp_sampler_cts = lambda model, init_sample, num_samples, burn: model.draw_samples(
                "slice", num_samples, init_sample, burn
            )
        elif self.options.post_hp_tune_method == "nuts":
            self.hp_sampler_cts = lambda model, init_sample, num_samples, burn: model.draw_samples(
                "nuts", num_samples, init_sample, burn
            )

        self.hp_sampler_dscr = lambda model, init_sample, num_samples: model.draw_samples(
            "metropolis", num_samples, init_sample
        )

        # Priors for all the hyper parameters
        self.hp_priors = []
        for _, bounds in enumerate(self.cts_hp_bounds):
            self.hp_priors.append(continuous.ContinuousUniform(bounds[0], bounds[-1]))

        for _, vals in enumerate(self.dscr_hp_vals):
            self.hp_priors.append(discrete.Categorical(vals, np.repeat(1.0 / len(vals), len(vals))))

    def _set_up_post_mean_hp_tune(self):
        """Sets up using the posterior mean for tuning the parameters of the GP."""
        raise NotImplementedError("Not implemented post_mean yet.")

    def build_gp(self, gp_cts_hps, gp_dscr_hps, other_gp_params=None, *args, **kwargs):
        """A method which builds a GP from the given gp_hyperparameters. It calls
        _child_build_gp after running some checks."""
        # pylint: disable=too-many-branches
        # Check the length of the hyper-parameters
        if self.num_hps != len(gp_cts_hps) + len(gp_dscr_hps):
            raise ValueError(
                "gp_hyperparams should be of length %d. Given length: %d."
                % (self.num_hps, len(gp_cts_hps) + len(gp_dscr_hps))
            )
        # Compute the mean and the noise variance first
        # Mean function -------------------------------------
        if hasattr(self.options, "mean_func") and self.options.mean_func is not None:
            mean_func = self.options.mean_func
        else:
            if self.options.mean_func_type == "mean":
                mean_func_const_value = np.mean(self.Y)
            elif self.options.mean_func_type == "median":
                mean_func_const_value = np.median(self.Y)
            elif self.options.mean_func_type == "upper_bound":
                mean_func_const_value = np.mean(self.Y) + 3 * np.std(self.Y)
            elif self.options.mean_func_type == "const":
                mean_func_const_value = self.options.mean_func_const
            elif self.options.mean_func_type == "tune":
                mean_func_const_value = np.asscalar(gp_cts_hps[0])
                gp_cts_hps = gp_cts_hps[1:]
            else:
                mean_func_const_value = 0

            def _get_mean_func(_mean_func_const_value):
                """Returns the mean function from the constant value."""
                return lambda x: np.array([_mean_func_const_value] * len(x))

            mean_func = _get_mean_func(mean_func_const_value)
        # Noise variance ------------------------------------
        if self.options.noise_var_type == "tune":
            noise_var = np.exp(gp_cts_hps[0])
            gp_cts_hps = gp_cts_hps[1:]
        elif self.options.noise_var_type == "label":
            noise_var = self.options.noise_var_label * (self.Y.std() ** 2)
        else:
            noise_var = self.options.noise_var_value
        ret_gp, ret_cts_hps, ret_dscr_hps = self._child_build_gp(
            mean_func,
            noise_var,
            gp_cts_hps,
            gp_dscr_hps,
            other_gp_params=other_gp_params,
            *args,
            **kwargs,
        )
        assert len(ret_cts_hps) == 0
        assert len(ret_dscr_hps) == 0
        return ret_gp

    def _child_build_gp(
        self, mean_func, noise_var, gp_cts_hps, gp_dscr_hps, other_gp_params=None, *args, **kwargs
    ):
        """A method which builds the child GP from the given gp_hyperparameters. Should be
        implemented in a child method."""
        raise NotImplementedError("Implement _child_build_gp in a child method.")

    def _tuning_objective(self, gp_cts_hps, gp_dscr_hps, other_gp_params=None, *args, **kwargs):
        """This function computes the tuning objective (such as the marginal likelihood)
        which is to be maximised in fit_gp."""
        built_gp = self.build_gp(
            gp_cts_hps, gp_dscr_hps, other_gp_params=other_gp_params, *args, **kwargs
        )
        if "ml" in self.methods_to_use or "marginal_likelihood" in self.methods_to_use:
            ret = built_gp.compute_log_marginal_likelihood()
        elif "cv" in self.methods_to_use or "cross_validation" in self.methods_to_use:
            raise NotImplementedError("Yet to implement cross validation based hp-tuning.")
        else:
            raise ValueError("hp_tune_criterion should be either ml or cv")
        return ret

    def _tuning_objective_post_sampling(
        self,
        gp_cts_hps,
        gp_dscr_hps,
        other_gp_params=None,
        param=None,
        param_num=None,
        gradient=False,
        *args,
        **kwargs,
    ):
        """This function computes the tuning objective for posterior sampling case."""
        built_gp = self.build_gp(
            gp_cts_hps, gp_dscr_hps, other_gp_params=other_gp_params, *args, **kwargs
        )
        if gradient:
            ret = built_gp.compute_grad_log_marginal_likelihood(param, param_num)
        else:
            ret = built_gp.compute_log_marginal_likelihood()
        return ret

    def _optimise_cts_hps_for_given_dscr_hps(self, given_dscr_hps):
        """Optimises the continuous hyper-parameters for the given discrete hyper-params.
        Can be overridden by a child class if necessary.
        """
        cts_tuning_obj = lambda arg: self._tuning_objective(arg, list(given_dscr_hps))
        opt_cts_val, opt_cts_hps, _ = self.cts_hp_optimise(cts_tuning_obj, self.hp_tune_max_evals)
        return opt_cts_val, opt_cts_hps, None

    def _sample_cts_dscr_hps_for_rand_exp_sampling(self):
        """Samples continous and discrete hyper-parameters for rand_exp_sampling."""
        sample_cts_hps, sample_dscr_hps, sample_probs = self.hp_sampler(
            self._tuning_objective, self.hp_tune_max_evals
        )
        sample_other_gp_params = [None] * len(sample_cts_hps)
        return sample_cts_hps, sample_dscr_hps, sample_other_gp_params, sample_probs

    def _sample_cts_dscr_hps_for_post_sampling(self, num_samples):
        """Samples continous and discrete hyper-parameters for post_sampling."""

        # pylint: disable=too-many-branches
        # pylint: disable=too-many-locals
        def _logp(x):
            """Computes log probability of observations at x."""
            # sum of log probability priors
            if self.parameter == "additive_grp":
                if x < 0:
                    return -np.inf
                permut = list(np.random.RandomState(seed=x[0]).permutation(self.add_dim))
                groupings = [
                    permut[i : i + self.group_size] for i in range(0, self.add_dim, self.group_size)
                ]
                self.other_gp_params = Namespace(add_gp_groupings=groupings)
            elif type(self.hp_priors[self.curr_hp]).__name__ == "Categorical":
                self.hps[self.curr_hp] = self.hp_priors[self.curr_hp].get_category(np.asscalar(x))
            else:
                self.hps[self.curr_hp] = np.asarray(x).item()

            lp = 0
            for i, prior in enumerate(self.hp_priors):
                if type(self.hp_priors[i]).__name__ == "Categorical":
                    lp += prior.logp(self.hp_priors[i].get_id(self.hps[i]))
                else:
                    lp += prior.logp(self.hps[i])
            if not np.isfinite(lp):
                return lp

            lp += self._tuning_objective_post_sampling(
                self.hps[0 : len(self.cts_hp_bounds)],
                self.hps[len(self.cts_hp_bounds) : self.num_hps],
                self.other_gp_params,
            )
            return lp

        def _pdf(x):
            """Computes probability of observations at x."""
            val = _logp(x)
            if not np.isfinite(val):
                return val
            return np.exp(val)

        def _grad_logp(x):
            """Computes gradient of log probability of observations at x."""
            if not np.isfinite(self.hp_priors[self.curr_hp].logp(x)):
                return 0
            self.hps[self.curr_hp] = np.asarray(x).item()
            lp = self.hp_priors[self.curr_hp].grad_logp(self.hps[self.curr_hp])
            lp += self._tuning_objective_post_sampling(
                self.hps[0 : len(self.cts_hp_bounds)],
                self.hps[len(self.cts_hp_bounds) : self.num_hps],
                self.other_gp_params,
                self.parameter,
                self.param_num,
                True,
            )
            return lp

        offset = self.options.post_hp_tune_offset
        _model = Model(_pdf, _logp, _grad_logp)
        self.hps = np.ones((self.num_hps,))
        best_cts_hps = np.zeros([num_samples, len(self.cts_hp_bounds)])
        best_dscr_hps = np.zeros([num_samples, len(self.dscr_hp_vals)])
        best_other_gp_params = [None] * num_samples
        total_samples = (num_samples - 1) * offset + 1
        cts_hps = np.zeros([total_samples, len(self.cts_hp_bounds)])
        dscr_hps = np.zeros([total_samples, len(self.dscr_hp_vals)])
        _other_gp_params = [None] * total_samples
        self.param_num = -1

        # burn value
        if self.options.post_hp_tune_burn == -1:
            burn = int(np.sqrt(self.num_hps) * 100)
            np.clip(burn, 100, 2000)
        else:
            burn = self.options.post_hp_tune_burn

        # Initializing hyper parameters
        for i in range(len(self.cts_hp_bounds)):
            self.hps[i] = self.hp_priors[i].get_mean()
        for i in range(len(self.cts_hp_bounds), self.num_hps):
            self.hps[i] = np.asarray(self.hp_priors[i].draw_samples("random", 1)).item()
            if type(self.hp_priors[i]).__name__ == "Categorical":
                self.hps[i] = self.hp_priors[i].get_category(int(self.hps[i]))

        # Grouping intitialization for additive gp
        if vars(self.options).get("use_additive_gp", False) or vars(self.options).get(
            "domain_use_additive_gp", False
        ):
            if vars(self.options).get("domain_use_additive_gp", False):
                self.add_dim = self.domain_dim
                add_max_group_size = self.domain_add_max_group_size
            else:
                self.add_dim = self.dim
                add_max_group_size = self.add_max_group_size
            permut = list(np.random.permutation(self.add_dim))
            groupings = [
                permut[i : i + int(self.hps[-1])] for i in range(0, self.add_dim, int(self.hps[-1]))
            ]
            self.other_gp_params = Namespace(add_gp_groupings=groupings)
        else:
            self.other_gp_params = Namespace(add_gp_groupings=None)

        # Randomly selcting a parameter and sampling it using tuning objective
        order = list(range(self.num_hps))
        np.random.shuffle(order)
        for i in order:
            self.curr_hp = i
            self.parameter = self.param_order[i][0]
            if self.parameter == "dim_bandwidths":
                self.param_num = self.param_num + 1
            dscr_i = i - len(self.cts_hp_bounds)
            if self.param_order[i][-1] == "cts":
                cts_hps[:, i] = np.squeeze(
                    self.hp_sampler_cts(_model, self.hps[i], total_samples, burn), axis=1
                )
                self.hps[i] = cts_hps[0, i]
            elif self.parameter != "additive_grp":
                if type(self.hp_priors[i]).__name__ == "Categorical":
                    init_sample = self.hp_priors[i].get_id(self.hps[i])
                    dscr_hp = np.squeeze(
                        self.hp_sampler_dscr(_model, init_sample, total_samples), axis=1
                    )
                    for j, val in enumerate(dscr_hp):
                        dscr_hps[j, dscr_i] = self.hp_priors[i].get_category(int(val))
                else:
                    dscr_hps[:, dscr_i] = np.squeeze(
                        self.hp_sampler_dscr(_model, self.hps[i], total_samples), axis=1
                    )
                self.hps[i] = dscr_hps[0, dscr_i]
            else:
                dscr_hps[:, dscr_i] = np.random.randint(1, add_max_group_size + 1, total_samples)
                seed = int(np.random.randint(self.add_dim))
                for j in range(total_samples):
                    self.group_size = int(dscr_hps[j, dscr_i])
                    seed = int(np.asarray(self.hp_sampler_dscr(_model, seed, 1)).item())
                    permut = list(np.random.RandomState(seed=seed).permutation(self.add_dim))
                    groupings = [
                        permut[k : k + self.group_size]
                        for k in range(0, self.add_dim, self.group_size)
                    ]
                    _other_gp_params[j] = Namespace(add_gp_groupings=groupings)
                self.other_gp_params = _other_gp_params[0]

        for i in range(num_samples):
            best_cts_hps[i, :] = cts_hps[i * offset, :]
            best_dscr_hps[i, :] = dscr_hps[i * offset, :]
            best_other_gp_params[i] = _other_gp_params[i * offset]

        return best_cts_hps, best_dscr_hps, best_other_gp_params

    def get_next_gp(self):
        """Gets the GP to use in the current iteration."""
        if self.options.hp_tune_probs == "adaptive":
            self.hp_tune_probs = self._get_adaptive_hp_tune_probs()
        method = np.random.choice(self.methods_to_use, p=self.hp_tune_probs)
        fit_type = self.hp_tune_results[method][0]
        if fit_type in ["fitted_gp", "post_fitted_gp"]:
            gp = self.hp_tune_results[method][1]
        elif fit_type in ["sample_hps_with_probs", "post_sample_hps_with_probs"]:
            next_gp_hps = self.hp_tune_results[method][1].pop(0)
            self.hp_tune_results[method][1].append(next_gp_hps)
            gp = self.build_gp(
                next_gp_hps[0],
                next_gp_hps[1],
                other_gp_params=next_gp_hps[2],
                build_posterior=False,
            )
        return (fit_type, method, gp)

    def update_hp_tune_method_weight(self, method, weight_to_add=1):
        """Updates hp_tune method weight in the case of adaptive tuning."""
        if self.options.hp_tune_probs == "adaptive":
            self.hp_tune_sampling_weights[method] += weight_to_add

    def fit_gp_for_gp_bandit(self, num_samples=1):
        """Fits a GP according to the tuning criterions."""
        self.hp_tune_results = {}
        for method in self.methods_to_use:
            ret = self.fit_gp(num_samples, method)
            if ret[0] == "fitted_gp":
                self.hp_tune_results[method] = (ret[0], ret[1])
            elif ret[0] == "post_fitted_gp":
                self.hp_tune_results[method] = (ret[0], ret[1])
            elif ret[0] == "sample_hps_with_probs":
                # in this case, ret is a 5-tuple where ret[0] is sample_hps_with_probs, ret[1] is
                # a list of continuous hps, ret[2] is a list of discrete hps, ret[3] is either
                # None or a namespace containing the field add_gp_groupings.
                # See gp.gp_core.fit_gp
                sample_hps = list(zip(ret[1], ret[2], ret[3]))
                sample_probs = ret[-1]
                # Sample the hyper_parameters we will use until the next time we build a model.
                if sum(sample_probs > 0) >= num_samples:
                    to_replace = self.options.rand_exp_sampling_replace
                else:
                    to_replace = True
                use_hps_idxs = np.random.choice(
                    len(sample_hps), size=(num_samples,), replace=to_replace, p=sample_probs
                )
                self.hp_tune_results[method] = (ret[0], [sample_hps[idx] for idx in use_hps_idxs])
            elif ret[0] == "post_sample_hps_with_probs":
                # in this case, ret is a 4-tuple where ret[0] is post_sample_hps_with_probs,
                # ret[1] is a list of continuous hps, ret[2] is a list of discrete hps, ret[3]
                # is either None or a namespace containing the field add_gp_groupings.
                # See gp.gp_core.fit_gp
                self.hp_tune_results[method] = (ret[0], list(zip(ret[1], ret[2], ret[3])))
            else:
                raise ValueError("Unknown option %s for results of fit_gp." % (ret[0]))

    def fit_gp(self, num_samples=1, hp_tune_criterion=None):
        """Fits a GP according to the tuning criterion. Returns the best GP along with the
        hyper-parameters."""
        if hp_tune_criterion is None:
            hp_tune_criterion = self.options.hp_tune_criterion
        if hp_tune_criterion == "ml":
            if self.ml_hp_tune_opt_method in ["direct", "rand", "pdoo"]:
                best_cts_hps = None
                best_dscr_hps = None
                best_other_params = None
                best_hps_val = -np.inf
                for dscr_hps in itertools_product(*self.dscr_hp_vals):
                    opt_cts_val, opt_cts_hps, opt_other_params = (
                        self._optimise_cts_hps_for_given_dscr_hps(dscr_hps)
                    )
                    if opt_cts_val > best_hps_val:
                        best_cts_hps = list(opt_cts_hps)
                        best_dscr_hps = list(dscr_hps)
                        best_other_params = opt_other_params
                        best_hps_val = opt_cts_val
                opt_gp = self.build_gp(
                    best_cts_hps, best_dscr_hps, other_gp_params=best_other_params
                )
                opt_hps = (best_cts_hps, best_dscr_hps)
                return "fitted_gp", opt_gp, opt_hps
            elif self.ml_hp_tune_opt_method == "rand_exp_sampling":
                sample_cts_hps, sample_dscr_hps, sample_other_gp_params, sample_probs = (
                    self._sample_cts_dscr_hps_for_rand_exp_sampling()
                )
                return (
                    "sample_hps_with_probs",
                    sample_cts_hps,
                    sample_dscr_hps,
                    sample_other_gp_params,
                    sample_probs,
                )
        elif hp_tune_criterion == "post_sampling":
            sample_cts_hps, sample_dscr_hps, sample_other_gp_params = (
                self._sample_cts_dscr_hps_for_post_sampling(num_samples)
            )
            if num_samples == 1:
                opt_gp = self.build_gp(
                    sample_cts_hps[0], sample_dscr_hps[0], other_gp_params=sample_other_gp_params[0]
                )
                opt_hps = (sample_cts_hps, sample_dscr_hps)
                return "post_fitted_gp", opt_gp, opt_hps
            else:
                return (
                    "post_sample_hps_with_probs",
                    sample_cts_hps,
                    sample_dscr_hps,
                    sample_other_gp_params,
                )


def _get_cholesky_decomp(K_trtr_wo_noise, noise_var, handle_non_psd_kernels):
    """Computes cholesky decomposition after checking how to handle non-psd kernels."""
    if handle_non_psd_kernels == "try_before_project":
        K_trtr_w_noise = K_trtr_wo_noise + noise_var * np.eye(K_trtr_wo_noise.shape[0])
        try:
            # If the cholesky decomposition on the (noise added) matrix succeeds, return!
            L = stable_cholesky(K_trtr_w_noise, add_to_diag_till_psd=False)
            return L
        except np.linalg.LinAlgError:
            # otherwise, project and return
            return _get_cholesky_decomp(K_trtr_wo_noise, noise_var, "project_first")
    elif handle_non_psd_kernels == "project_first":
        # project the Kernel (without noise) to the PSD cone and return
        K_trtr_wo_noise = project_symmetric_to_psd_cone(K_trtr_wo_noise)
        return _get_cholesky_decomp(K_trtr_wo_noise, noise_var, "guaranteed_psd")
    elif handle_non_psd_kernels == "guaranteed_psd":
        K_trtr_w_noise = K_trtr_wo_noise + noise_var * np.eye(K_trtr_wo_noise.shape[0])
        return stable_cholesky(K_trtr_w_noise)
    else:
        raise ValueError("Unknown option for handle_non_psd_kernels: %s" % (handle_non_psd_kernels))


def get_post_covar_from_raw_covar(raw_post_covar, noise_var, is_guaranteed_psd):
    """Computes the posterior covariance from the raw_post_covar. This is mostly to
    account for the fact that the kernel may not be psd.
    """
    if is_guaranteed_psd:
        return raw_post_covar
    else:
        epsilon = 0.05 * noise_var
        return project_symmetric_to_psd_cone(raw_post_covar, epsilon=epsilon)


gp_core = Namespace(GP=GP, GPFitter=GPFitter, mandatory_gp_args=mandatory_gp_args)


# Source: gp/euclidean_gp.py SHA256 19fd110faa398cbc59136a93a1ac785f61fb0bc3177261e20b012577a0cd68d6

_DFLT_KERNEL_TYPE = "matern"

basic_euc_gp_args = [
    get_option_specs(
        "kernel_type", False, "default", "Specify type of kernel. This depends on the application."
    ),
    get_option_specs(
        "use_same_bandwidth",
        False,
        False,
        (
            "If true, will use same bandwidth on all dimensions. Applies only "
            "when kernel_type is se or matern. Default=False."
        ),
    ),
]

se_gp_args: list[dict[str, Any]] = []

matern_gp_args = [
    get_option_specs(
        "matern_nu",
        False,
        2.5,
        ("Specify the nu value for the matern kernel. If negative, will fit."),
    ),
]

poly_gp_args = [
    get_option_specs(
        "use_same_scalings",
        False,
        False,
        "If true uses same scalings on all dimensions. Default is False.",
    ),
    get_option_specs(
        "poly_order", False, 1, "Order of the polynomial to be used. Default is 1 (linear kernel)."
    ),
]

add_gp_args = [
    get_option_specs("use_additive_gp", False, False, "Whether or not to use an additive GP. "),
    get_option_specs(
        "add_max_group_size", False, 6, "The maximum number of groups in the additive grouping. "
    ),
    get_option_specs(
        "add_grouping_criterion",
        False,
        "randomised_ml",
        "Specify the grouping algorithm, should be one of {randomised_ml}",
    ),
    get_option_specs(
        "num_groups_per_group_size", False, -1, "The number of groups to try per group size."
    ),
    get_option_specs(
        "add_group_size_criterion",
        False,
        "sampled",
        "Specify how to pick the group size, should be one of {max, sampled}.",
    ),
]

esp_gp_args = [
    get_option_specs("esp_order", False, -1, "Order of the esp kernel. "),
    get_option_specs(
        "esp_kernel_type", False, "se", "Specify type of kernel. This depends on the application."
    ),
    get_option_specs(
        "esp_matern_nu",
        False,
        -1.0,
        ("Specify the nu value for matern kernel. If negative, will fit."),
    ),
]

euclidean_gp_args = (
    gp_core.mandatory_gp_args
    + basic_euc_gp_args
    + se_gp_args
    + matern_gp_args
    + poly_gp_args
    + add_gp_args
    + esp_gp_args
)


class EuclideanGP(GP):
    """euclidean GP factory"""

    # pylint: disable=attribute-defined-outside-init

    def __init__(
        self,
        X,
        Y,
        kernel,
        mean_func,
        noise_var,
        kernel_hyperparams=None,
        build_posterior=True,
        reporter=None,
    ):
        """
        X, Y: data
        kern: could be an object or one of the following strings, 'se', 'poly', 'matern'
        kernel_hyperparams: dictionary specifying the hyper-parameters for the kernel.
              'se' : 'dim', 'scale' (optional), 'dim_bandwidths' (optional)
              'poly' : 'dim', 'order', 'scale', 'dim_scalings' (optional)
              'matern' : 'dim', 'nu' (optional), 'scale' (optional),
                         'dim_bandwidths' (optional)
        """
        if isinstance(kernel, str):
            kernel = self._get_kernel_from_type(kernel, kernel_hyperparams)
        super(EuclideanGP, self).__init__(
            X, Y, kernel, mean_func, noise_var, build_posterior, reporter
        )

    @classmethod
    def _get_kernel_from_type(cls, kernel_type, kernel_hyperparams):
        """Get different euclidean kernels based on kernel_type"""
        if kernel_type in ["se"]:
            return gp_kernel.SEKernel(
                kernel_hyperparams["dim"],
                kernel_hyperparams["scale"],
                kernel_hyperparams["dim_bandwidths"],
            )
        elif kernel_type in ["poly"]:
            return gp_kernel.PolyKernel(
                kernel_hyperparams["dim"],
                kernel_hyperparams["order"],
                kernel_hyperparams["scale"],
                kernel_hyperparams["dim_scalings"],
            )
        elif kernel_type in ["matern"]:
            return gp_kernel.MaternKernel(
                kernel_hyperparams["dim"],
                kernel_hyperparams["nu"],
                kernel_hyperparams["scale"],
                kernel_hyperparams["dim_bandwidths"],
            )
        elif kernel_type in ["esp"]:
            return gp_kernel.ESPKernelSE(
                kernel_hyperparams["dim"],
                kernel_hyperparams["scale"],
                kernel_hyperparams["order"],
                kernel_hyperparams["dim_bandwidths"],
            )
        else:
            raise ValueError("Cannot construct kernel from kernel_type %s." % (kernel_type))

    def _child_str(self):
        """String representation for child GP."""
        ke_str = self._get_kernel_str(self.kernel)
        dim = 0 if len(self.X) == 0 else len(self.X[0])
        mean_str = (
            "mu(0)=%0.3f"
            % (
                self.mean_func(
                    [
                        np.zeros(
                            dim,
                        )
                    ]
                )[0]
            )
        )
        ret = "scale: %0.3f, %s, %s" % (self.kernel.hyperparams["scale"], ke_str, mean_str)
        return ret

    @classmethod
    def _get_kernel_str(cls, kern):
        """Gets a string format of the kernel depending on whether it is SE/Poly."""
        if isinstance(kern, gp_kernel.AdditiveKernel):
            return str(kern)
        if isinstance(kern, gp_kernel.SEKernel) or isinstance(kern, gp_kernel.MaternKernel):
            hp_name = "dim_bandwidths"
            kern_name = (
                "se"
                if isinstance(kern, gp_kernel.SEKernel)
                else "matern(%0.1f)" % (kern.hyperparams["nu"])
            )
        elif isinstance(kern, gp_kernel.PolyKernel):
            hp_name = "dim_scalings"
            kern_name = "poly"
        else:  # Return an empty string.
            return ""
        if kern.dim > 6:
            ret = "%0.4f(avg)" % (kern.hyperparams[hp_name].mean())
        else:
            ret = get_list_of_floats_as_str(kern.hyperparams[hp_name])
        ret = kern_name + "-" + ret
        return ret


class EuclideanGPFitter(GPFitter):
    """Fits a GP by tuning the kernel hyper-params."""

    # pylint: disable=attribute-defined-outside-init

    def __init__(self, X, Y, options=None, reporter=None):
        """Constructor."""
        self.dim = len(X[0])
        reporter = get_reporter(reporter)
        options = load_options(euclidean_gp_args, partial_options=options)
        super(EuclideanGPFitter, self).__init__(X, Y, options, reporter)

    def _child_set_up(self):
        """Sets parameters for GPFitter."""
        # IMPORTANT: Keep this order when tuning for the hyper-parameters.
        # Continuous: Mean value, GP noise, kernel scale, then the remaining Kernel params.
        # Discrete: add_group_size, then the remaining kernel params.
        # Check args - so that we don't have to keep doing this all the time
        if self.options.kernel_type not in ["se", "matern", "poly", "esp", "default"]:
            raise ValueError("Unknown kernel_type. Should be either se, matern or poly.")
        if self.options.noise_var_type not in ["tune", "label", "value"]:
            raise ValueError("Unknown noise_var_type. Should be either tune, label or value.")
        if self.options.mean_func_type not in ["mean", "median", "const", "zero", "tune"]:
            raise ValueError("Unknown mean_func_type. Should be mean/median/const/zero/tune.")
        # Set kernel type
        if self.options.kernel_type == "default":
            self.kernel_type = _DFLT_KERNEL_TYPE
        else:
            self.kernel_type = self.options.kernel_type
        # 1 & 2: mean value and noise variance - done in parent class.
        # 3. Kernel parameters
        if self.kernel_type == "se":
            self._se_kernel_set_up()
        elif self.kernel_type == "matern":
            self._matern_kernel_set_up()
        elif self.kernel_type in ["poly"]:
            self._poly_kernel_set_up()
        elif self.kernel_type == "esp":
            self._esp_kernel_set_up()
        # 4. Additive grouping
        if self.options.use_additive_gp and self.kernel_type != "esp":
            self.add_group_size_idx_in_dscr_hp_vals = len(self.dscr_hp_vals)
            self.add_max_group_size = min(self.options.add_max_group_size, self.dim)
            self.dscr_hp_vals.append([x + 1 for x in range(self.add_max_group_size)])
            self.param_order.append(["additive_grp", "dscr"])
        elif self.kernel_type == "esp" and self.options.esp_order == -1:
            self.dscr_hp_vals.append(list(range(1, max(self.dim, self.options.esp_order) + 1)))
            self.param_order.append(["esp_order", "dscr"])

    def _se_kernel_set_up(self):
        """Set up for the SE kernel."""
        # Scale
        self.scale_log_bounds = [np.log(0.1 * self.Y_var), np.log(10 * self.Y_var)]
        self.param_order.append(["scale", "cts"])
        # Bandwidths
        X_std_norm = np.linalg.norm(self.X, "fro") + 1e-4
        single_bandwidth_log_bounds = [np.log(0.01 * X_std_norm), np.log(10 * X_std_norm)]
        if self.options.use_same_bandwidth:
            self.bandwidth_log_bounds = [single_bandwidth_log_bounds]
            self.param_order.append(["same_dim_bandwidths", "cts"])
        else:
            self.bandwidth_log_bounds = [single_bandwidth_log_bounds] * self.dim
            for _ in range(self.dim):
                self.param_order.append(["dim_bandwidths", "cts"])
        self.cts_hp_bounds += [self.scale_log_bounds] + self.bandwidth_log_bounds

    def _matern_kernel_set_up(self):
        """Set up for the Matern kernel."""
        # Set up scale and bandwidth - which is identical to the SE kernel.
        self._se_kernel_set_up()
        # Set up optimisation values for the nu parameter.
        if self.options.matern_nu < 0:
            self.dscr_hp_vals.append([0.5, 1.5, 2.5])
            self.param_order.append(["nu", "dscr"])

    def _poly_kernel_set_up(self):
        """Set up for the Poly kernel."""
        raise NotImplementedError("Not implemented Poly kernel yet.")

    def _esp_kernel_set_up(self):
        """Set up for the ESP kernel."""
        if self.options.esp_kernel_type not in ["se", "matern"]:
            raise NotImplementedError("Not implemented yet.")
        # Scale
        self.scale_log_bounds = [np.log(0.1 * self.Y_var), np.log(10 * self.Y_var)]
        self.param_order.append(["scale", "cts"])
        # Bandwidths
        X_std_norm = np.linalg.norm(self.X, "fro") + 1e-4
        single_bandwidth_log_bounds = [np.log(0.01 * X_std_norm), np.log(10 * X_std_norm)]
        self.bandwidth_log_bounds = [single_bandwidth_log_bounds] * self.dim
        for _ in range(self.dim):
            self.param_order.append(["dim_bandwidths", "cts"])
        self.cts_hp_bounds += [self.scale_log_bounds] + self.bandwidth_log_bounds
        if self.options.esp_kernel_type == "matern" and self.options.esp_matern_nu < 0:
            self.dscr_hp_vals.append([0.5, 1.5, 2.5])
            self.param_order.append(["nu", "dscr"])

    def _prep_init_kernel_hyperparams(self, kernel_type):
        """Wrapper to pack the kernel hyper-parameters into a dictionary."""
        return prep_euclidean_integral_kernel_hyperparams(kernel_type, self.options, self.dim)

    def _optimise_cts_hps_for_given_dscr_hps(self, given_dscr_hps):
        """Optimises the continuous hyper-parameters for the given discrete hyper-params."""
        if not self.options.use_additive_gp:
            return super(EuclideanGPFitter, self)._optimise_cts_hps_for_given_dscr_hps(
                given_dscr_hps
            )
        else:
            return optimise_cts_hps_for_given_dscr_hps_in_add_model(
                given_dscr_hps,
                self.options.num_groups_per_group_size,
                self.dim,
                self.hp_tune_max_evals,
                self.cts_hp_optimise,
                self._tuning_objective,
            )

    def _sample_cts_dscr_hps_for_rand_exp_sampling(self):
        """Samples continous and discrete hyper-parameters for rand_exp_sampling."""
        if not self.options.use_additive_gp:
            return super(EuclideanGPFitter, self)._sample_cts_dscr_hps_for_rand_exp_sampling()
        else:
            return sample_cts_dscr_hps_for_rand_exp_sampling_in_add_model(
                self.hp_tune_max_evals,
                self.cts_hp_bounds,
                self.dim,
                self.dscr_hp_vals,
                self.add_group_size_idx_in_dscr_hp_vals,
                self._tuning_objective,
            )

    def _child_build_gp(
        self, mean_func, noise_var, gp_cts_hps, gp_dscr_hps, other_gp_params=None, *args, **kwargs
    ):
        """Builds the GP."""
        # Domain kernel --------------------------------------
        kernel_hyperparams = self._prep_init_kernel_hyperparams(self.kernel_type)
        add_gp_groupings = None
        if self.options.use_additive_gp:
            gp_dscr_hps = gp_dscr_hps[:-1]  # The first element is the group size
            add_gp_groupings = other_gp_params.add_gp_groupings
        kernel, gp_cts_hps, gp_dscr_hps = get_euclidean_integral_gp_kernel(
            self.kernel_type,
            kernel_hyperparams,
            gp_cts_hps,
            gp_dscr_hps,
            self.options.use_same_bandwidth,
            add_gp_groupings,
            self.options.esp_kernel_type,
        )
        ret_gp = EuclideanGP(self.X, self.Y, kernel, mean_func, noise_var, *args, **kwargs)
        return ret_gp, gp_cts_hps, gp_dscr_hps


def optimise_cts_hps_for_given_dscr_hps_in_add_model(
    given_dscr_hps,
    num_groups_per_group_size,
    dim,
    hp_tune_max_evals,
    cts_hp_optimise,
    tuning_objective,
):
    """Optimises the continuous hyper-parameters for an additive model."""
    group_size = given_dscr_hps[-1]  # The first is the max group size
    if num_groups_per_group_size < 0:
        if group_size == 1:
            num_groups_per_group_size = 1
        else:
            num_groups_per_group_size = max(5, min(2 * dim, 25))
    grp_best_hps = None
    grp_best_val = -np.inf
    grp_best_other_params = None
    # Now try out different groups picking a random grouping each time.
    for _ in range(num_groups_per_group_size):
        rand_perm = list(np.random.permutation(dim))
        groupings = [rand_perm[i : i + group_size] for i in range(0, dim, group_size)]
        other_gp_params = Namespace(add_gp_groupings=groupings)
        # _tuning_objective is usually defined in gp_core.py
        cts_tuning_objective = lambda arg: tuning_objective(
            arg, given_dscr_hps[:], other_gp_params=other_gp_params
        )
        max_evals = int(max(500, hp_tune_max_evals / num_groups_per_group_size))
        opt_cts_val, opt_cts_hps, _ = cts_hp_optimise(cts_tuning_objective, max_evals)
        if opt_cts_val > grp_best_val:
            grp_best_val = opt_cts_val
            grp_best_hps = opt_cts_hps
            grp_best_other_params = other_gp_params
    return grp_best_val, grp_best_hps, grp_best_other_params


def sample_cts_dscr_hps_for_rand_exp_sampling_in_add_model(
    num_evals,
    cts_hp_bounds,
    dim,
    dscr_hp_vals,
    add_group_size_idx_in_dscr_hp_vals,
    tuning_objective,
):
    # IMPORTANT: We are assuming that the add_group_size is the first
    """Samples the hyper-paramers for an additive model."""
    agsidhp = add_group_size_idx_in_dscr_hp_vals
    sample_cts_hps = []
    sample_dscr_hps = []
    sample_other_gp_params = []
    sample_obj_vals = []
    for _ in range(num_evals):
        group_size = np.random.choice(dscr_hp_vals[agsidhp])
        rand_perm = list(np.random.permutation(dim))
        groupings = [rand_perm[i : i + group_size] for i in range(0, dim, group_size)]
        curr_other_gp_params = Namespace(add_gp_groupings=groupings)
        curr_dscr_hps = random_sample_from_discrete_domain(dscr_hp_vals)
        curr_dscr_hps[agsidhp] = group_size
        curr_cts_hps = map_to_bounds(np.random.random((len(cts_hp_bounds),)), cts_hp_bounds)
        curr_obj_val = tuning_objective(curr_cts_hps, curr_dscr_hps, curr_other_gp_params)
        # Now add to the lists
        sample_cts_hps.append(curr_cts_hps)
        sample_dscr_hps.append(curr_dscr_hps)
        sample_other_gp_params.append(curr_other_gp_params)
        sample_obj_vals.append(curr_obj_val)
    sample_probs = np.exp(sample_obj_vals)
    sample_probs = sample_probs / sample_probs.sum()
    return sample_cts_hps, sample_dscr_hps, sample_other_gp_params, sample_probs


def prep_euclidean_integral_kernel_hyperparams(kernel_type, gp_fitter_options, domain_dim):
    """Wrapper to pack the kernel hyper-parameters into a dictionary."""
    hyperparams = {}
    hyperparams["dim"] = domain_dim
    if kernel_type == "matern" and gp_fitter_options.matern_nu > 0:
        hyperparams["nu"] = gp_fitter_options.matern_nu
    elif kernel_type == "poly":
        hyperparams["order"] = gp_fitter_options.poly_order
    elif kernel_type == "esp":
        if gp_fitter_options.esp_order > 0:
            hyperparams["esp_order"] = gp_fitter_options.esp_order
        if gp_fitter_options.esp_matern_nu > 0:
            hyperparams["esp_matern_nu"] = gp_fitter_options.esp_matern_nu
    return hyperparams


def get_euclidean_integral_gp_kernel(
    kernel_type,
    kernel_hyperparams,
    gp_cts_hps,
    gp_dscr_hps,
    use_same_bandwidth,
    add_gp_groupings=None,
    esp_kernel_type=None,
):
    """A method to parse the GP kernel from hyper-parameters. Assumes the scale is
    part of the hyper-parameters as well."""
    scale = np.exp(gp_cts_hps[0])  # First the scale and then the rest of the hyper-params.
    gp_cts_hps = gp_cts_hps[1:]
    return get_euclidean_integral_gp_kernel_with_scale(
        kernel_type,
        scale,
        kernel_hyperparams,
        gp_cts_hps,
        gp_dscr_hps,
        use_same_bandwidth,
        add_gp_groupings,
        esp_kernel_type,
    )


def get_euclidean_integral_gp_kernel_with_scale(
    kernel_type,
    scale,
    kernel_hyperparams,
    gp_cts_hps,
    gp_dscr_hps,
    use_same_bandwidth,
    add_gp_groupings=None,
    esp_kernel_type=None,
):
    """A method to parse the GP kernel from hyper-parameters. Assumes the scale is
    not part of the hyper-parameters but given as an argument."""
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    dim = kernel_hyperparams["dim"]
    # For the additive ones ---------------------
    esp_order = None
    if kernel_type == "esp":
        if "esp_order" in kernel_hyperparams:
            esp_order = kernel_hyperparams["esp_order"]
        else:
            esp_order = gp_dscr_hps[-1]
            gp_dscr_hps = gp_dscr_hps[:-1]
    is_additive = False
    if add_gp_groupings is None:
        add_gp_groupings = [list(range(dim))]
        grp_scale = scale
    elif esp_order is None:
        is_additive = True
        grp_scale = 1.0
    # Extract features and create a placeholder for the kernel constructors.
    # Some common functionality for the se, matern and poly kernels.
    if kernel_type in ["se", "matern", "poly"]:
        if use_same_bandwidth:
            ke_dim_bandwidths = [np.exp(gp_cts_hps[0])] * dim
            gp_cts_hps = gp_cts_hps[1:]
        else:
            ke_dim_bandwidths = np.exp(gp_cts_hps[0:dim])
            gp_cts_hps = gp_cts_hps[dim:]
    elif kernel_type in ["esp"]:
        if use_same_bandwidth:
            esp_dim_bandwidths = np.exp(gp_cts_hps[0:dim])
            gp_cts_hps = gp_cts_hps[dim:]
        else:
            esp_dim_bandwidths = np.exp(gp_cts_hps[0 : dim * dim])
            gp_cts_hps = gp_cts_hps[dim * dim :]
    # Now iterate through the kernels for the rest of the parameters.
    if kernel_type == "se":
        grp_kernels = [
            gp_kernel.SEKernel(
                dim=len(grp),
                scale=grp_scale,
                dim_bandwidths=get_sublist_from_indices(ke_dim_bandwidths, grp),
            )
            for grp in add_gp_groupings
        ]
    elif kernel_type == "matern":
        if "nu" not in kernel_hyperparams or kernel_hyperparams["nu"] < 0:
            matern_nu = gp_dscr_hps[0]
            gp_dscr_hps = gp_dscr_hps[1:]
        else:
            matern_nu = kernel_hyperparams["nu"]
        grp_kernels = [
            gp_kernel.MaternKernel(
                dim=len(grp),
                nu=matern_nu,
                scale=grp_scale,
                dim_bandwidths=get_sublist_from_indices(ke_dim_bandwidths, grp),
            )
            for grp in add_gp_groupings
        ]
    elif kernel_type == "poly":
        if "order" not in kernel_hyperparams or kernel_hyperparams["order"] < 0:
            poly_order = gp_dscr_hps[0]
            gp_dscr_hps = gp_dscr_hps[1:]
        else:
            poly_order = kernel_hyperparams["order"]
        grp_kernels = [
            gp_kernel.PolyKernel(
                dim=len(grp),
                order=poly_order,
                scale=grp_scale,
                dim_scalings=get_sublist_from_indices(ke_dim_bandwidths, grp),
            )
            for grp in add_gp_groupings
        ]
    elif kernel_type == "expdecay":
        exp_decay_offset = np.exp(gp_cts_hps[0])
        exp_decay_powers = np.exp(gp_cts_hps[1 : dim + 1])
        gp_cts_hps = gp_cts_hps[dim + 1 :]
        grp_kernels = [
            gp_kernel.ExpDecayKernel(
                dim=len(grp), scale=grp_scale, offset=exp_decay_offset, powers=exp_decay_powers
            )
            for grp in add_gp_groupings
        ]
    elif kernel_type == "esp":
        if not np.isscalar(esp_order):
            esp_order = np.asscalar(esp_order)
        if esp_kernel_type == "se":
            grp_kernels = [
                gp_kernel.ESPKernelSE(
                    dim=dim, scale=scale, order=int(esp_order), dim_bandwidths=esp_dim_bandwidths
                )
            ]
        elif esp_kernel_type == "matern":
            if "esp_matern_nu" not in kernel_hyperparams:
                matern_nu = [gp_dscr_hps[0]] * dim
                gp_dscr_hps = gp_dscr_hps[1:]
            else:
                matern_nu = [kernel_hyperparams["esp_matern_nu"]] * dim
            grp_kernels = [
                gp_kernel.ESPKernelMatern(
                    dim=dim,
                    nu=matern_nu,
                    scale=scale,
                    order=int(esp_order),
                    dim_bandwidths=esp_dim_bandwidths,
                )
            ]
    else:
        raise Exception("Unknown kernel type %s!" % (kernel_type))
    if is_additive:
        euc_kernel = gp_kernel.AdditiveKernel(
            scale=scale, kernel_list=grp_kernels, groupings=add_gp_groupings
        )
    else:
        euc_kernel = grp_kernels[0]
    return euc_kernel, gp_cts_hps, gp_dscr_hps


# Exact, bounded caches belong to each learner, not process-global state.
_uncached_additive = AdditiveKernel._child_evaluate
_uncached_scalar = MaternKernel._eval_kernel_values_unnormalised


def make_exact_kernel_cache() -> Namespace:
    """Memoize unchanged source components without changing arithmetic or RNG.

    Owned by one Basat learner in a context-local numerical scope. Hyperparameter
    samplers frequently change one coordinate while recomputing every additive
    component. Exact input/parameter byte keys avoid this redundant work; original
    left-to-right summation and uncached matrix ownership are preserved. Payload
    accounting deliberately charges shared data bytes to *every* component entry,
    so the stated memory limit is conservative (bounded Python metadata is extra).
    """

    original = _uncached_additive
    original_scalar = _uncached_scalar
    values: OrderedDict[object, tuple[object, int]] = OrderedDict()
    datasets: OrderedDict[object, int] = OrderedDict()
    constants: OrderedDict[object, object] = OrderedDict()
    audited_error_policy = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
    limits = {
        "component_entries": 1024,
        "component_payload_bytes": 32 * 1024 * 1024,
        "dataset_entries": 16,
        "dataset_payload_bytes": 4 * 1024 * 1024,
        "scalar_entries": 64,
        "matrix_elements": 2500,
    }
    statistics: dict = {
        "implementation": "exact-additive-component-lru-v1",
        "limits": limits,
        "numpy_error_policy": audited_error_policy,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "bypassed": 0,
        "component_payload_bytes": 0,
        "dataset_payload_bytes": 0,
    }

    def fingerprint(value: Any) -> tuple[Any, ...]:
        array = np.asarray(value)
        return array.dtype.str, array.shape, array.tobytes()

    def scalar(kernel: Any, distance: Any) -> Any:
        if isinstance(distance, np.ndarray) or distance != 0 or np.geterr() != audited_error_policy:
            return original_scalar(kernel, distance)
        key = (kernel.p, fingerprint(kernel.hyperparams["nu"]), type(distance))
        if key in constants:
            constants.move_to_end(key)
            return constants[key]
        result = original_scalar(kernel, distance)
        if np.isfinite(result):
            constants[key] = result
            if len(constants) > limits["scalar_entries"]:
                constants.popitem(last=False)
        return result

    def cached(additive: Any, x1: Any, x2: Any) -> Any:
        if (
            len(x1) * len(x2) > limits["matrix_elements"]
            or np.geterr() != audited_error_policy
            or any(type(kernel) is not MaternKernel for kernel in additive.kernel_list)
        ):
            statistics["bypassed"] += 1
            return original(additive, x1, x2)
        # Keep the source's array conversion and floating-point summation order.
        x1, x2 = np.array(x1), np.array(x2)
        data_bytes = x1.nbytes + x2.nbytes
        if data_bytes > limits["dataset_payload_bytes"]:
            statistics["bypassed"] += 1
            return original(additive, x1, x2)
        candidate = (fingerprint(x1), fingerprint(x2))
        if candidate in datasets:
            datasets.move_to_end(candidate)
            # Reuse the canonical key's bytes instead of retaining new copies.
            data = next(reversed(datasets))
        else:
            data = candidate
            datasets[data] = data_bytes
            statistics["dataset_payload_bytes"] += data_bytes
            while (
                len(datasets) > limits["dataset_entries"]
                or statistics["dataset_payload_bytes"] > limits["dataset_payload_bytes"]
            ):
                _, evicted_bytes = datasets.popitem(last=False)
                statistics["dataset_payload_bytes"] -= evicted_bytes
        result = np.zeros((len(x1), len(x2)))
        for kernel, group in zip(additive.kernel_list, additive.groupings, strict=True):
            parameters = (
                fingerprint(kernel.norm_constant),
                fingerprint(kernel.hyperparams["nu"]),
                fingerprint(kernel.hyperparams["scale"]),
                fingerprint(kernel.hyperparams["dim_bandwidths"]),
            )
            key = (data, tuple(group), kernel.dim, kernel.p, *parameters)
            if key in values:
                statistics["hits"] += 1
                values.move_to_end(key)
                component, _ = values[key]
            else:
                component = kernel(x1[:, group], x2[:, group])
                statistics["misses"] += 1
                entry_bytes = (
                    data_bytes
                    + component.nbytes
                    + sum(len(parameter[-1]) for parameter in parameters)
                )
                # Do not cache invalid computations: repeated numerical warnings
                # must remain visible through the source-warning translator.
                if (
                    np.isfinite(component).all()
                    and entry_bytes <= limits["component_payload_bytes"]
                ):
                    values[key] = (component.copy(), entry_bytes)
                    statistics["component_payload_bytes"] += entry_bytes
                    while (
                        len(values) > limits["component_entries"]
                        or statistics["component_payload_bytes"] > limits["component_payload_bytes"]
                    ):
                        _, (_, evicted_bytes) = values.popitem(last=False)
                        statistics["component_payload_bytes"] -= evicted_bytes
                        statistics["evictions"] += 1
            # Cached component matrices never escape or become mutation targets.
            result += component
        return additive.hyperparams["scale"] * result

    return Namespace(additive=cached, scalar=scalar, statistics=statistics)


def _cached_additive_dispatch(kernel, x1, x2):
    cache = _active_cache.get()
    return _uncached_additive(kernel, x1, x2) if cache is None else cache.additive(kernel, x1, x2)


def _cached_scalar_dispatch(kernel, distance):
    cache = _active_cache.get()
    return _uncached_scalar(kernel, distance) if cache is None else cache.scalar(kernel, distance)


AdditiveKernel._child_evaluate = _cached_additive_dispatch  # type: ignore[method-assign]
MaternKernel._eval_kernel_values_unnormalised = _cached_scalar_dispatch  # type: ignore[method-assign]
