# -*- coding: utf-8 -*-
from collections.abc import Callable
from typing import Tuple
from typing_extensions import Self

from jax import numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike
from scipy.special import roots_legendre


class Basis:
    """Polynomial basis functions.

    Parameters
    ----------
    xi : array_like
        (`n_nodes`,) array of node coordinates along the interval
        ``[-1, 1]``.
    order : int
        Order of the numerical quadratures over any interval ``[a, b]``.
    order_nodes : int
        Order of the numerical quadratures over the full interval
        ``[-1, 1]`` that use the node coordinates `xi` as stop points.
    w : array_like or None, default: None
        Quadrature weights for numerical integration using `xi` as
        quadrature points. If `w` is ``None``, the weights are evaluated
        through numerical integration of order `order`.

    Attributes
    ----------
    n_nodes : int
        Number of nodes.

    """

    def __init__(self, xi: ArrayLike, order: int = 101, order_nodes: int = 21, w: ArrayLike | None = None):
        # Runtime type validation
        if not isinstance(xi, ArrayLike):
            raise TypeError(f"Expected arraylike input; got {xi}")
        # Convert input to jax.Array
        xi = jnp.asarray(xi)

        # --- Class atributes ---
        self.xi = xi
        self.n_nodes = len(xi)
        self.order = order
        self.order_nodes = order_nodes

        # --- Basis functions (psi) ---
        self._psi_coefs = jnp.polyfit(
            self.xi,
            jnp.eye(self.n_nodes),
            self.n_nodes - 1)

        # --- Initialize quadrature methods ---
        self._initialize_quad_methods(order, order_nodes)
        # Integration weights
        if w is None:
            w = self.f_psi(self._x_gl).T @ self._w_gl
        self.w = w

    def f(self, xi: float | Array, f_nodes: Array) -> float | Array:
        """Value at coordinate from values at nodes.

        Parameters
        ----------
        xi : float or array
            Coordinate or (M,) array of coordinates.
        f_nodes : array
            (`n_nodes`,) array of values at nodes.

        Returns
        -------
        float or array
            Value or (M,) array of values at requested coordinates
            evaluated using polynomial basis functions.

        """
        if len(jnp.shape(xi)) > 0:
            return vmap(self.f, in_axes=(0, None), out_axes=0)(xi, f_nodes)
        return self._f(xi, f_nodes, self._psi_coefs)

    def f_psi(self, xi: float | Array) -> Array:
        """Polynomial basis functions.

        Parameters
        ----------
        xi : float or array
            Coordinate or (M,) array of coordinates.

        Returns
        -------
        array
            (`n_nodes`,) or (M, `n_nodes`,) array of polynomial basis
            functions.

        """
        if len(jnp.shape(xi)) > 0:
            return vmap(self.f_psi, in_axes=0, out_axes=0)(xi)
        return self._f_psi(xi, self._psi_coefs)

    def _initialize_quad_methods(self, order: int, order_nodes: int):
        """Initialize the points and weights for quadrature methods.

        Parameters
        ----------
        order : int
            Order of the numerical quadratures over any interval
            ``[a, b]``.
        order_nodes : int
            Order of the numerical quadratures over the full interval
            ``[-1, 1]`` that use the node coordinates `xi` as stop points.

        """
        self._x_gl, self._w_gl = self._gauss_legendre_rule(order)
        self._x_gl_nodes, self._w_gl_nodes, self._psi_gl_nodes = self._initialize_quad_nodes(*self._gauss_legendre_rule(order_nodes))
        self._x_ts, self._w_ts = self._tanh_sinh_rule(order)
        self._x_ts_nodes, self._w_ts_nodes, self._psi_ts_nodes = self._initialize_quad_nodes(*self._tanh_sinh_rule(order_nodes))

    def _initialize_quad_nodes(self, x: ArrayLike, w: ArrayLike) -> Tuple[Array, Array, Array]:
        """Points and weights for quadrature with stop points at nodes.

        Parameters
        ----------
        x : array_like
            Quadrature points over an interval ``[-1, 1]``.
        w : array_like
            Quadrature weights associated with `x`.

        Returns
        -------
        x_nodes : array
            Quadrature points for integration with stop points at nodes.
        w_nodes : array
            Quadrature weights for integration with stop points at nodes.
        psi_nodes : array
            Values of the basis functions at quadrature points.

        """
        interval = jnp.concatenate([-jnp.ones(1), self.xi, jnp.ones(1)])
        low, high = interval[:-1], interval[1:]
        x_nodes = jnp.concatenate([0.5 * (b + a) + 0.5 * (b - a) * x for a, b in zip(low, high)])
        w_nodes = jnp.concatenate([0.5 * (b - a) * w for a, b in zip(low, high)])
        psi_nodes = self.f_psi(x_nodes)
        return x_nodes, w_nodes, psi_nodes

    def _quad(self, fun: Callable[[Array], Array], a: float, b: float, x: Array, w: Array) -> Array:
        """Integral of a function multiplied with basis functions.

        Parameters
        ----------
        fun : callable
            Function to integrate over the interval ``[a, b]``.
        a, b : float
            Lower (`a`) and upper (`b`) bounds of integration.
        x : array
            Quadrature points.
        w : array
            Quadrature weights.

        Returns
        -------
        array
            Integral of `fun` multiplied by the basis functions over the
            interval ``[a, b]``.

        """
        x = 0.5 * (b + a) + 0.5 * (b - a) * x
        w = 0.5 * (b - a) * w
        psi = self.f_psi(x)
        return self._quad_psi(fun, x, w, psi)

    def _quad_psi(self, fun: Callable[[Array | float], Array | float], x: Array, w: Array, psi: Array) -> Array:
        """Multiply function with basis and evaluate quadrature.

        Parameters
        ----------
        fun : callable
            Function to integrate.
        x : array
            Quadrature points.
        w : array
            Quadrature weights.
        psi : array
            Basis functions evaluated at quadrature points.

        Returns
        -------
        array
            Integral of `fun` multiplied by the basis functions.

        """
        integral = lambda _eta, _w, _psi: (fun(_eta) * _psi) @ _w
        return vmap(integral, in_axes=(None, None, 1), out_axes=-1)(x, w, psi)

    def quad_gl(self, fun: Callable[[Array | float], Array | float], a: float, b: float) -> Array:
        """Gauss-Legendre quadrature.

        Parameters
        ----------
        fun : callable
            Function to integrate over the interval ``[a, b]``.
        a, b : float
            Lower (`a`) and upper (`b`) bounds of integration.

        Returns
        -------
        array
            Integral of `fun` multiplied by the basis functions over the
            interval ``[a, b]``.

        """
        x, w = self._x_gl, self._w_gl
        return self._quad(fun, a, b, x, w)

    def quad_gl_nodes(self, fun: Callable[[Array | float], Array | float]) -> Array:
        """Gauss-Legendre quadrature with stop points at nodes.

        Parameters
        ----------
        fun : callable
            Function to integrate over the interval ``[-1, 1]``.

        Returns
        -------
        array
            Integral of `fun` multiplied by the basis functions over the
            interval ``[-1, 1]``.

        """
        x, w, psi = self._x_gl_nodes, self._w_gl_nodes, self._psi_gl_nodes
        return self._quad_psi(fun, x, w, psi)

    def quad_ts_nodes(self, fun: Callable[[Array | float], Array | float]) -> Array:
        """tanh-sinh quadrature with stop points at nodes.

        Parameters
        ----------
        fun : callable
            Function to integrate over the interval ``[-1, 1]``.

        Returns
        -------
        array
            Integral of `fun` multiplied by the basis functions over the
            interval ``[-1, 1]``.

        """
        x, w, psi = self._x_ts_nodes, self._w_ts_nodes, self._psi_ts_nodes
        return self._quad_psi(fun, x, w, psi)

    @staticmethod
    @jit
    def _f(xi: float, f_nodes: Array, psi_coefs: Array) -> float:
        """Value at coordinate from values at nodes.

        Parameters
        ----------
        xi : float
            Coordinate.
        f_nodes : array
            (`n_nodes`,) array of values at nodes.
        psi_coefs : array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients
            for the evaluation of the polynomial basis functions.

        Returns
        -------
        float
            Value at requested coordinate
            evaluated using polynomial basis functions.

        """
        psi = Basis._f_psi(xi, psi_coefs)
        return f_nodes @ psi

    @staticmethod
    @jit
    def _f_psi(xi: float, psi_coefs: Array) -> float:
        """Polynomial basis functions.

        Parameters
        ----------
        xi : float
            Coordinate.
        psi_coefs : array
            (`n_nodes`, `n_nodes`,) array of polynomial coefficients
            for the evaluation of the polynomial basis functions.

        Returns
        -------
        array
            (`n_nodes`,) array of polynomial basis functions.

        """
        psi = vmap(
            jnp.polyval,
            in_axes=(1, None),
            out_axes=0
        )(psi_coefs, xi)
        return psi

    @staticmethod
    def _gauss_legendre_rule(order: int) -> Tuple[Array, Array]:
        """Points and weights of the Gauss-Legendre quadrature.

        Returns
        -------
        x : array
            Quadrature points.
        w : array
            Quadrature weights.

        """
        x, w = roots_legendre(order)
        x = jnp.asarray(x)
        w = jnp.asarray(w)
        return x, w

    @staticmethod
    def _tanh_sinh_rule(order: int) -> Tuple[Array, Array]:
        """Points and weights of the tanh-sinh quadrature.

        Returns
        -------
        x : array
            Quadrature points.
        w : array
            Quadrature weights.

        """
        one_minus_eps = jnp.array(1.0) - 10 * jnp.finfo(jnp.array(1.0).dtype).eps
        tanh_inverse = lambda x: 0.5 * jnp.log((1 + x) / (1 - x))
        sinh_inverse = lambda x: jnp.log(x + jnp.sqrt(x**2 + 1))
        t_max = sinh_inverse(2. / jnp.pi * tanh_inverse(one_minus_eps))
        t = jnp.linspace(-t_max, t_max, num=order)

        x = jnp.tanh(0.5 * jnp.pi * jnp.sinh(t))
        w = 0.5 * jnp.pi * jnp.cosh(t) / jnp.cosh(0.5 * jnp.pi * jnp.sinh(t))**2
        w = w * jnp.diff(t)[0]
        w = 2 * w / jnp.sum(w)
        return x, w

    @classmethod
    def Legendre(cls, n_nodes: int, order: int = 101, order_nodes: int = 21) -> Self:
        """Legendre basis functions.

        Parameters
        ----------
        n_nodes : int
            Number of nodes. Corresponds to the order of the basis
            functions.
        order : int
            Order of the numerical quadratures over any interval
            ``[a, b]``.
        order_nodes : int
            Order of the numerical quadratures over the full interval
            ``[-1, 1]`` that use the node coordinates `xi` as stop points.

        Returns
        -------
        basis
            Instance of the `Basis` class with nodes located at
            Gauss-Legendre roots.

        """
        xi, w = roots_legendre(n_nodes)
        xi = jnp.array(xi)
        w = jnp.array(w)
        return cls(xi, order=order, order_nodes=order_nodes, w=w)
