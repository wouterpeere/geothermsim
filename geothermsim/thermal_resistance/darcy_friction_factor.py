# -*- coding: utf-8 -*-
from jax import numpy as jnp


def darcy_friction_factor(Re: float, E: float) -> float:
    """
    Darcy-Weisbach friction factor.

    Parameters
    ----------
    Re : float
        Reynolds number.
    E : float
        Relative pipe roughness (epsilon / D).

    Returns
    -------
    f_darcy : float
        Darcy friction factor.

    """
    Re_crit = 2300.
    # Condition on the flow regime
    condlist = [Re <= Re_crit]
    # Piecewise evaluation of Darcy friction factor
    funclist = [
        darcy_friction_factor_laminar_flow,
        darcy_friction_factor_turbulent_flow
    ]
    f_darcy = jnp.piecewise(jnp.float32(Re), condlist, funclist, E)
    return f_darcy


def darcy_friction_factor_laminar_flow(Re: float, *args) -> float:
    """
    Darcy-Weisbach friction factor for laminar flow in circular pipe.

    Parameters
    ----------
    Re : float
        Reynolds number.

    Returns
    -------
    f_darcy : float
        Darcy friction factor.

    """
    return 64 / Re


def darcy_friction_factor_turbulent_flow(Re: float, E: float) -> float:
    """
    Darcy-Weisbach friction factor for turbulent flow in circular pipe.

    Parameters
    ----------
    Re : float
        Reynolds number.
    E : float
        Relative pipe roughness (epsilon / D).

    Returns
    -------
    f_darcy : float
        Darcy friction factor.

    """
    # Churchill equation for rough pipes
    A = (
        2.457 * jnp.log(
            1 / ((7 / Re)**0.9 + (0.27 * E))
            )
        )**16
    B = (37_530 / Re)**16
    f_darcy = 8 * ((8 / Re)**12 + 1 / (A + B)**1.5)**(1 / 12)
    return f_darcy
