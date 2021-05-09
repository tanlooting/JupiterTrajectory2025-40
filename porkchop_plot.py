# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 22:55:09 2019

@author: Loo Ting
"""

"""
This is the script for porkchop plotting
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import coordinates as coord, units as u

from poliastro.bodies import (
    Earth,
    Jupiter,
    Mars,
    Mercury,
    Moon,
    Neptune,
    Pluto,
    Saturn,
    Sun,
    Uranus,
    Venus,
)
from poliastro.maneuver import Maneuver
from poliastro.twobody.orbit import Orbit
from poliastro.util import norm

def _get_state(body, time):
    """ Computes the position of a body for a given time. """

    solar_system_bodies = [
        Sun,
        Mercury,
        Venus,
        Earth,
        Moon,
        Mars,
        Jupiter,
        Saturn,
        Uranus,
        Neptune,
        Pluto,
    ]

    # We check if body belongs to poliastro.bodies
    if body in solar_system_bodies:
        rr, vv = coord.get_body_barycentric_posvel(body.name, time)
    else:
        rr, vv = body.propagate(time).rv()
        rr = coord.CartesianRepresentation(rr)
        vv = coord.CartesianRepresentation(vv)

    return rr.xyz, vv.xyz


def _targetting(departure_body, target_body, t_launch, t_arrival):
    """This function returns the increment in departure and arrival velocities."""

    # Get position and velocities for departure and arrival
    rr_dpt_body, vv_dpt_body = _get_state(departure_body, t_launch)
    rr_arr_body, vv_arr_body = _get_state(target_body, t_arrival)

    # Transform into Orbit objects
    attractor = departure_body.parent
    ss_dpt = Orbit.from_vectors(attractor, rr_dpt_body, vv_dpt_body, epoch=t_launch)
    ss_arr = Orbit.from_vectors(attractor, rr_arr_body, vv_arr_body, epoch=t_arrival)

    # Define time of flight
    tof = ss_arr.epoch - ss_dpt.epoch

    if tof <= 0:
        return None, None, None, None, None

    try:
        # Lambert is now a Maneuver object
        man_lambert = Maneuver.lambert(ss_dpt, ss_arr)

        # Get norm delta velocities
        dv_dpt = norm(man_lambert.impulses[0][1])
        dv_arr = norm(man_lambert.impulses[1][1])

        # Compute all the output variables
        c3_launch = dv_dpt ** 2
        c3_arrival = dv_arr ** 2

        return (
            dv_dpt.to(u.km / u.s).value,
            dv_arr.to(u.km / u.s).value,
            c3_launch.to(u.km ** 2 / u.s ** 2).value,
            c3_arrival.to(u.km ** 2 / u.s ** 2).value,
            tof.jd,
        )

    except AssertionError:
        return None, None, None, None, None


# numpy.vectorize is amazing
targetting_vec = np.vectorize(
    _targetting,
    otypes=[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    excluded=[0, 1],
)


def porkchop(
    departure_body,
    target_body,
    launch_span,
    arrival_span,
    ax=None,
    tfl=True,
    vhp=True,
    max_c3=45.0 * u.km ** 2 / u.s ** 2,
    max_vhp=5 * u.km / u.s,
):
    """Plots porkchop between two bodies.
    Parameters
    ----------
    departure_body: poliastro.bodies.Body
        Body from which departure is done
    target_body: poliastro.bodies.Body
        Body for targetting
    launch_span: astropy.time.Time
        Time span for launch
    arrival_span: astropy.time.Time
        Time span for arrival
    ax: matplotlib.axes.Axes:
        For custom figures
    tfl: boolean
        For plotting time flight contour lines
    vhp: boolean
        For plotting arrival velocity contour lines
    max_c3: float
        Sets the maximum C3 value for porkchop
    max_vhp: float
        Sets the maximum arrival velocity for porkchop
    Returns
    -------
    c3_launch: np.ndarray
        Characteristic launch energy
    c3_arrrival: np.ndarray
        Characteristic arrival energy
    tof: np.ndarray
        Time of flight for each transfer
    Example
    -------
    >>> from poliastro.plotting.porkchop import porkchop
    >>> from poliastro.bodies import Earth, Mars
    >>> from poliastro.util import time_range
    >>> launch_span = time_range("2005-04-30", end="2005-10-07")
    >>> arrival_span = time_range("2005-11-16", end="2006-12-21")
    >>> dv_launch, dev_dpt, c3dpt, c3arr, tof = porkchop(Earth, Mars, launch_span, arrival_span)
    """

    dv_launch, dv_arrival, c3_launch, c3_arrival, tof = targetting_vec(
        departure_body,
        target_body,
        launch_span[np.newaxis, :],
        arrival_span[:, np.newaxis],
    )

    # Start drawing porkchop

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 10))
    else:
        fig = ax.figure

    c3_levels = np.linspace(0, max_c3.to(u.km ** 2 / u.s ** 2).value, 30)

    c = ax.contourf(
        [D.to_datetime() for D in launch_span],
        [A.to_datetime() for A in arrival_span],
        c3_arrival,
        c3_levels,
    )
    
    line = ax.contour(
        [D.to_datetime() for D in launch_span],
        [A.to_datetime() for A in arrival_span],
        c3_arrival,
        c3_levels,
        colors="black",
        linestyles="solid",
        linewidths=0.5,
    )

    cbar = fig.colorbar(c)
    cbar.set_label("km2 / s2")
    ax.clabel(line, inline=1, fmt="%1.1f", colors="k", fontsize=10)

    if tfl:

        time_levels = np.linspace(500, 5000, 10)

        tfl_contour = ax.contour(
            [D.to_datetime() for D in launch_span],
            [A.to_datetime() for A in arrival_span],
            tof,
            time_levels,
            colors="red",
            linestyles="dashed",
            linewidths=1,
        )

        ax.clabel(tfl_contour, inline=1, fmt="%1.1f", colors="r", fontsize=14)

    if vhp:

        vhp_levels = np.linspace(0, max_vhp.to(u.km / u.s).value, 5)

        vhp_contour = ax.contour(
            [D.to_datetime() for D in launch_span],
            [A.to_datetime() for A in arrival_span],
            dv_arrival,
            vhp_levels,
            colors="navy",
            linewidths=1.0,
        )

        ax.clabel(vhp_contour, inline=1, fmt="%1.1f", colors="navy", fontsize=12)

    ax.grid()
    fig.autofmt_xdate()

    if not hasattr(target_body, "name"):
        ax.set_title(
            f"{departure_body.name} - Target Body for year {launch_span[0].datetime.year}, C3 Arrival",
            fontsize=14,
            fontweight="bold",
        )
    else:
        ax.set_title(
            f"{departure_body.name} - {target_body.name} for year {launch_span[0].datetime.year}, C3 Arrival",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xlabel("Launch date", fontsize=10, fontweight="bold")
    ax.set_ylabel("Arrival date", fontsize=10, fontweight="bold")

    return (
        dv_launch * u.km / u.s,
        dv_arrival * u.km / u.s,
        c3_launch * u.km ** 2 / u.s ** 2,
        c3_arrival * u.km ** 2 / u.s ** 2,
        tof * u.d,
    )
    
    
launch_span = time_range("2032-10-15", end="2033-11-30")
arrival_span = time_range("2032-07-01", end="2040-12-31")

dv_dpt, dv_arr, c3dpt, c3arr, tof = porkchop(Earth, Jupiter, launch_span, arrival_span,tfl=True,vhp=True,max_c3=115 * u.km**2 / u.s**2)