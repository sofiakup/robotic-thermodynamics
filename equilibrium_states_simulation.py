import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import math
from scipy.special import ellipk, ellipe
from mpmath import ellippi
import time
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import os

ATMOSPHERIC_PRESSURE = 101325  # [Pa]


class Simulation:
    DEF_COIL_MAGNET_CONFIG = {
        "coil_inner_radius": (50 / 2) / 1000,  # [m]
        # "coil_outer_radius": (89 / 2) / 1000,  # [m]
        "coil_thickness": ((89 / 2) - (50 / 2)) / 1000,  # [m]
        "coil_length": 65 / 1000,  # [m]
        "magnet_radius": 12.7 / 1000,  # [m]
        "magnet_magnetization": 1.2,  # [T]
        # "wire_diameter": 1.12 / 1000,  # [m]
        "coil_length_turns": (65 / 1000) / (1.12 / 1000),
        "current": 1,  # [A]
        "magnet_length": 3.4 * 3 / 1000,  # [m], 3 magnets in a row
    }

    DEF_CAPSULE_GEOMETRY_CONFIG = {
        "frustum_open": 5.8 / 1000,  # [m]
        "frustum_close": 1.45 / 1000,  # [m]
        "inner_radius": 11.1 / 1000,  # [m]
        "snap_delta_length": (5.8 - 1.45) / 1000,  # [m]
    }

    DEF_SNAP_DISTANCE_CONFIG = {
        "snap_opening_distance": 2.826 / 1000,  # [m]
        "snap_closing_distance": 2.435 / 1000,  # [m]
        "snap_frusta_length_change": (5.8 - 1.45) / 1000,  # [m] # The difference in length between "frustum_open" and "frustum_close"
    }

    DEF_STIFFNESS_CONFIG = {
        "stiffness_open_opening": 24.77 * 1000,  # [N/m]
        "stiffness_open_closing": 15.43 * 1000,  # [N/m]
        "stiffness_closed_opening": 16.81 * 1000,  # [N/m]
        "stiffness_closed_closing": 230.14 * 1000,  # [N/m]
    }

    DEF_SNAP_FORCE_CONFIG = {
        "f_close": 6.147,  # [N]
        "f_open": 8.0567,  # [N]
    }

    DEF_DISTANCES_CONFIG = {
        "start": 0,
        "end": 150 / 1000,  # [m]
        "steps": 50,
    }

    DEF_INIT_CURRENT_1AMP = 1  # [A]
    DEF_INIT_GUESS_LOCATION_INTERSECTION = 10 / 1000  # [m]

    is_label_added = []

    def __init__(
        self,
        num_initial_frusta_open,
        num_initial_frusta_closed,
        initial_distance_from_coil_center,
        is_opening,
        is_magnet_force_calculated,
        init_current,
        init_length_change,
        gas_heat_capacity_ratio,
        state,
        coil_magnet_config: dict = None,
        capsule_geometry_config: dict = None,
        snap_distance: dict = None,
        stiffnesses_config: dict = None,
        snap_forces: dict = None,
        distance_config: dict = None,
    ):
        self.coil_magnet_config = self.DEF_COIL_MAGNET_CONFIG.copy()
        if coil_magnet_config is not None:
            self.coil_magnet_config.update(coil_magnet_config)

        self.capsule_geometry_config = self.DEF_CAPSULE_GEOMETRY_CONFIG.copy()
        if capsule_geometry_config is not None:
            self.capsule_geometry_config.update(capsule_geometry_config)

        self.snap_distance = self.DEF_SNAP_DISTANCE_CONFIG.copy()
        if snap_distance is not None:
            self.snap_distance.update(snap_distance)

        self.stiffnesses_config = self.DEF_STIFFNESS_CONFIG.copy()
        if stiffnesses_config is not None:
            self.snap_forces.update(stiffnesses_config)

        self.snap_forces = self.DEF_SNAP_FORCE_CONFIG.copy()
        if snap_forces is not None:
            self.snap_forces.update(snap_forces)

        self.distance_config = self.DEF_DISTANCES_CONFIG.copy()
        if distance_config is not None:
            self.distance_config.update(distance_config)

        self.init_current = self.DEF_INIT_CURRENT_1AMP

        self.state = {}

        self.init_intersection_location_guess = (
            self.DEF_INIT_GUESS_LOCATION_INTERSECTION
        )

        self.number_frusta_open = num_initial_frusta_open
        self.number_frusta_closed = num_initial_frusta_closed
        self.init_capsule_length = (
            num_initial_frusta_open * self.capsule_geometry_config["frustum_open"]
            + num_initial_frusta_closed * self.capsule_geometry_config["frustum_close"]
        )

        self.capsule_length = self.init_capsule_length

        self.is_opening = True

        self.is_magnet_force_calculated = False
        self.length_change = init_length_change
        self.init_magnet_distance_from_coil_center = initial_distance_from_coil_center
        self.magnet_distance_from_coil_center = initial_distance_from_coil_center

        self.distance_range = np.linspace(
            self.distance_config["start"],
            self.distance_config["end"],
            self.distance_config["steps"],
        )

        self.gas_heat_capacity_ratio = gas_heat_capacity_ratio

        self.is_label_added = False
        self.is_label_added2 = False
        self.is_first_point = True

        self.distance_range_capsule_squeeze = []

        self.fig, self.axs = plt.subplots(
            1, 2, figsize=(9, 4), dpi=100
        )
        plt.tight_layout()

    def calculate_magnetic_force(self, distance_from_coil_center, current):
        coil_inner_radius = self.coil_magnet_config["coil_inner_radius"]
        coil_thickness = self.coil_magnet_config["coil_thickness"]
        coil_length = self.coil_magnet_config["coil_length"]
        magnet_radius = self.coil_magnet_config["magnet_radius"]
        magnet_magnetization = self.coil_magnet_config["magnet_magnetization"]
        coil_length_turns = self.coil_magnet_config["coil_length_turns"]
        magnet_length = self.coil_magnet_config["magnet_length"]

        wire_diamer = coil_length / coil_length_turns
        coil_radial_turns = int(coil_thickness / wire_diamer)
        coil_total_turns = coil_length_turns * coil_radial_turns

        Fs_sum = 0

        for n_r in range(1, coil_radial_turns + 1):
            if coil_radial_turns > 1:
                r = coil_inner_radius + (n_r - 1) / (coil_radial_turns - 1) * (
                    coil_thickness
                )
            else:
                r = coil_inner_radius

            e_i = [1, -1]
            combinations = product(e_i, repeat=2)

            J1 = magnet_magnetization
            J2 = coil_total_turns * current / coil_length

            for e_i in combinations:
                m1 = (
                    distance_from_coil_center
                    - 1 / 2 * e_i[0] * magnet_length
                    - 1 / 2 * e_i[1] * coil_length
                )
                m2 = (magnet_radius - r) ** 2 / (m1**2) + 1
                m3 = math.sqrt((magnet_radius + r) ** 2 + m1**2)
                # m4 = 4*magnet_radius*r/m3 # corection - mistake in the paper (2012)
                m4 = 4 * magnet_radius * r / m3**2

                Fs = (
                    J1
                    * J2
                    / 2
                    * e_i[0]
                    * e_i[1]
                    * m1
                    * m2
                    * m3
                    * self._fs(m1, m2, m3, m4)
                )

                Fs_sum = Fs_sum + Fs

        F_z_2 = Fs_sum / coil_radial_turns
        F_m = float(F_z_2)

        return F_m

    def _fs(self, m1, m2, m3, m4):
        if m2 != 1 and m1 != 0:
            fs_calc = (
                ellipk(m4)
                - 1 / m2 * ellipe(m4)
                + ((m1**2) / (m3**2) - 1) * ellippi(m4 / (1 - m2), m4)
            )
        elif m1 == 0:
            fs_calc = 0
            m2 = 0
        elif m2 == 1:
            fs_calc = ellipk(m4) - 1 / m2 * ellipe(m4)

        return fs_calc

    def step(self, current):
        self._update_force_sign(current)

        [
            intersection_locations,
            intersection_values,
            elastic_force2,
            magnetic_force2,
            gas_force2,
        ] = self.load_capsule(current)

        self.state = {
            "magnetic_force": magnetic_force2,  # [N]
        }

        while (
            self.is_opening is True
            and intersection_values[0] > self.snap_forces["f_open"]
        ) or (
            self.is_opening is False
            and intersection_values[0] < -self.snap_forces["f_close"]
        ):
            if (self.is_opening is True and self.number_frusta_closed == 0) or (
                self.is_opening is False and self.number_frusta_open == 0
            ):
                break

            self.update_frustas_after_snap()
            [
                intersection_locations,
                intersection_values,
                elastic_force2,
                magnetic_force2,
                gas_force2,
            ] = self.load_capsule(current)

        self.state = {
            "current": current,  # [A]
            "equilibrium_locations": intersection_locations,  # [m]
            "equilibrium_forces": intersection_values,  # [N]
            "elastic_force": elastic_force2,  # [N]
            "magnetic_force": magnetic_force2,  # [N]
            "gas_force": gas_force2,  # [N]
        }

        self._plot_F_eq_vs_distance_fig_current_state()
        self._plot_I_vs_distance_fig_current_state()

        return self.state

    def load_capsule(self, current):
        self._update_effective_stiffness()
        elastic_force = self.interp_elastic_force()

        magnetic_force = self.interp_magnetic_force()
        gas_force = self.interp_gas_force()

        [intersection_locations, intersection_values] = self.find_equilibrium(
            magnetic_force, current, elastic_force, gas_force
        )

        if current != 0 and (
            (intersection_values[0] < 0 and self.is_opening != False)
            or (intersection_values[0] > 0 and self.is_opening != True)
        ):
            self.is_opening = not self.is_opening

            self._update_effective_stiffness()
            elastic_force = self.interp_elastic_force()

            [intersection_locations, intersection_values] = self.find_equilibrium(
                magnetic_force, current, elastic_force, gas_force
            )

        return (
            intersection_locations,
            intersection_values,
            elastic_force,
            magnetic_force,
            gas_force,
        )

    def _update_force_sign(self, current):
        # Special case - when the current is zero. The stiffness depends on the previous equilibrium value, i.e. whether the gas in the capsule will resist:
        # Below "1e-06" the equilibrium forces are considered zero
        # if abs(self.state["equilibrium_forces"])>1e-06 and self.state["equilibrium_forces"]>0:
        # According to the convention: for a magnet on the right side of the coil -> F_m < 0 contracts the capsule
        if current > 0:
            self.is_opening = True
        elif current < 0:
            self.is_opening = False
        else:
            if self.state == {} or abs(self.state["equilibrium_forces"]) < 1e-06:
                pass
            else:
                if self.state["equilibrium_forces"] > 0:
                    self.is_opening = True
                elif self.state["equilibrium_forces"] > 0:
                    self.is_opening = False

    def _update_effective_stiffness(self):
        if self.is_opening is True:
            self.effective_stiffness = (
                self.number_frusta_open
                / self.stiffnesses_config["stiffness_open_opening"]
                + self.number_frusta_closed
                / self.stiffnesses_config["stiffness_closed_opening"]
            ) ** (-1)

        if self.is_opening is False:
            self.effective_stiffness = (
                self.number_frusta_open
                / self.stiffnesses_config["stiffness_open_closing"]
                + self.number_frusta_closed
                / self.stiffnesses_config["stiffness_closed_closing"]
            ) ** (-1)

    def interp_magnetic_force(self):
        if self.is_magnet_force_calculated is False:
            magnet_force_1_amp = np.zeros_like(self.distance_range)

            # Calculating the magnetic force in the entire range
            for index, distance_from_coil_center in enumerate(self.distance_range):
                magnet_force_1_amp[index] = self.calculate_magnetic_force(
                    distance_from_coil_center, self.init_current
                )

            # Defining an interpolation function for the magnetic force
            interp_func_magnet_force = interp1d(
                self.distance_range,
                -magnet_force_1_amp.T,
                kind="cubic",
            )  # Minus sign needed because the convension in the paper is the opposite sign

            self.is_magnet_force_calculated = True

            return interp_func_magnet_force
        else:
            return self.state["magnetic_force"]

    def interp_elastic_force(self):

        elastic_force = -self.effective_stiffness * (
            self.distance_range - self.magnet_distance_from_coil_center
        )

        elastic_force_interp_func = interp1d(
            self.distance_range, elastic_force, kind="cubic"
        )

        return elastic_force_interp_func

    def interp_gas_force(self):
        gas_heat_capacity_ratio = self.gas_heat_capacity_ratio
        inner_radius = self.capsule_geometry_config["inner_radius"]
        A_capsule = math.pi * (inner_radius**2)
        p_atm = ATMOSPHERIC_PRESSURE

        # Inf when capsule length is zero
        max_capsule_elongation_or_compression = 10
        self.distance_range_capsule_squeeze = np.linspace(
            self.init_capsule_length * max_capsule_elongation_or_compression,
            self.init_capsule_length / max_capsule_elongation_or_compression,
            max_capsule_elongation_or_compression * 100,
        )

        gas_force = (
            A_capsule
            * p_atm
            * (
                1
                - (
                    (
                        self.init_capsule_length
                        / (
                            self.distance_range_capsule_squeeze
                            + (self.capsule_length - self.init_capsule_length)
                        )
                    )
                    ** gas_heat_capacity_ratio
                )
            )
        )

        self.distance_range_magnet_location = self.magnet_distance_from_coil_center + (
            self.init_capsule_length - self.distance_range_capsule_squeeze
        )

        gas_force_interp_func = interp1d(
            self.distance_range_magnet_location, gas_force, kind="cubic"
        )

        return gas_force_interp_func

    def find_equilibrium(
        self, magnetic_force, current_amplification, elastic_force, gas_force
    ):
        # Finding the intersection points
        def _intersection_equation(location):
            return elastic_force(location) - (
                current_amplification * magnetic_force(location) - gas_force(location)
            )

        intersection_locations = fsolve(
            _intersection_equation, self.init_intersection_location_guess
        )

        intersection_values = elastic_force(intersection_locations)


        return intersection_locations, intersection_values

    def update_frustas_after_snap(self):
        if self.number_frusta_open != 0 and self.number_frusta_closed != 0:
            if self.is_opening is True:
                self.number_frusta_open = self.number_frusta_open + 1
                self.number_frusta_closed = self.number_frusta_closed - 1
                self.capsule_length = (
                    self.capsule_length
                    + self.snap_distance["snap_frusta_length_change"]
                )
                self.length_change = (
                    self.length_change + self.snap_distance["snap_frusta_length_change"]
                )

                self.magnet_distance_from_coil_center = (
                    self.magnet_distance_from_coil_center
                    - self.snap_distance["snap_frusta_length_change"]
                )

            if self.is_opening is False:
                self.number_frusta_open = self.number_frusta_open - 1
                self.number_frusta_closed = self.number_frusta_closed + 1
                self.capsule_length = (
                    self.capsule_length
                    - self.snap_distance["snap_frusta_length_change"]
                )
                self.length_change = (
                    self.length_change - self.snap_distance["snap_frusta_length_change"]
                )

                self.magnet_distance_from_coil_center = (
                    self.magnet_distance_from_coil_center
                    + self.snap_distance["snap_frusta_length_change"]
                )

    def _plot_F_eq_fig_base(self):
        local_axs = self.axs[0]
        # Plotting referece line
        local_axs.plot(
            [0, self.distance_config["end"] * 1000],
            0 * np.ones(2),
            color="black",
            linewidth=1,
            linestyle="dashed",
        )
        # Plotting force to sanp open
        local_axs.plot(
            [0, self.distance_config["end"] * 1000],
            np.ones(2) * -self.snap_forces["f_close"],
            color="red",
            linewidth=1,
            label="$F_{open}$ or $F_{close}$",
        )
        # Plotting force to sanp open
        local_axs.plot(
            [0, self.distance_config["end"] * 1000],
            np.ones(2) * self.snap_forces["f_open"],
            color="red",
            linewidth=1,
        )

        local_axs.set_xlim([0, self.distance_config["end"] * 1000])
        local_axs.set_ylim([-15, 15])
        local_axs.set_xlabel("Distance from coil center [mm]")
        local_axs.set_ylabel("Forces [N]")


    def _plot_F_eq_vs_distance_fig_current_state(self):
        local_axs = self.axs[0]

        # Visualizing the magnetic force interpolation, finer discritization
        self.distance_range2 = np.linspace(
            self.distance_config["start"],
            self.distance_config["end"],
            self.distance_config["steps"]
            * 100,  # High resolution here is important to determine the zero for elastic force graphic representation
        )

        # Transforming distance from center of coil -> distance_range_magnet_location
        # Based on distance_range_capsule_squeeze
        max_gas_force_interp_location = self.distance_range_magnet_location[-1]
        #  Identifiying the range of the gas forces to plot
        max_gas_plot_location = math.floor(
            max_gas_force_interp_location / self.distance_range2[1]
        )

        # Finding the intersection of the elastic force with F=0 axis
        closest_index_zero_force = np.argmin(
            np.abs(self.distance_range2 - self.magnet_distance_from_coil_center)
        )

        elastic_force = self.state["elastic_force"](self.distance_range2)
        magnetic_force = (
            self.state["magnetic_force"](self.distance_range2) * self.state["current"]
        )
        # For F_{m}-F_{gas} plotting
        magnetic_force2 = (
            self.state["magnetic_force"](self.distance_range2[1:max_gas_plot_location])
            * self.state["current"]
        )
        gas_force = self.state["gas_force"](
            self.distance_range2[1:max_gas_plot_location]
        )

        if (
            self.state["current"] == 0 and self.is_first_point == True
        ):
            pass
        else:
            if self.is_opening == False and self.number_frusta_open != 0:
                index_f_close = np.argmin(
                    np.abs(elastic_force - (-self.snap_forces["f_close"]))
                )
                local_axs.plot(
                    self.distance_range2[closest_index_zero_force:index_f_close] * 1000,
                    elastic_force[closest_index_zero_force:index_f_close].T,
                    color="C0",
                    label="$F_{e}$",
                )
                if not self.is_label_added2:
                    local_axs.legend()
                    self.is_label_added2 = True
            elif self.is_opening == False and self.number_frusta_open == 0:
                local_axs.plot(
                    self.distance_range2[closest_index_zero_force:] * 1000,
                    elastic_force[closest_index_zero_force:].T,
                    color="C0",
                    label="$F_{e}$",
                )
                if not self.is_label_added2:
                    local_axs.legend()
                    self.is_label_added2 = True

            if self.is_opening == True and self.number_frusta_closed != 0:
                index_f_close = np.argmin(
                    np.abs(elastic_force - self.snap_forces["f_open"])
                )
                local_axs.plot(
                    self.distance_range2[index_f_close:closest_index_zero_force] * 1000,
                    elastic_force[index_f_close:closest_index_zero_force].T,
                    color="C0",
                    label="$F_{e}$",
                )
                if not self.is_label_added2:
                    local_axs.legend()
                    self.is_label_added2 = True
            elif self.is_opening == True and self.number_frusta_closed == 0:
                local_axs.plot(
                    self.distance_range2[1:closest_index_zero_force] * 1000,
                    elastic_force[1:closest_index_zero_force].T,
                    color="C0",
                    label="$F_{e}$",
                )
                if not self.is_label_added2:
                    local_axs.legend()
                    self.is_label_added2 = True

        local_axs.plot(
            self.distance_range2 * 1000,
            magnetic_force.T,
            color="black",
            linewidth=0.5,
            label="$F_{m}$",
        )

        local_axs.plot(
            self.distance_range2[1:max_gas_plot_location] * 1000,
            gas_force.T,
            color="C3",
            # linewidth=0.5,
            label="$F_{gas}$",
        )

        local_axs.plot(
            self.distance_range2[1:max_gas_plot_location] * 1000,
            magnetic_force2.T - gas_force.T,
            color="C9",
            label="$F_{m}-F_{gas}$",
            linewidth=0.5,
        )

        if not self.is_label_added:
            local_axs.legend()
            self.is_label_added = True

        local_axs.set_xlim([0, self.distance_config["end"] * 1000])
        local_axs.set_ylim([-15, 15])

        # Plot intersection points
        for x_intersection in self.state["equilibrium_locations"]:
            y_intersection = self.state["elastic_force"](
                x_intersection
            ) 
            local_axs.scatter(
                x_intersection * 1000,
                y_intersection,
                s=25,
                marker="o",
                facecolors="none",
                edgecolors="C4",
                linewidths=1.5,
            )


    def _plot_I_vs_distance_fig_current_state(self):
        local_axs = self.axs[1]

        local_axs.scatter(
            self.state["equilibrium_locations"] * 1000,
            self.state["current"],
            s=25,
            marker="o",
            facecolors="none",
            edgecolors="C4",
            linewidths=1.5,
        )

        local_axs.set_xlabel("Distance from coil center [mm]")
        local_axs.set_ylabel("Current [A]")


    def Calculate_magnetic_force_radial_component(self, mu_0, n, II, rho, a, b, z):
        B_0 = mu_0 * n * II / math.pi
        z_plus = z + b
        z_minus = z - b
        alpha_plus = a / math.sqrt(z_plus**2 + (rho + a) ** 2)
        alpha_minus = a / math.sqrt(z_minus**2 - (rho + a) ** 2)
        k_plus = math.sqrt((z_plus**2 + (a - rho) ** 2) / (z_plus**2 + (a + rho) ** 2))
        k_minus = math.sqrt(
            (z_minus**2 - (a - rho) ** 2) / (z_minus**2 - (a + rho) ** 2)
        )
        B_r = B_0 * (
            alpha_plus * self.c_func(k_plus, 1, 1, -1)
            - alpha_minus * self.c_func(k_minus, 1, 1, -1)
        )
        return B_r

    def Calculate_magnetic_force_axial_component(self, mu_0, n, II, rho, a, b, z):
        B_0 = mu_0 * n * II / math.pi
        z_plus = z + b
        z_minus = z - b
        gamma = (a - rho) / (a + rho)
        beta_plus = 1
        beta_minus = -1
        k_plus = math.sqrt((z_plus**2 + (a - rho) ** 2) / (z_plus**2 + (a + rho) ** 2))
        k_minus = math.sqrt(
            (z_minus**2 - (a - rho) ** 2) / (z_minus**2 - (a + rho) ** 2)
        )
        B_z = (
            B_0
            * a
            / (a + rho)
            * (
                beta_plus * self.c_func(k_plus, gamma**2, 1, gamma)
                - beta_minus * self.c_func(k_minus, gamma**2, 1, gamma)
            )
        )
        return B_z

    def c_func(self, kc, p, c, s):
        if kc == 0:
            raise Exception("Error: kc is zero.")
        err_tolerance = 10**-6
        k = abs(kc)
        pp = p
        cc = c
        ss = s
        em = 1
        if p > 0:
            pp = math.sqrt(p)
            ss = s / pp
        else:
            f = kc * kc
            q = 1 - f
            g = 1 - pp
            f = f - pp
            q = q * (ss - c * pp)
            pp = math.sqrt(f / g)
            cc = (c - ss) / g
            ss = -q / (g * g * pp) + cc * pp
        f = cc
        cc = cc + ss / pp
        g = k / pp
        ss = 2 * (ss + f * g)
        pp = g + pp
        g = em
        em = k + em
        kk = k
        while abs(g - k) > g * err_tolerance:
            k = 2 * math.sqrt(kk)
            kk = k * em
            f = cc
            cc = cc + ss / pp
            g = kk / pp
            ss = 2 * (ss + f * g)
            pp = g + pp
            g = em
            em = k + em
        c_value = (math.pi / 2) * (ss + cc * em) / (em * (em + pp))
        return c_value


def saving_fig(figure_name, figure_folder_path, formats):
    # Ensure the folder path exists; create it if not
    if not os.path.exists(figure_folder_path):
        os.makedirs(figure_folder_path)
    for ext in formats:
        plt.savefig(os.path.join(figure_folder_path, f"{figure_name}.{ext}"))


if __name__ == "__main__":
    begin_time_count_seconds = time.time()

    NUM_INITIAL_FRUSTA_OPEN = 20
    NUM_INITIAL_FRUSTA_CLOSED = 20
    is_opening = True
    is_magnet_force_calculated = False
    INIT_CURRENT = 1
    INIT_DISTANCE = 30 / 1000
    INIT_LENGTH_CHANGE = 0
    GAS_HEAT_CAPACITY_RATIO = 1.4

    DISTANCES = {
        "start": 0,
        "end": 150 / 1000,
        "steps": 50,
    }
    distance_range = np.linspace(
        DISTANCES["start"], DISTANCES["end"], DISTANCES["steps"]
    )

    formats = ["svg", "pdf", "png"]

    # Simulation configurations
    sim_config = {}
    state = {}

    sim = Simulation(
        NUM_INITIAL_FRUSTA_OPEN,
        NUM_INITIAL_FRUSTA_CLOSED,
        INIT_DISTANCE,
        is_opening,
        is_magnet_force_calculated,
        INIT_CURRENT,
        INIT_LENGTH_CHANGE,
        GAS_HEAT_CAPACITY_RATIO,
        sim_config,
        state,
        distance_config=DISTANCES,
    )

    sim._plot_F_eq_fig_base()

    Results = {
        "currents": [],
        "equilibrium_locations": [],
        "equilibrium_forces": [],
        "elastic_force": [],
        "gas_force": [],
    }

    CURRENT_SPANS = [
        (0, -11, -1),  # [A], start_val, end_val, step
        (-11, 0.1, 1),
        (0, 11, 1),
        (11, -0.1, -1),
    ]

    for start_val, end_val, step in CURRENT_SPANS:
        for current in np.arange(start_val, end_val, step):
            state = sim.step(current)
            Results["currents"].append(state["current"])
            Results["equilibrium_locations"].append(state["equilibrium_locations"])
            Results["equilibrium_forces"].append(state["equilibrium_forces"])
            Results["elastic_force"].append(state["elastic_force"])
            Results["gas_force"].append(state["gas_force"])

    figure_folder_path = "figure_folder_path" # INPUR REQUIRED
    formats = ["png", "pdf", "svg"]


    plt.annotate(
        "Initial state:\n"
        f"Num frusta open: {NUM_INITIAL_FRUSTA_OPEN}\n"
        f"Num frusta closed: {NUM_INITIAL_FRUSTA_CLOSED}",
        xy=(0.55, 0.155),
        xycoords="figure fraction",  # Fraction of figure from lower left
        fontsize=10,
    )

    elapsed_seconds = time.time() - begin_time_count_seconds
    print("Runtime: %.1f [s]" % elapsed_seconds)

    plt.show(block=True)
