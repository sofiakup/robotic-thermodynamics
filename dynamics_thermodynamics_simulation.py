import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math
from scipy.signal import find_peaks


def spring_dynamics(
    t,
    y,
    external_force,
    F_drag_magnetic,
    gas_force,
    elastic_force_func,
    temperature_external,
    stiffnesses,
    elastic_spring_stiffness_change_locations,
    parameters,
    force_actuation,
    distance_range_B_r,
    radial_magnetic_field_component_B_r_values,
    z_Br_eval,
    int_Br,
    int_Br_squared,
    thermal_actuation_parameters,
):
    l0_aux = parameters["capsule_auxiliary_length"]
    capsule_side_area = parameters["capsule_side_area"]
    drag_coeff = parameters["drag_coeff"]
    p_atm = parameters["pressure_atm"]
    m_gas = parameters["mass_gas"]
    h = parameters["h"]
    R_specific_heat = parameters["R_specific_heat"]
    magnet_mass = parameters["magnet_mass"]
    r_capsule = parameters["capsule_radius"]
    drag_coeff = parameters["drag_coeff"]
    coil_resistance = parameters["coil_resistance"]

    x, v, T_in, s_entropy, E_losses, W_F_ext, Int_Ugas_dt = y

    # Dynamics of the spring-mass system
    dxdt = v
    dvdt = (
        external_force(x, t, force_actuation)
        - capsule_side_area * p_atm
        - (
            0 * drag_coeff * v
            + F_drag_magnetic(x, v, parameters, z_Br_eval, int_Br_squared)
            + elastic_force_func(
                x, stiffnesses, elastic_spring_stiffness_change_locations
            )
            - gas_force(x, T_in, parameters)
        )
    ) / magnet_mass

    # Temperature
    dVdt = v
    dTdt = (
        (2 * math.pi * r_capsule)
        * (x + l0_aux)
        * h
        * (temperature_external(t, thermal_actuation_parameters) - T_in)
        - (m_gas * R_specific_heat * T_in * dVdt) / (x + l0_aux)
    ) / (c_v * m_gas)

    # Enthropy
    dsdt = c_v * m_gas * dTdt / T_in + m_gas * R_specific_heat * dVdt / (x + l0_aux)

    # Power and energy - losses (heat & dissipation)
    dE_losses_dt = power_losses(x, v, parameters, z_Br_eval, int_Br_squared)

    dW_Fext_dt = external_force(x, t, force_actuation) * v

    Int_Ugas_dt = c_v * mass_gas * T_in

    return [dxdt, dvdt, dTdt, dsdt, dE_losses_dt, dW_Fext_dt, Int_Ugas_dt]


def gas_force(x, T_in, parameters):
    m_gas = parameters["mass_gas"]
    R_specific_heat = parameters["R_specific_heat"]
    l0_aux = parameters["capsule_auxiliary_length"]

    F_gas = m_gas * R_specific_heat * T_in / (x + l0_aux)

    return F_gas


def elastic_force_func(x, stiffnesses, elastic_spring_stiffness_change_locations):
    X0 = elastic_spring_stiffness_change_locations
    K = stiffnesses
    F_el = 0
    if x < X0["x1"]:
        F_el = K["stiffness_closed_closing"] * (x - X0["x1"])
    elif X0["x1"] <= x and x < X0["Delta_1"]:
        F_el = K["stiffness_closed_opening"] * (x - X0["x1"])
    elif X0["Delta_1"] <= x and x < X0["Delta_2"]:
        _k_s = (
            K["stiffness_open_closing"] * (X0["Delta_2"] - X0["x2"])
            - K["stiffness_closed_opening"] * X0["Delta_1"]
        ) / (X0["Delta_2"] - X0["Delta_1"])
        F_el = (
            _k_s * (x - X0["Delta_1"]) + K["stiffness_closed_opening"] * X0["Delta_1"]
        )  # k_s<0
    elif X0["Delta_2"] <= x and x < X0["x2"]:
        F_el = K["stiffness_open_closing"] * (x - X0["x2"])
    elif x >= X0["x2"]:
        F_el = K["stiffness_open_opening"] * (x - X0["x2"])
    else:
        F_el = 0

    return F_el


def elastic_energy_func(x, STIFFNESSES, ELASTIC_SPRING_STIFFNESS_CHANGE_LOCATIONS):
    X0 = ELASTIC_SPRING_STIFFNESS_CHANGE_LOCATIONS
    K = STIFFNESSES
    F_el = 0
    E_el = 0
    force_1 = K["stiffness_closed_opening"] * (X0["Delta_1"] - X0["x1"])
    force_2 = K["stiffness_open_closing"] * (X0["Delta_2"] - X0["x2"])
    _k_s = (force_2 - force_1) / (X0["Delta_2"] - X0["Delta_1"])
    E_I = 1 / 2 * K["stiffness_closed_opening"] * (X0["Delta_1"] - X0["x1"]) ** 2
    E_S = (
        force_1 * (X0["Delta_2"] - X0["Delta_1"])
        + 1 / 2 * _k_s * (X0["Delta_2"] - X0["Delta_1"]) ** 2
    )
    E_II = (
        force_2 * (X0["x2"] - X0["Delta_2"])
        + 1 / 2 * K["stiffness_open_closing"] * (X0["x2"] - X0["Delta_2"]) ** 2
    )

    # Closed
    if x < X0["x1"]:
        F_el = K["stiffness_closed_closing"] * (x - X0["x1"])  # x<x_1 -> F_el<0
        E_el = 1 / 2 * F_el * (x - X0["x1"])

    # Closed -> Opening
    elif X0["x1"] <= x and x < X0["Delta_1"]:
        F_el = K["stiffness_closed_opening"] * (x - X0["x1"])
        E_el = 1 / 2 * F_el * (x - X0["x1"])

    # Spinodal
    elif X0["Delta_1"] <= x and x < X0["Delta_2"]:
        E_el = (
            E_I
            + force_1 * (x - X0["Delta_1"])
            + 1 / 2 * _k_s * (x - X0["Delta_1"]) ** 2
        )

    # Open -> Closing
    elif X0["Delta_2"] <= x and x < X0["x2"]:
        E_el = (
            E_I
            + E_S
            + force_2 * (x - X0["Delta_2"])
            + 1 / 2 * K["stiffness_open_closing"] * (x - X0["Delta_2"]) ** 2
        )

    # Open
    elif x >= X0["x2"]:
        F_el = K["stiffness_open_opening"] * (x - X0["x2"])
        # E_el = -E_I + E_S + 1 / 2 * F_el * (x - X0["x2"])
        E_el = E_I + E_S + E_II + 1 / 2 * F_el * (x - X0["x2"])
    else:
        F_el = 0

    return E_el


def gas_elastic_energy_func(x, parameters):
    c_v = parameters["c_v"]
    m_gas = parameters["mass_gas"]
    R = parameters["R_specific_heat"]
    lc_0 = parameters["capsule_auxiliary_length"]
    r_in = parameters["capsule_radius"]
    p_atm = parameters["pressure_atm"]

    c1 = (math.pi * (r_in**2)) / (m_gas * R * (c_v + R))

    x_eff = x - parameters["initial_capsule_length"]

    gas_elastic_energy = (
        p_atm
        * c1
        * (
            R * (lc_0 + x_eff)
            + c_v * (lc_0 ** (1 + R / c_v)) * ((lc_0 + x_eff) ** (-R / c_v))
        )
    )

    return gas_elastic_energy * c_v * m_gas


def find_extrema_elastic_energy(
    STIFFNESSES, ELASTIC_SPRING_STIFFNESS_CHANGE_LOCATIONS, num_points=5000
):
    x2 = ELASTIC_SPRING_STIFFNESS_CHANGE_LOCATIONS["x2"]
    Delta_1 = ELASTIC_SPRING_STIFFNESS_CHANGE_LOCATIONS["Delta_1"]
    x_start = -Delta_1 / 1000
    x_end = x2 * 1.1

    x_range = np.linspace(x_start, x_end, num_points)
    elastic_energy_values = [
        elastic_energy_func(x, STIFFNESSES, ELASTIC_SPRING_STIFFNESS_CHANGE_LOCATIONS)
        for x in x_range
    ]

    peaks, _ = find_peaks(elastic_energy_values)
    minima, _ = find_peaks(-np.array(elastic_energy_values))

    maxima = [(x_range[i], elastic_energy_values[i]) for i in peaks]
    minima = [(x_range[i], elastic_energy_values[i]) for i in minima]

    return maxima, minima


def F_drag_magnetic(x, v, parameters, z_Br_eval, int_Br_squared):
    coil_length = parameters["coil_length"]

    mag_len = parameters["magnet_length"]
    d0_edges = parameters["initial_destince_from_magnet_edge_to_coil_edge"]

    # Defining intergation range
    integration_lower_limit = mag_len + d0_edges - x
    integration_upper_limit = integration_lower_limit + coil_length

    z_range = z_Br_eval
    integrated_Br_squared_values = int_Br_squared

    idx_start = np.searchsorted(z_range, integration_lower_limit)
    idx_end = np.searchsorted(z_range, integration_upper_limit)

    B_r_integration_up_to_z_start = integrated_Br_squared_values[idx_start]
    B_r_integration_up_to_z_final = integrated_Br_squared_values[idx_end]

    coil_resistance_per_unit_length = parameters["coil_resistance_per_unit_length"]

    F_drag_induced = (
        v
        / coil_resistance_per_unit_length
        * (B_r_integration_up_to_z_final - B_r_integration_up_to_z_start)
    )

    return F_drag_induced


def Br_int_over_coil(sol, parameters, z_Br_eval, int_Br, int_Br_squared):
    sim_magnet_location = sol.y[0]
    magnet_velocities = sol.y[1]
    Br_int_values = np.zeros_like(sol.y[0])  # Initialization
    v2_Br_int2_values = np.zeros_like(sol.y[0])
    v2_Br2_int_values = np.zeros_like(sol.y[0])

    coil_length = parameters["coil_length"]
    mag_len = parameters["magnet_length"]
    d0_edges = parameters["initial_destince_from_magnet_edge_to_coil_edge"]

    for i, x in enumerate(sim_magnet_location):
        integration_lower_limit = mag_len + d0_edges - x
        integration_upper_limit = integration_lower_limit + coil_length

        z_range = z_Br_eval
        integrated_Br_values = int_Br
        integrated_Br_squared_values = int_Br_squared

        idx_start = np.searchsorted(z_range, integration_lower_limit)
        idx_end = np.searchsorted(z_range, integration_upper_limit)

        B_r_integration_up_to_z_start = integrated_Br_values[idx_start]
        B_r_integration_up_to_z_final = integrated_Br_values[idx_end]

        B_r_squared_integration_up_to_z_start = integrated_Br_squared_values[idx_start]
        B_r_squared_integration_up_to_z_final = integrated_Br_squared_values[idx_end]

        Br_int_values[i] = B_r_integration_up_to_z_final - B_r_integration_up_to_z_start
        v2_Br_int2_values[i] = (
            magnet_velocities[i] ** 2
            * (B_r_integration_up_to_z_final - B_r_integration_up_to_z_start) ** 2
        )
        v2_Br2_int_values[i] = magnet_velocities[i] ** 2 * (
            B_r_squared_integration_up_to_z_final
            - B_r_squared_integration_up_to_z_start
        )

    return Br_int_values, v2_Br_int2_values, v2_Br2_int_values


def power_losses(x, v, parameters, z_Br_eval, int_Br_squared):
    coil_length = parameters["coil_length"]
    mag_len = parameters["magnet_length"]
    d0_edges = parameters["initial_destince_from_magnet_edge_to_coil_edge"]

    integration_lower_limit = mag_len + d0_edges - x
    integration_upper_limit = integration_lower_limit + coil_length

    z_range = z_Br_eval
    integrated_Br_squared_values = int_Br_squared

    idx_start = np.searchsorted(z_range, integration_lower_limit)
    idx_end = np.searchsorted(z_range, integration_upper_limit)

    B_r2_integration_up_to_z_start = integrated_Br_squared_values[idx_start]
    B_r2_integration_up_to_z_final = integrated_Br_squared_values[idx_end]

    coil_resistance_per_unit_length = parameters["coil_resistance_per_unit_length"]

    power_loss_value = (
        v**2
        / coil_resistance_per_unit_length
        * (B_r2_integration_up_to_z_final - B_r2_integration_up_to_z_start)
    )

    return power_loss_value


def external_force(x, t, force_actuation):
    t1 = force_actuation["t1"]
    F1 = force_actuation["F1"]
    dt1 = force_actuation["dt1"]
    t2 = force_actuation["t2"]
    F2 = force_actuation["F2"]
    dt2 = force_actuation["dt2"]

    F = 0
    # Setting force over time
    if t < t1:
        F = 0
    elif t < t1 + dt1:
        F = F1
    elif t < t2:
        F = 0
    elif t < t2 + dt2:
        F = F2
    else:
        F = 0
    return F


def temperature_external(t, thermal_actuation_parameters):
    actuation = thermal_actuation_parameters["actuation"]

    delay = thermal_actuation_parameters["delay"]

    if actuation == "Force":
        T_ext = thermal_actuation_parameters["T_const_baseline"]

    if actuation == "Mixed":
        T_cold = thermal_actuation_parameters["T_cold"]
        T_amb = thermal_actuation_parameters["T_amb"]
        T_hot = thermal_actuation_parameters["T_hot"]
        t_shift = thermal_actuation_parameters["t_shift"]

        T_ext = T_amb

        # Ambient -> Cold -> Hot
        if t < delay:
            T_ext = T_amb
        elif t < (delay + t_shift):
            T_ext = T_cold
        elif t < (delay + 2 * t_shift):
            T_ext = T_hot
        elif t < (delay + 3 * t_shift):
            T_ext = T_cold
        elif t < (delay + 4 * t_shift):
            T_ext = T_hot
        elif t < (delay + 5 * t_shift):
            T_ext = T_cold

    if actuation == "Thermal cycling":
        delay = thermal_actuation_parameters["delay"]
        T_avg = thermal_actuation_parameters["T_avg"]
        Amplitude = thermal_actuation_parameters["Amplitude"]
        T_period = thermal_actuation_parameters["T_period"]
        T_const_baseline = thermal_actuation_parameters["T_const_baseline"]
        T_ext = T_const_baseline
        T_amb = T_const_baseline

        if (t > delay) and (T_period != 0):
            omega = 2 * math.pi / T_period

            phase_shift = math.asin((T_const_baseline - T_avg) / Amplitude)

            T_ext = Amplitude * math.sin(omega * (t - delay) + phase_shift) + T_avg
        else:
            T_ext = T_const_baseline

    if actuation == "Thermal hot cold linear change":
        T_cold = thermal_actuation_parameters["T_cold"]
        T_amb = thermal_actuation_parameters["T_amb"]
        T_hot = thermal_actuation_parameters["T_hot"]
        t_shift = thermal_actuation_parameters["t_shift"]

        T_ext = T_amb

        if t < delay:
            T_ext = T_cold
        elif t < (delay + t_shift):
            T_ext = T_cold + (T_hot - T_cold) / t_shift * (t - delay)
        elif t < (delay + 2 * t_shift):
            T_ext = T_hot
        elif t < (delay + 3 * t_shift):
            T_ext = T_hot + (T_cold - T_hot) / t_shift * (t - (delay + 2 * t_shift))
        elif t < (delay + 4 * t_shift):
            T_ext = T_cold
        elif t < (delay + 5 * t_shift):
            T_ext = T_cold + (T_hot - T_cold) / t_shift * (t - (delay + 4 * t_shift))
        elif t < (delay + 6 * t_shift):
            T_ext = T_hot
        elif t < (delay + 7 * t_shift):
            T_ext = T_hot + (T_cold - T_hot) / t_shift * (t - (delay + 6 * t_shift))
        elif t < (delay + 8 * t_shift):
            T_ext = T_cold
        elif t < (delay + 9 * t_shift):
            T_ext = T_amb

    if (
        actuation == "Eenrgy storage when snapped down"
        or actuation == "Eenrgy storage when snapped up"
    ):
        T_cold = thermal_actuation_parameters["T_cold"]
        T_amb = thermal_actuation_parameters["T_amb"]
        T_hot = thermal_actuation_parameters["T_hot"]
        t_shift = thermal_actuation_parameters["t_shift"]

        if "T_const_baseline" in thermal_actuation_parameters:
            T_const_baseline = thermal_actuation_parameters["T_const_baseline"]
        else:
            T_const_baseline = thermal_actuation_parameters["T_amb"]

        T_ext = T_amb

        # # Ambient -> Cold -> Hot
        if t < delay:
            T_ext = T_amb
        elif t < (delay + t_shift):
            T_ext = T_cold
        elif t < (delay + 2 * t_shift):
            T_ext = T_amb

    return T_ext


def radial_magnetic_field_component_B_r_func(parameters):
    initial_magnet_location = parameters["initial_capsule_length"]

    B_r_calc_range_mean = initial_magnet_location
    B_r_calc_range_plus_minus_range = 2 * parameters["coil_length"]  # arbitrary choice

    steps = int(1e4)  # arbitrary choice
    z_values = np.linspace(
        initial_magnet_location - B_r_calc_range_plus_minus_range,
        initial_magnet_location + B_r_calc_range_plus_minus_range,
        steps,
    )

    magnet_B0 = parameters["magnet_magnetic_field_amp_B0"]
    magnet_radius = parameters["magnet_radius"]
    magnet_half_length = parameters["magnet_length"] / 2
    rho = parameters["coil_raduis_inner"]

    B_r_values = np.array(
        [
            _calculate_magnetic_force_radial_component(
                magnet_B0, magnet_radius, magnet_half_length, rho, zi
            )
            for zi in z_values
        ]
    )

    return z_values, B_r_values


def _calculate_magnetic_force_radial_component(
    magnet_polarization, magnet_radius, magnet_half_length, rho, z
):
    B_0 = magnet_polarization

    z_plus = z + magnet_half_length
    z_minus = z - magnet_half_length

    alpha_plus = magnet_radius / np.sqrt(z_plus**2 + (rho + magnet_radius) ** 2)
    alpha_minus = magnet_radius / np.sqrt(z_minus**2 + (rho + magnet_radius) ** 2)

    k_plus = np.sqrt(
        (z_plus**2 + (magnet_radius - rho) ** 2)
        / (z_plus**2 + (magnet_radius + rho) ** 2)
    )
    k_minus = np.sqrt(
        (z_minus**2 + (magnet_radius - rho) ** 2)
        / (z_minus**2 + (magnet_radius + rho) ** 2)
    )

    B_r = B_0 * (
        alpha_plus * _c_func(k_plus, 1, 1, -1)
        - alpha_minus * _c_func(k_minus, 1, 1, -1)
    )
    return B_r


def _c_func(kc, p, c, s):
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


def induced_emf(sol, parameters, z_Br_eval, int_Br):
    magnet_velocities = sol.y[1]
    sim_magnet_location = sol.y[0]
    induced_emf = np.zeros_like(sol.y[0])  # Initialization

    coil_length = parameters["coil_length"]
    r_inner = parameters["coil_raduis_inner"]
    mag_len = parameters["magnet_length"]
    d0_edges = parameters["initial_destince_from_magnet_edge_to_coil_edge"]

    for i, x in enumerate(sim_magnet_location):
        integration_lower_limit = mag_len + d0_edges - x
        integration_upper_limit = integration_lower_limit + coil_length

        z_range = z_Br_eval
        integrated_Br_values = int_Br

        idx_start = np.searchsorted(z_range, integration_lower_limit)
        idx_end = np.searchsorted(z_range, integration_upper_limit)

        B_r_integration_up_to_z_start = integrated_Br_values[idx_start]
        B_r_integration_up_to_z_final = integrated_Br_values[idx_end]

        induced_emf[i] = (
            2
            * math.pi
            * r_inner
            * magnet_velocities[i]
            * (B_r_integration_up_to_z_final - B_r_integration_up_to_z_start)
        )

    return induced_emf


def integrand_Br(z, y, parameters):
    magnet_B0 = parameters["magnet_magnetic_field_amp_B0"]
    magnet_radius = parameters["magnet_radius"]
    magnet_half_length = parameters["magnet_length"] / 2
    rho = parameters["coil_raduis_inner"]

    B_r_value = _calculate_magnetic_force_radial_component(
        magnet_B0, magnet_radius, magnet_half_length, rho, z
    )

    return B_r_value / magnet_B0


def integrand_Br_squared(z, y, parameters):
    magnet_B0 = parameters["magnet_magnetic_field_amp_B0"]
    magnet_radius = parameters["magnet_radius"]
    magnet_half_length = parameters["magnet_length"] / 2
    rho = parameters["coil_raduis_inner"]

    B_r_value = _calculate_magnetic_force_radial_component(
        magnet_B0, magnet_radius, magnet_half_length, rho, z
    )

    B_r_value_squared = B_r_value**2

    return B_r_value_squared / (magnet_B0**2)


def integrate_Br_and_Br_squared(distance_range_B_r, parameters):
    z_initial = distance_range_B_r[0]
    z_final = distance_range_B_r[-1]

    y0 = [0]  # initialization, The integral at z_initial equals zero
    bounds = (z_initial, z_final)
    num_z_samples = int(1e4)

    z_eval = np.linspace(z_initial, z_final, num_z_samples)

    sol_Br = solve_ivp(
        integrand_Br,
        bounds,
        y0,
        args=(parameters,),
        t_eval=z_eval,
        atol=1e-8,
        rtol=1e-8,
    )

    sol_Br_squared = solve_ivp(
        integrand_Br_squared,
        bounds,
        y0,
        args=(parameters,),
        t_eval=z_eval,
        atol=1e-8,
        rtol=1e-8,
    )

    return (
        sol_Br.t,
        sol_Br.y[0] * parameters["magnet_magnetic_field_amp_B0"],
        sol_Br_squared.y[0] * (parameters["magnet_magnetic_field_amp_B0"] ** 2),
    )


stiffnesses = {
    "stiffness_closed_closing": 230.14 * 1000,  # [N/m]
    # "stiffness_closed_opening": 16.81 * 1000,  # [N/m]
    "stiffness_closed_opening": 5 * 1000,  # [N/m]
    # "stiffness_spinodal": -7.08 * 1000,  # [N/m]
    # "stiffness_open_closing": 15.43 * 1000,  # [N/m]
    "stiffness_open_closing": 3 * 1000,  # [N/m]
    "stiffness_open_opening": 24.77 * 1000,  # [N/m]
}


elastic_spring_stiffness_change_locations = {
    "x1": 0,  # [mm] By definition
    "Delta_1": 0.35 / 1000,  # [mm]
    "Delta_2": (0.35 + 2.83 - (0.52 + 0.3)) / 1000,  # 2.36 [mm]
    "x2": (0.35 + 2.83 - 0.3) / 1000,  # 2.88 [mm]
}

# System parameters
magnet_radius = 12.7 / 1000  # [m]
magnet_length = 3.4 / 1000  # [m]
magnet_magnetic_field_amp_B0 = 1.2  # [T]
magnet_mass = 12 / 1000  # Mass [kg]

coil_raduis_inner = 1.05 * magnet_radius  # [m]
coil_length = 65 / 1000  # [m]
coil_resistance = 5 * 1e-4  # [Ohm]

coil_resistance_per_unit_length = coil_resistance / coil_length

coil_one_loop_length = 2 * math.pi * coil_raduis_inner
wire_diameter = 1 / 1000  # [m]

initial_destince_from_magnet_edge_to_coil_edge = -1 / 1000
initial_distance_from_magnet_center_to_coil_center = (
    coil_length / 2 + magnet_length / 2 + initial_destince_from_magnet_edge_to_coil_edge
)

initial_capsule_length = 1.22 / 1000  # [mm]

capsule_radius = 11.1 / 1000  # [mm]
capsule_side_area = math.pi * ((capsule_radius) ** 2)  # Capsule side area [mm^2]
capsule_auxiliary_length = 30 / 1000
capsule_volume_initial = (
    math.pi
    * ((capsule_radius) ** 2)
    * (initial_capsule_length + capsule_auxiliary_length)
)

pressure_atm = 101325  # Atmospheric pressure
R_specific_heat = 287  # Gas constant [J/(kg·K)] air
heat_transfer_coefficient = 10  # Heat transfer coefficient [W/(m^2 K)]
c_v = 718  # Specific heat at constant volume - air [J/(kg*K)]
air_density = 1.2  # [kg/m^3]
mass_gas = (
    capsule_volume_initial * air_density
)  # Mass of the gas [kg], based on the volume of a fruse and air desnsity (1.2 [kg/m^3])
sealing_temp = pressure_atm * capsule_volume_initial / (mass_gas * R_specific_heat)

drag_coeff = 0  # Damping coefficient, no viscous damping !!!

## Choose actuation mode
# actuation = "Force"
# actuation = "Thermal cycling"
# actuation = "Mixed"
# actuation = "Thermal hot cold linear change"  # Fig 4
# actuation = "Eenrgy storage when snapped down"
actuation = "Eenrgy storage when snapped up"  # Fig 3


if actuation == "Force":
    force_actuation = {
        "t1": 2.5,  # [s]
        "F1": 3.8,  # [N]
        "dt1": 0.02,  # [s]
        "t2": 5,  # [s]
        "F2": -4.9,  # [N]
        "dt2": 0.12,  # [s]
    }

    t_final = 8  # [s]

    thermal_actuation_parameters = {
        "actuation": actuation,
        "T_const_baseline": 20 + 273,  # [K]
        "delay": 0,  # [s]
        "T_avg": 0,  # [K]
        "Amplitude": 0,  # [K]
        "T_period": 0,  # [s]
    }

    initial_capsule_length = 1.23 / 1000  # [mm]

if actuation == "Thermal cycling":
    force_actuation = {
        "t1": 2.5,  # [s]
        "F1": 0,  # [N]
        "dt1": 0.02,  # [s]
        "t2": 5,  # [s]
        "F2": 0,  # [N]
        "dt2": 0.13,  # [s]
    }

    t_final = 40  # [s]

    thermal_actuation_parameters = {
        "actuation": actuation,
        "T_const_baseline": 20 + 273,  # [K]
        "delay": 2,  # [s]
        "T_avg": 22.5 + 273,  # [K]
        "Amplitude": 8,  # [K]
        "T_period": 10,  # [s] # For soft capsule
    }

    initial_capsule_length = 2 / 1000  # [mm] # Soft capsule

if actuation == "Mixed":
    force_actuation = {
        "t1": 9,  # [s]
        "F1": 1,  # [N]
        "dt1": 0.01,  # [s]
        "t2": 2 + 4 + 4 + 3,  # [s]
        "F2": 0,  # [N]
        "dt2": 0.5,  # [s]
    }

    t_final = 11  # [s]

    thermal_actuation_parameters = {
        "actuation": actuation,
        "delay": 2,  # [s]
        "T_hot": 55 + 273,  # [K]
        "T_cold": -40 + 273,  # [K]
        "T_amb": 20 + 273,  # [K]
        "t_shift": 4,  # [s]
    }
    initial_capsule_length = 1.23 / 1000  # [mm]

if actuation == "Thermal hot cold linear change":
    force_actuation = {
        "t1": 2 + 1,  # [s]
        "F1": 0,  # [N]
        "dt1": 0.5,  # [s]
        "t2": 2 + 4 + 4 + 3,  # [s]
        "F2": 0,  # [N]
        "dt2": 0.5,  # [s]
    }

    t_final = 40  # [s]

    initial_capsule_length = 2.3 / 1000  # [mm]

    thermal_actuation_parameters = {
        "actuation": actuation,
        "delay": 2,  # [s]
        "T_hot": 30 + 273,  # [K]
        "T_cold": 19 + 273,  # [K]
        "T_amb": 24 + 273,  # [K]
        "t_shift": 4,  # [s]
    }

if actuation == "Eenrgy storage when snapped down":
    force_actuation = {
        "t1": 11,  # [s]
        "F1": 5,  # [N]
        "dt1": 0.05,  # [s]
        "t2": 2 + 4 + 4 + 3,  # [s]
        "F2": 0,  # [N]
        "dt2": 0.5,  # [s]
    }

    t_final = 15  # [s]

    thermal_actuation_parameters = {
        "actuation": actuation,
        "delay": 2,  # [s]
        "T_hot": 55 + 273,  # [K]
        "T_cold": -35 + 273,  # [K]
        "T_amb": 20 + 273,  # [K]
        "t_shift": 5,  # [s]
    }
    initial_capsule_length = 1.23 / 1000  # [mm]

if actuation == "Eenrgy storage when snapped up":
    force_actuation = {
        "t1": 11,  # [s]
        "F1": -2,  # [N]
        "dt1": 0.05,  # [s]
        "t2": 2 + 4 + 4 + 3,  # [s]
        "F2": 0,  # [N]
        "dt2": 0.5,  # [s]
    }

    t_final = 15  # [s]

    thermal_actuation_parameters = {
        "actuation": actuation,
        "delay": 2,  # [s]
        "T_hot": 0 + 273,  # [K]
        "T_cold": 30 + 273,  # [K]
        "T_amb": 20 + 273,  # [K]
        "t_shift": 5,  # [s]
    }
    initial_capsule_length = 2.3 / 1000  # [mm]


parameters = {
    "magnet_radius": magnet_radius,
    "magnet_length": magnet_length,
    "magnet_magnetic_field_amp_B0": magnet_magnetic_field_amp_B0,
    "magnet_mass": magnet_mass,
    "coil_raduis_inner": coil_raduis_inner,
    "coil_length": coil_length,
    "coil_resistance": coil_resistance,
    "coil_resistance_per_unit_length": coil_resistance_per_unit_length,
    "initial_distance_from_magnet_center_to_coil_center": initial_distance_from_magnet_center_to_coil_center,
    "initial_destince_from_magnet_edge_to_coil_edge": initial_destince_from_magnet_edge_to_coil_edge,
    "initial_capsule_length": initial_capsule_length,
    "capsule_radius": capsule_radius,
    "capsule_auxiliary_length": capsule_auxiliary_length,
    "capsule_side_area": capsule_side_area,
    "pressure_atm": pressure_atm,
    "R_specific_heat": R_specific_heat,
    "h": heat_transfer_coefficient,
    "c_v": c_v,
    "mass_gas": mass_gas,
    "drag_coeff": drag_coeff,
    "sealing_temp": sealing_temp,
}

distance_range_B_r, radial_magnetic_field_component_B_r_values = (
    radial_magnetic_field_component_B_r_func(parameters)
)
z_Br_eval, int_Br, int_Br_squared = integrate_Br_and_Br_squared(
    distance_range_B_r, parameters
)

# Initial conditions
x0 = initial_capsule_length  # Initial displacement [m]
v0 = 0  # Initial velocity [m/s]
T0 = 20 + 273  # Initial temperature [K]
s_entropy_t0 = 1  # Initial entropy
E_losses0 = 0
W_F_ext0 = 0
Int_Ugas0 = 0
y0 = [
    x0,
    v0,
    T0,
    s_entropy_t0,
    E_losses0,
    W_F_ext0,
    Int_Ugas0,
]

t_span = (0, t_final)
num_sim_smaples = int(5000)
t_eval = np.linspace(t_span[0], t_span[1], num_sim_smaples)

# Solve the set of differential equations
sol = solve_ivp(
    spring_dynamics,
    t_span,
    y0,
    args=(
        external_force,
        F_drag_magnetic,
        gas_force,
        elastic_force_func,
        temperature_external,
        stiffnesses,
        elastic_spring_stiffness_change_locations,
        parameters,
        force_actuation,
        distance_range_B_r,
        radial_magnetic_field_component_B_r_values,
        z_Br_eval,
        int_Br,
        int_Br_squared,
        thermal_actuation_parameters,
    ),
    t_eval=t_eval,
    atol=1e-6,
    rtol=1e-6,
)

# Extracting the results
t = sol.t
x = sol.y[0]
v = sol.y[1]
T_in = sol.y[2]
s_entropy = sol.y[3]
E_losses = sol.y[4]
W_F_ext = sol.y[5]
Int_Ugas = sol.y[6]
T_ext = [temperature_external(ti, thermal_actuation_parameters) for ti in t]


# Calculate forces over time
external_forces = np.array(
    [external_force(xi, ti, force_actuation) for xi, ti in zip(x, t)]
)
gas_forces = np.array(
    [
        gas_force(xi, Ti, parameters) - (capsule_side_area * pressure_atm)
        for xi, Ti in zip(x, T_in)
    ]
)
elastic_forces = np.array(
    [
        elastic_force_func(xi, stiffnesses, elastic_spring_stiffness_change_locations)
        for xi in x
    ]
)
magnetic_drag_force = np.array(
    [
        F_drag_magnetic(xi, vi, parameters, z_Br_eval, int_Br_squared)
        for xi, vi in zip(x, v)
    ]
)

gas_internal_energy = np.array([c_v * mass_gas * T_in_i for T_in_i in T_in])
elastic_energy = np.array(
    [
        elastic_energy_func(xi, stiffnesses, elastic_spring_stiffness_change_locations)
        for xi in x
    ]
)
kinetic_energy = np.array([1 / 2 * magnet_mass * v_i for v_i in v])

gas_work = np.array(
    [
        (gas_force(xi, Ti, parameters) / capsule_side_area)
        * (np.pi * parameters["capsule_radius"] ** 2 * (xi - initial_capsule_length))
        for xi, Ti in zip(x, T_in)
    ]
)

gas_enthalpy = np.array(
    [
        c_v * mass_gas * T_in_i
        + (gas_force(xi, T_in_i, parameters) / capsule_side_area)
        * (np.pi * parameters["capsule_radius"] ** 2 * (xi - initial_capsule_length))
        for xi, T_in_i in zip(x, T_in)
    ]
)

# Energy loss, power and induced EMF
power_loss = power_losses(x, v, parameters, z_Br_eval, int_Br_squared)
induced_emf_values = np.sqrt(power_loss * coil_resistance)
induced_current = np.sqrt(power_loss / coil_resistance)
R_heater = coil_resistance * 2

power_inner_loss = power_loss * coil_resistance / (coil_resistance + R_heater)
power_heater = power_loss * R_heater / (coil_resistance + R_heater)

E_inner_loss = E_losses * coil_resistance / (coil_resistance + R_heater)
E_heater = E_losses * R_heater / (coil_resistance + R_heater)

# %%
# Normalized dynamics results

from matplotlib.patches import Rectangle
import os


def save_image(figname, saving_location):
    # Ensure directory exists
    os.makedirs(saving_location, exist_ok=True)
    figname = f"{figname}.jpg"
    filepath = os.path.join(saving_location, figname)
    plt.savefig(
        filepath,
        bbox_inches="tight",
        pad_inches=0.05,
        dpi=1000,
    )


def add_grey_background_delay(fig, ax):
    for i in range(len(ax)):
        ax[i].set_xlim(left=t[0], right=t[-1])
        ax[i].add_patch(
            Rectangle(
                (t[0], min(ax[i].get_ylim())),
                delay,
                max(ax[i].get_ylim()) - min(ax[i].get_ylim()),
                color="grey",
                alpha=0.5,
                zorder=-1,
            )
        )


def storage_subfig1_temp(fig, ax):
    if "T_const_baseline" in thermal_actuation_parameters:
        T_avg = thermal_actuation_parameters["T_const_baseline"] - 273
    else:
        T_avg = thermal_actuation_parameters["T_amb"] - 273

    T_avg = 1  # No temp normalization

    p11 = ax[index_ax].plot(
        t,
        (T_in - 273) / T_avg,
        color="tab:red",
        linewidth="1",
        label="$T_{in}$",
        linestyle="-",
    )

    p12 = ax[index_ax].plot(
        t,
        np.array(
            [temperature_external(ti, thermal_actuation_parameters) - 273 for ti in t]
        )
        / T_avg,
        color="black",
        label="$T_{ext}$",
        linestyle="-",
    )

    ax[index_ax].set_ylabel("Temperature [°C]")

    ext_value = max(
        abs(max(np.abs(ax[index_ax].get_ylim()))),
        abs(max(np.abs(ax[index_ax].get_ylim()))),
    )

    ax[index_ax].legend(loc="upper right")

    ax[index_ax].set_xlim(left=t[0], right=t[-1])


def storage_subfig2_forces(fig, ax):
    F_max = np.max(np.abs(np.hstack([-gas_forces, external_forces])))

    ax[index_ax].plot(
        t,
        elastic_forces / F_max,
        "-",
        color="tab:blue",
        linewidth=1,
        label="$F_{el}$",
        zorder=-1,
    )
    ax[index_ax].plot(
        t,
        -gas_forces / F_max,
        "-",
        color="red",
        linewidth=1,
        label="$F_{gas}$",
        zorder=-2,
    )

    # F_ext plotted only if any elements are non-zero
    if np.any(external_forces):
        ax[index_ax].plot(
            t, external_forces / F_max, color="black", label="$F_{ext}$", zorder=1
        )

    ax[index_ax].set_ylabel("Forces/$F_{max}$")
    ax[index_ax].legend(loc="upper right")
    ylim_padding = 0.05

    ax[index_ax].set_ylim([-1, 1])
    ax[index_ax].set_xlim(left=t[0])


def storage_subfig3_energy(fig, ax):
    E_max = np.max(
        np.abs(
            np.hstack(
                [
                    gas_internal_energy - np.mean(gas_internal_energy),
                    elastic_energy * 100,
                    kinetic_energy * 10,
                ]
            )
        )
    )
    E_max_tot = np.max(np.abs(gas_internal_energy + elastic_energy + kinetic_energy))

    h_max = np.max(np.abs(gas_enthalpy))

    # Total energy
    ax[index_ax].plot(
        t,
        (gas_internal_energy + elastic_energy + kinetic_energy) / E_max_tot,
        linestyle="--",
        color="black",
        linewidth=2,
        label="$E_{tot}$",
        zorder=-1,
    )

    gas_internal_energy_var = gas_internal_energy - np.mean(gas_internal_energy)

    U_g_norm = (
        np.max(gas_internal_energy_var) - np.abs(np.min(gas_internal_energy_var))
    ) / 2

    ax[index_ax].plot(
        sol.t,
        (gas_internal_energy_var - U_g_norm)
        / (np.max(np.abs(gas_internal_energy_var - U_g_norm))),
        label="$U_{g}$",
        color="tab:red",
    )

    E_el_norm = (np.max(elastic_energy) - np.min(elastic_energy)) / 2

    # E_el
    ax[index_ax].plot(
        t,
        (elastic_energy - E_el_norm) / np.max(np.abs(elastic_energy - E_el_norm)),
        label="$E_{el}$",
        color="tab:blue",
    )

    h_var = gas_enthalpy - np.mean(gas_enthalpy)

    h_norm = (np.max(h_var) - np.abs(np.min(h_var))) / 2

    ax[index_ax].set_ylim(
        [-min(np.abs(ax[index_ax].get_ylim())), max(np.abs(ax[index_ax].get_ylim()))]
    )

    ax[index_ax].set_ylabel("Energy/$E_{max}$")
    ax[index_ax].legend(loc="upper right")

    ax[index_ax].set_xlim(left=t[0], right=t[-1])


def storage_subfig4_displacement_velocity(fig, ax):
    disp_norm = (
        np.abs(
            elastic_spring_stiffness_change_locations["x2"]
            - elastic_spring_stiffness_change_locations["x1"]
        )
        * 1000
    )

    velocity_max = E_max = np.max(np.abs(v)) * 1000

    p1 = ax[index_ax].plot(
        t,
        x * 1000 / disp_norm,
        label="$x$",
        color="tab:blue",
    )

    # x_open horizontal line
    p1a = ax[index_ax].plot(
        sol.t,
        np.ones_like(sol.t)
        * elastic_spring_stiffness_change_locations["x2"]
        * 1000
        / disp_norm,
        color="tab:blue",
        linestyle=":",
        linewidth=1,
    )

    # x_closed horizontal line
    ax[index_ax].plot(
        sol.t,
        np.ones_like(sol.t)
        * elastic_spring_stiffness_change_locations["x1"]
        * 1000
        / disp_norm,
        color="tab:blue",
        linestyle=":",
        linewidth=1,
    )

    y_padding = 1
    ax[index_ax].set_ylabel("Length")
    ax[index_ax].set_ylim(
        [
            min(ax[index_ax].get_ylim()),
            y_padding * max(np.abs(ax[index_ax].get_ylim())),
        ]
    )
    ax[index_ax].set_yticks([0, 1])
    ax[index_ax].set_yticklabels([r"$x_{c}$", r"$x_{o}$"])

    ax2_twinned = ax[index_ax].twinx()
    p2 = ax2_twinned.plot(
        t,
        v * 1000 / velocity_max,
        color="black",
        linestyle="-",
        linewidth=0.5,
        label="$v/v_{max}$",
    )
    ax2_twinned.set_ylim(
        [-max(np.abs(ax2_twinned.get_ylim())), max(np.abs(ax2_twinned.get_ylim()))]
    )
    ax2_twinned.set_ylabel("Velocity")

    lns = p1 + p2
    p_labels = [l.get_label() for l in lns]
    ax2_twinned.legend(lns, p_labels, loc="upper right")

    ax2_twinned.set_xlim(left=t[0], right=t[-1])


def harvesting_subfig5_emf_battery(fig, ax):
    emf_end_cycle = t_final

    delay_emf_index = np.searchsorted(t, delay)
    emf_end_cycle_index = np.searchsorted(t, emf_end_cycle)

    E_heater_0 = E_heater[delay_emf_index]
    E_heater_end_cycle = E_heater[emf_end_cycle_index]
    E_emf_gained_during_cycle = E_heater_end_cycle - E_heater_0

    ax[index_ax].plot(
        sol.t,
        induced_emf_values / np.max(np.abs(induced_emf_values)),
        color="black",
        linestyle="-",
        linewidth="0.5",
        label="$|E_{emf}|$",
    )

    ax[index_ax].plot(
        t[delay_emf_index:emf_end_cycle_index],
        (E_heater[delay_emf_index:emf_end_cycle_index] - E_heater_0)
        / np.max(np.abs(E_heater[delay_emf_index:emf_end_cycle_index] - E_heater_0)),
        label="$E_{stored}$",
        color="tab:green",
        zorder=-1,
    )

    ax[index_ax].set_ylabel("Energy/$E_{max}$")
    ax[index_ax].legend(loc="upper right")


fig, ax = plt.subplots(5, 1, figsize=[7, 8], sharex=True)

ax[-1].set_xlabel("Time [s]")

delay = 2
index_ax = 0

storage_subfig1_temp(fig, ax)
index_ax = index_ax + 1
storage_subfig2_forces(fig, ax)
index_ax = index_ax + 1
storage_subfig4_displacement_velocity(fig, ax)
index_ax = index_ax + 1
storage_subfig3_energy(fig, ax)
index_ax = index_ax + 1
harvesting_subfig5_emf_battery(fig, ax)
index_ax = index_ax + 1

add_grey_background_delay(fig, ax)
fig.align_ylabels(ax)
fig.tight_layout()


# %%
# p-T and s-T diagrams

emf_end_cycle = 15

delay_emf_index = np.searchsorted(t, delay)
emf_end_cycle_index = np.searchsorted(t, emf_end_cycle)

sliced_initial_transient_indices = delay_emf_index
sliced_initial_transient_indices_end = emf_end_cycle_index

time = np.arange(len(t))

# Time normalization for color mapping
time_sliced = t[sliced_initial_transient_indices:sliced_initial_transient_indices_end]
time_norm = (
    time[sliced_initial_transient_indices:] - time[sliced_initial_transient_indices]
) / (
    time[sliced_initial_transient_indices_end - 1]
    - time[sliced_initial_transient_indices]
)

p_gas_slice = gas_forces[
    sliced_initial_transient_indices:sliced_initial_transient_indices_end
]
V_slice = (
    x[sliced_initial_transient_indices:sliced_initial_transient_indices_end]
    + capsule_auxiliary_length
)

s_entropy_slice = s_entropy[sliced_initial_transient_indices:]
h_enthalpy_slice = gas_enthalpy[sliced_initial_transient_indices:]
T_in_slice = T_in[sliced_initial_transient_indices:] - 273

cmap = plt.get_cmap("winter")
colors = cmap(time_norm)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

for i in range(len(time_sliced) - 1):
    if i == 1:
        ax1.scatter(
            T_in_slice[i : i + 2],
            p_gas_slice[i : i + 2] / p_gas_slice[0],
            100,
            color="tab:green",
            marker=">",
            edgecolors="black",
            zorder=10,
        )
        ax2.scatter(
            T_in_slice[i : i + 2],
            (s_entropy_slice[i : i + 2] - s_entropy_slice[0]) * 1000,
            100,
            color="tab:green",
            marker=">",
            edgecolors="black",
            zorder=10,
        )
    if i == len(p_gas_slice) - 2:
        ax1.scatter(
            T_in_slice[i : i + 2],
            p_gas_slice[i : i + 2] / p_gas_slice[0],
            100,
            color="tab:red",
            marker="s",
            edgecolors="black",
        )
        ax2.scatter(
            T_in_slice[i : i + 2],
            (s_entropy_slice[i : i + 2] - s_entropy_slice[0]) * 1000,
            100,
            color="tab:red",
            marker="s",
            edgecolors="black",
        )
    ax1.plot(
        T_in_slice[i : i + 2],
        p_gas_slice[i : i + 2],
        color=colors[i],
        linestyle=":",
    )
    ax2.plot(
        T_in_slice[i : i + 2],
        (s_entropy_slice[i : i + 2] - s_entropy_slice[0]) * 1000,
        color=colors[i],
        linestyle=":",
    )

ax1.set_xlabel("Temperature [°C]")
ax1.set_ylabel("Gas pressure, $p_{gas}/p_{0}$")

ax2.set_xlabel("Temperature [°C]")
ax2.set_ylabel("Enthalpy, $h/h_{0}$")
ax2.set_ylabel("Enthropy, $s-s_{0}$")
ax2.locator_params(axis="y", nbins=4)


from mpl_toolkits.axes_grid1 import make_axes_locatable

divider1 = make_axes_locatable(ax1)
divider2 = make_axes_locatable(ax2)

# Shared colorbar outside the plot area to the right
cax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=plt.Normalize(
        vmin=t[sliced_initial_transient_indices],
        vmax=t[sliced_initial_transient_indices_end],
    ),
)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cax)
cbar.set_label("Time [s]")
cbar.locator = plt.MaxNLocator(nbins=5, integer=True)

plt.tight_layout(rect=[0, 0, 0.9, 1])
