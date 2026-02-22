import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


BINDING_ENERGIES = {
    "Ne": 21.564,
    "Ar": 15.759,
    "Kr": 14.000,
    "Xe": 12.130,
}


def photon_energy_from_wavelength(wavelength_nm):
    hc = 1239.841984  # eV·nm
    return hc / wavelength_nm

def calibration_function(t, D, t0, E0):
    return (D**2) / ((t - t0)**2) + E0


def fit_from_inputs(
    wavelength_nm,
    gas,
    harmonic_orders,
    tof_values,
    initial_guess=(3000, 0, 0),
    plot_output="CalibrationFit.png",
):

    E_photon = photon_energy_from_wavelength(wavelength_nm)
    print(f"\nPhoton energy = {E_photon:.6f} eV")

    if gas not in BINDING_ENERGIES:
        raise ValueError(f"Unknown gas '{gas}'. Add it to BINDING_ENERGIES.")

    E_bind = BINDING_ENERGIES[gas]
    print(f"Binding energy of {gas}: {E_bind:.6f} eV\n")


    harmonic_orders = np.array(harmonic_orders)
    E_kin = harmonic_orders * E_photon - E_bind

    print("Computed kinetic energies:")
    for n, Ek in zip(harmonic_orders, E_kin):
        print(f"  n={n:2d} → E_kin={Ek:.6f} eV")


    print("\nFitting curve...\n")

    t_vals = np.array(tof_values)
    popt, pcov = curve_fit(
        calibration_function,
        t_vals,
        E_kin,
        p0=initial_guess,
        maxfev=20000
    )

    D, t0, E0 = popt

    print("===== Fit Results (SciPy curve_fit) =====")
    print(f"D  = {D:.6f}")
    print(f"t0 = {t0:.6f}")
    print(f"E0 = {E0:.6f}")


    perr = np.sqrt(np.diag(pcov))
    print("\nParameter uncertainties (1σ):")
    print(f"σ_D  = {perr[0]:.6f}")
    print(f"σ_t0 = {perr[1]:.6f}")
    print(f"σ_E0 = {perr[2]:.6f}")


    t_fit = np.linspace(min(t_vals)*0.9, max(t_vals)*1.1, 2000)
    E_fit = calibration_function(t_fit, D, t0, E0)

    plt.figure(figsize=(8, 5))
    plt.scatter(t_vals, E_kin, s=70, label="Calculated E_kin")
    plt.plot(t_fit, E_fit, linewidth=2.5, label="Fit")
    plt.xlabel("Time-of-flight (a.u.)", fontsize=14)
    plt.ylabel("Kinetic Energy (eV)", fontsize=14)
    #plt.title("Calibration Curve Fit", fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_output, dpi=200)
    plt.show()

    return D, t0, E0


if __name__ == "__main__":
    wavelength_nm = 815                         
    gas = "Ar"                                   
    harmonic_orders = [13, 15, 17, 19, 21, 23, 25]       
    tof_values = np.array([4.60, 3.59, 3.05, 2.72, 2.47, 2.28, 2.13])  #  TOF points
    tof_values *= 10**3
    fit_from_inputs(
        wavelength_nm,
        gas,
        harmonic_orders,
        tof_values,
        initial_guess=(10000, 0, 0),
        plot_output="CalibrationFit.png"
    )
