import numpy as np
import matplotlib
matplotlib.use("QtAgg")  # or "Qt5Agg", etc.
matplotlib.rcParams["mathtext.fontset"] = "cm"     # Computer Modern
matplotlib.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt

def TofToEnergy(tof, intensity, L0, T0=0.0, E0=0.0):
    tof = np.asarray(tof)
    intensity = np.asarray(intensity)
    """
    # Avoid division near singularity
    mask = tof > T0 + 1e-12
    tof = tof[mask]
    intensity = intensity[mask]
    """
    # Energy conversion
    E = (L0**2) / ((tof - T0)**2) + E0

    #  JACOBIAN CORRECTION 
    jacobian = np.abs((tof - T0)**3 / (2 * L0**2))
    intensity_E = intensity * jacobian
    
    # Sort so energy increases
    idx = np.argsort(E)
    return E[idx], intensity_E[idx]








def hybrid_rebin_energy(E, Y, E_grid):
    """
    Hybrid histogram–interpolation rebinning.
    Conserves intensity while producing smooth,
    uniformly spaced energy spectra.
    """

    # local spacing (irregular grid)
    dE_local = np.gradient(E, edge_order=2)

    #  convert density -> integrated contribution
    weights = Y * dE_local

    # cumulative integral
    cumulative = np.cumsum(weights)

    # ensure starts at zero (important)
    cumulative = cumulative - cumulative[0]

    # interpolate cumulative function
    cumulative_interp = np.interp(
        E_grid,
        E,
        cumulative,
        left=0,
        right=cumulative[-1]
    )

    #  differentiate back to density
    Y_uniform = np.gradient(cumulative_interp, E_grid)

    return Y_uniform










def AllEnergy(df, L0, T0=0.0, E0=0.0, dE=0.05):
    """
    Convert multiple TOF spectra into energy spectra
    using UNIFORM ENERGY BIN WIDTH (dE in eV).

    Parameters
    ----------
    df : pandas.DataFrame
        Columns: ["time", "intensity", "position"]
    L0, T0, E0 : float
        Calibration constants
    dE : float
        Desired energy bin width in eV

    Returns
    -------
    E_grid : np.ndarray
        Uniformly spaced energy axis
    Y_matrix : np.ndarray
        2D array (len(E_grid), n_positions)
    time_delays_fs : np.ndarray
        Time delay axis centered at delay-zero
    """

    import numpy as np

    positions = np.sort(df["position"].unique())
    dt = 1e-3  # user scaling for TOF

    spectra = []
    E_min_global = np.inf
    E_max_global = -np.inf


    # Convert each position to energy space
    for pos in positions:
        subset = df[df["position"] == pos]
        E, Y = TofToEnergy(subset["time"] * dt,
                           subset["intensity"],
                           L0, T0, E0)

        spectra.append((E, Y))

        E_min_global = min(E_min_global, E.min())
        E_max_global = max(E_max_global, E.max())

    # Create UNIFORM energy grid
    E_grid = np.arange(E_min_global, E_max_global, dE)
    

    # Hybrid rebinning onto uniform energy grid
    Y_list = []

    for E, Y in spectra:
        Y_uniform = hybrid_rebin_energy(E, Y, E_grid)
        Y_list.append(Y_uniform)

    Y_matrix = np.stack(Y_list, axis=1)
    


    # Convert positions -> time delay (fs)
    nm_to_fs = 0.1395 * 1e-5
    time_delays_fs = positions * nm_to_fs

    

    # Determine delay-zero
    total_intensity = Y_matrix.sum(axis=0)
    idx_max = np.argmax(total_intensity)
    delay_zero = time_delays_fs[idx_max]
    time_delays_fs = time_delays_fs - delay_zero

    # Normalize
    Y_matrix /= Y_matrix.max()

    return E_grid, Y_matrix, time_delays_fs




def PlotSurface(
    E, Y_matrix, positions,
    output_file="SurfacePlot.png",
    cmap="jet",
    clim=None,
    ylim=None,
    elev=45,
    azim=-30,
    integrate=False,
    pos_limit=None,
    td_plot=True,
    pd_plot=False
):
    """
    Generate both a 3D surface plot and a 2D colormap (intensity map)
    from the same dataset.

    Parameters
    ----------
    E : np.ndarray
        Energy scale (eV), shape (nE,).
    Y_matrix : np.ndarray
        2D array (nE, nPositions).
    positions : np.ndarray
        Position values corresponding to each column in Y_matrix.
    output_file : str
        File name for saving the 3D figure. The 2D version will be saved as
        output_file with "_2D" appended.
    cmap : str, optional
        Colormap name.
    clim : tuple, optional
        (vmin, vmax) color scale limits.
    ylim : tuple, optional
        (ymin, ymax) energy clipping range (eV).
    elev, azim : float
        3D view angles.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize

    # Ensure numpy arrays
    E = np.asarray(E)
    Y_matrix = np.asarray(Y_matrix)
    positions = np.asarray(positions)

    if pos_limit is not None:
        pmin, pmax = pos_limit
        pos_mask = (positions >= pmin) & (positions <= pmax)
        if not np.any(pos_mask):
            raise ValueError("pos_limit removed all data!")
        positions = positions[pos_mask]
        Y = Y_matrix[:, pos_mask]
        Y_matrix = Y.copy()   # maintain original variable name



    # Clip to energy limits if requested
    if ylim is not None:
        ymin, ymax = ylim
        mask = (E >= ymin) & (E <= ymax)
        if not np.any(mask):
            raise ValueError(f"No energy points remain after applying ylim={ylim}")
        E_plot = E[mask]
        Y_plot = Y_matrix[mask, :]
    else:
        E_plot = E
        Y_plot = Y_matrix

    if integrate:
        Y_sum = np.sum(Y_plot, axis=0, keepdims=True)
        Y_plot = np.broadcast_to(Y_sum, Y_plot.shape)

    # Create meshgrid
    P, E_grid = np.meshgrid(positions, E_plot)

    # Determine normalization
    vmin = Y_plot.min() if clim is None else clim[0]
    vmax = Y_plot.max() if clim is None else clim[1]
    norm = Normalize(vmin=vmin, vmax=vmax)

    # FIGURE 1 — 3D Surface Plot
    if td_plot:
        fig = plt.figure(figsize=(10, 6), facecolor=None)
        ax = fig.add_subplot(111, projection="3d")

        ax.set_axis_off()
        surf = ax.plot_surface(
            P,2*E_grid, Y_plot,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1
            #norm=norm
        )

        #ax.set_xlabel("Time delay (fs)")
        #ax.set_ylabel("Energy (eV)")
        #ax.set_zlabel("Intensity")
        #ax.set_title("3D Surface Plot of Energy Spectra")

        if ylim is not None:
            #ax.set_ylim(*ylim)
            ax.set_ylim(8,24)
        if clim is not None:
            ax.set_zlim(*clim)

        #fig.colorbar(surf, shrink=0.6, pad=0.1, label="Intensity")

        ax.view_init(elev=elev, azim=azim)

        plt.grid(False)
        plt.tight_layout()
        plt.savefig(output_file, dpi=500, transparent=True)
        plt.close(fig)
    
    # FIGURE 2 — 2D Colormap
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im = ax2.pcolormesh(
        positions, E_plot, Y_plot,
        shading='auto',
        cmap=cmap,
        #norm=norm
    )

    ax2.set_xlabel("Time delay (fs)")
    ax2.set_ylabel("Energy (eV)")
    ax2.set_title("2D Intensity Map")

    if ylim is not None:
        ax2.set_ylim(*ylim)

    fig2.colorbar(im, ax=ax2, label="Intensity")

    plt.tight_layout()
    plt.savefig(output_file.replace(".png", "_2D.png"), dpi=500)
    plt.close(fig2)


    if pd_plot:
        plt.figure()

        Y_sum = np.mean(Y_plot, axis=0)

        plt.plot(positions, Y_sum, color="k", lw=3)

        plt.xlabel("Time delay (fs)")
        plt.ylabel("Energy")
        plt.title(f"Mean energy in eV range {ylim}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file.replace(".png","_sideview.png"), dpi=500)
        plt.close()
    


from scipy.optimize import curve_fit

def sine_func(x, A, phi, offset, omega):
    return A * np.cos(omega * x - phi) + offset



def CurveFit(
    E, Y_matrix, positions,
    omega,
    output_file="CurveFit.png",
    ylim=None,
    pos_limit=None,
    ax=None,                 # NEW: axis to plot on
    label=None,             # NEW: label for legend
    show_plot=True,         # NEW: control saving/closing
):
    E = np.asarray(E)
    Y_matrix = np.asarray(Y_matrix)
    positions = np.asarray(positions)

    #  Apply position limits 
    if pos_limit is not None:
        pmin, pmax = pos_limit
        pos_mask = (positions >= pmin) & (positions <= pmax)
        if not np.any(pos_mask):
            raise ValueError("pos_limit removed all data!")
        positions = positions[pos_mask]
        Y_matrix = Y_matrix[:, pos_mask]

    #  Apply energy limits 
    if ylim is not None:
        ymin, ymax = ylim
        mask = (E >= ymin) & (E <= ymax)
        if not np.any(mask):
            raise ValueError(f"No energy points remain after applying ylim={ylim}")
        E_plot = E[mask]
        Y_plot = Y_matrix[mask, :]
    else:
        E_plot = E
        Y_plot = Y_matrix

    # Mean signal
    Y_sum = np.mean(Y_plot, axis=0)

    #  Initial guesses 
    A0 = (np.max(Y_sum) - np.min(Y_sum)) / 2
    offset0 = np.mean(Y_sum)
    phi0 = np.pi
    p0 = [A0, phi0, offset0]

    
    #  Fit 
    try:
        popt, pcov = curve_fit(
            lambda x, A, phi, offset: sine_func(x, A, phi, offset, omega),
            positions,
            Y_sum,
            p0=[A0, phi0, offset0]
        )
        
        A, phi, offset = popt

        perr = np.sqrt(np.diag(pcov))
        A_err, phi_err, offset_err = perr

        print("Fit results (omega fixed):")
        print(f"  A      = {A:.6g} ± {A_err:.2g}")
        print(f"  omega  = {omega:.6g} (fixed)")
        print(f"  phi    = {phi:.6g} ± {phi_err:.2g}")
        print(f"  offset = {offset:.6g} ± {offset_err:.2g}")



        fit_curve = sine_func(positions, A, phi, offset, omega)
        fit_success = True
    except Exception as e:
        print("Curve fitting failed:", e)
        fit_success = False
    

    #  Plotting 
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created_fig = True

    ax.scatter(positions, Y_sum, lw=2,#c='b',
               label=f"{label} data" if label else "Mean(Y)")

    if fit_success:
        ax.plot(
            positions, fit_curve, lw=2,# c='r', 
            label=(f"{label} fit" if label else f"Sine fit (A={A:.2g}, ω={omega:.2g})")
        )

    ax.set_xlabel("Time delay (fs)")
    ax.set_ylabel("Signal")
    #ax.set_title(fr"Mean signal in eV range {ylim}.(A={A:.2g}, ω={omega:.2g},$\Delta\phi$ = {phi:.3f} ) ")
    
    if fit_success:
        ax.set_title(
            fr"Mean signal in eV range {ylim}. "
            fr"(A={A:.2g}, ω={omega:.2g}, $\Delta\phi$={phi:.3f})"
        )
    else:
        ax.set_title(f"Mean signal in eV range {ylim} (fit failed)")

    
    ax.grid(True)
    ax.legend()

    #  Save only if this function created the figure 
    if created_fig and show_plot:
        plt.tight_layout()
        plt.savefig(output_file, dpi=500)
        plt.close()

    if fit_success:
        return {
            "A": A,
            "omega": omega,
            "phi": phi,
            "offset": offset
        }
    else:
        return None







