# Main script
import pandas as pd
import numpy as np
import matplotlib
matplotlib.rcParams["mathtext.fontset"] = "cm"     # Computer Modern
matplotlib.rcParams["font.family"] = "serif"
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import os
from tools import TofToEnergy, AllEnergy, PlotSurface, CurveFit
from funky import PlotSurfaceFFT

plt.rcParams['font.size'] = 14



liimmmmittt = (-7, 20)

# Sideband limit definitions
sidebands = {
    "Spectra":{
        "ylime": (0,15.5),
        "pos_lime": liimmmmittt
    },
}
"""
    "Spectra2":{
        "ylime": (2.5, 8),
        "pos_lime": liimmmmittt
        },
    "SB1": {
        "ylime": (4.8, 5.4),
        "pos_lime": liimmmmittt
    },
    "SB1-12": {
        "ylime": (4.88, 5.08),
        "pos_lime": liimmmmittt
    },
    "SB1-32": {
        "ylime": (5.130, 5.280),
        "pos_lime": liimmmmittt
    },
    "SB2": {
        "ylime": (8.3, 9.05),
        "pos_lime": liimmmmittt
    },
    "SB2-12": {
        "ylime": (1.05, 1.55),
        "pos_lime": liimmmmittt
    },
    "SB2-32": {
        "ylime": (1.65, 2.0),
        "pos_lime": liimmmmittt
    }


}
"""

# Directories
base_output_dir = "figs/cod/WJ/"
os.makedirs(base_output_dir, exist_ok=True)

data_dir = "data/"
os.makedirs(data_dir, exist_ok=True)

# Load Data
df = pd.read_csv(
    os.path.join(data_dir, "cod.csv"),
    names=["time", "intensity", "position"]
)


def PlotMeanTof(df, xlim=None, vlines=None):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Normalize intensity within each position
    df["norm_intensity"] = df.groupby("position")["intensity"].transform(
        lambda x: x / x.max()
    )
    mean_curve = df.groupby("time")["norm_intensity"].mean()
    sum_curve = df.groupby("time")["norm_intensity"].sum()
    ax.plot(sum_curve.index, sum_curve.values, color="black")

    ax.set_xlabel("Time of flight (ns?)")
    ax.set_ylabel("Mean intensity")
    if xlim is not None:
        ax.set_xlim(xlim)

    # Add vertical lines with labels
    if vlines is not None:
        for xv in vlines:
            ax.axvline(x=xv, color="red", linestyle="--")
            ax.text(
                xv, ax.get_ylim()[1]*0.95,   # near the top of the plot
                f"{xv:.2e}",                # format the value
                rotation=90,
                va="top", ha="right",
                color="red",
                fontsize=9,
            )

    fig.savefig(os.path.join(base_output_dir, "RawTof_mean.png"))
    #plt.close(fig)
    plt.show()

#PlotMeanTof(df,xlim=(1e+6,0.75e+7), vlines=[4.62*10**6, 3.59*10**6, 3.05*10**6, 2.72*10**6, 2.47*10**6, 2.28*10**6, 2.20*10**6])



def PlotMeanEnergy(E, Y_matrix, xlim=None):
    mean_spectrum = Y_matrix.mean(axis=1)

    plt.figure(figsize=(8,5), constrained_layout=True)
    plt.plot(E, mean_spectrum, color="black")
    plt.title(r"With jacobian correction")
    plt.xlabel(r"E$_{\text{kin}}$ (eV)")
    plt.ylabel(r"Intensity (arb. units)")
    if xlim:
        plt.xlim(xlim)
    #plt.ylim(mean_spectrum.min(), mean_spectrum.max())
    plt.grid(True)
    #plt.tight_layout()
    plt.savefig(os.path.join(base_output_dir,"EnergyEV.png"))
    plt.show()


def PlotCOD(tof, signal,L0,T0,E0, xlim=None):
    dt = 1
    signal /= signal.max()  
    plt.figure(figsize=(8,5), constrained_layout=True)
    plt.plot(tof, signal, color="black")
    plt.title(r"Raw time-of-flight")
    plt.xlabel(r"ToF ()")
    plt.ylabel(r"Intensity (arb. units)")
    if xlim:
        plt.xlim(xlim)
    #plt.ylim(mean_spectrum.min(), mean_spectrum.max())
    plt.grid(True)
    #plt.tight_layout()
    #plt.savefig(os.path.join(base_output_dir,"EnergyEV.png"))
    #plt.show()
    plt.close()   #mean_spectrum = Y_matrix.mean(axis=1)

    E, signal = TofToEnergy(tof*dt, signal, L0, T0, E0)

    E *= 10**(1)
    signal /= signal.max()
    
    plt.figure(figsize=(8,5), constrained_layout=True)
    plt.plot(E, signal, color="black")
    plt.title(r"With jacobian correction")
    plt.xlabel(r"E$_{\text{kin}}$ (eV)")
    plt.ylabel(r"Intensity (arb. units)")
    if xlim:
        plt.xlim(xlim)
    #plt.ylim(mean_spectrum.min(), mean_spectrum.max())
    plt.grid(True)
    #plt.tight_layout()
    #plt.savefig(os.path.join(base_output_dir,"EnergyEV.png"))
    plt.show()
    plt.close()





def reconstruct_pulse_train(sideband_params, tau):
    pulse = np.zeros_like(tau)

    for sb, p in sideband_params.items():
        A = p["A"]
        omega = p["omega"]
        phi = p["phi"]

        pulse += A * np.sin(omega * tau - phi)

    return pulse




"""
# Constants 1622
L0  = 11912.015017
T0 = -262.816143
E0 = -2.674607
"""

# COD
L0  = 9675.436057
T0 = 110.139466
E0 = -0.653012


# Compute energy maps
#E, Y_matrix, positions = AllEnergy(df, L0, T0, E0, dE=0.010)

#PlotMeanEnergy(E, Y_matrix, xlim=(0,20))
PlotCOD(df["time"], df["intensity"], L0, T0, E0)#,xlim=(-0.654,-0.650))#, xlim=(2*10**6, 6*10**6))














"""   
# Dictionary to collect fit parameters
sideband_params = {}

# Create one figure for all fits
fig_cf, ax_cf = plt.subplots(figsize=(10, 6))

for sb_name, cfg in sidebands.items():

    yl = cfg["ylime"]
    pl = cfg["pos_lime"]

    out_dir = os.path.join(base_output_dir, sb_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing {sb_name} → {out_dir}")
    print(f"   Energy range: {yl}")
    print(f"   Position range: {pl}")

    # surface plot
    PlotSurface(
        E, Y_matrix, positions,
        output_file=os.path.join(out_dir, "SurfacePlot.png"),
        cmap="jet",
        clim=None,
        ylim=yl,
        #elev=35,
        #azim=-55,
        integrate=False,
        pos_limit=pl
    )
    # fft surface plots
    kx, fft_amp, kx_at_max, E_at_max = PlotSurfaceFFT(
        E, Y_matrix, positions,
        output_file=os.path.join(out_dir, "SurfaceFFT_2D.png"),
        clim=None,
        ylim=yl,
        cmap="jet",
        normalize=False,
        detrend=True,
        log_scale=False,
        plot_2d=False,
        output_file_2d=os.path.join(out_dir, "FFT_Amp.png"),
        unwrap_phase=False,
        output_file_phase=os.path.join(out_dir, "FFT_Phase.png"),
        integrate=False,
        pos_limit=pl
    )

    print(f'f={kx_at_max:.3} and omega={2*np.pi*kx_at_max:.3}')

    # --- CurveFit: overlay on shared axis ---'
    #!!!!! FOR COMBINED PLOT SET: "ax=ax_cf" and "show_plot=False". FOR INDIVIDUAL SET: "ax=None" and "show_plot=True"!!!!!!
    fit_result = CurveFit(
        E, Y_matrix, positions,
        omega=np.abs(2*np.pi*kx_at_max),
        output_file=os.path.join(out_dir, "CurveFit.png"),  # still saves individual if desired
        ylim=yl,
        pos_limit=pl,
        ax=None,              # SAME axis for all
        label=sb_name,         # label each sideband
        show_plot=True        # don't auto-save/close here
    )

    if fit_result is not None:
        sideband_params[sb_name] = fit_result
"""


# Save the combined comparison figure
#plt.tight_layout()
#plt.savefig(os.path.join(base_output_dir, "All_Sidebands_CurveFit_Comparison.png"), dpi=500)
#plt.show()


























