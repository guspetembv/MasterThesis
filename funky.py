import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def PlotSurfaceFFT(
    E, Y_matrix, positions,
    output_file="FFT_2D_Map.png",
    clim=None, ylim=None,
    cmap="jet",
    normalize=False,
    detrend=True,
    log_scale=False,
    plot_2d=True,
    output_file_2d="FFT_Amp.png",
    unwrap_phase=True,
    output_file_phase="FFT_Phase.png",
    integrate=False,
    pos_limit=None,     # (pos_min, pos_max) → trims data before FFT
    kx_limit=None       # (kx_min, kx_max) → trims FFT results
):
    """
    Generate a 2D heatmap of FFT amplitude after FFT along the position axis,
    with optional limits on the position input range or kx-frequency output range.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    #  Convert inputs
    Y = np.array(Y_matrix, dtype=float)
    pos = np.array(positions)
    nE, nPos_total = Y.shape

    #  Limit by POSITION (before FFT)
    if pos_limit is not None:
        pmin, pmax = pos_limit
        pos_mask = (pos >= pmin) & (pos <= pmax)
        if not np.any(pos_mask):
            raise ValueError("pos_limit removed all data!")
        pos = pos[pos_mask]
        Y = Y[:, pos_mask]
        Y_matrix = Y.copy()   # maintain original variable name

    #  Optional detrending
    if detrend:
        Y = Y - np.mean(Y, axis=1, keepdims=True)
        Y_matrix = Y_matrix - np.mean(Y_matrix, axis=1, keepdims=True)

    nE, nPos = Y.shape

    #  FFT along position axis
    fft_vals = np.fft.fftshift(np.fft.fft(Y_matrix,n=4000, axis=1), axes=1)
    fft_amp = np.abs(fft_vals)
    fft_phase = np.angle(fft_vals)

    #  Compute spatial frequency axis
    """
    dx = np.mean(np.diff(pos))
    kx = np.fft.fftshift(np.fft.fftfreq(nPos, d=dx))
    """

    dx = np.mean(np.diff(pos))
    Nfft = fft_vals.shape[1]  
    kx = np.fft.fftshift(np.fft.fftfreq(Nfft, d=dx))
    #  Limit by kx-range (after FFT)
    if kx_limit is not None:
        kxmin, kxmax = kx_limit
        kx_mask = (kx >= kxmin) & (kx <= kxmax)
        if not np.any(kx_mask):
            raise ValueError("kx_limit removed all FFT points!")
        kx = kx[kx_mask]
        fft_amp = fft_amp[:, kx_mask]
        fft_phase = fft_phase[:, kx_mask]

    #  Normalize if needed
    if normalize:
        fft_amp /= np.max(fft_amp)

    #  Energy range clipping
    if ylim is not None:
        ymin, ymax = ylim
        mask_E = (E >= ymin) & (E <= ymax)
        E_plot = E[mask_E]
        fft_amp = fft_amp[mask_E, :]
        fft_phase = fft_phase[mask_E, :]
    else:
        E_plot = E

    #  Integrate if needed
    if integrate:
        summed = np.sum(fft_amp, axis=0, keepdims=True)
        fft_amp = np.broadcast_to(summed, fft_amp.shape)

    #  Phase unwrapping
    if unwrap_phase:
        fft_phase_unwrapped = unwrap2d(fft_phase, weight=fft_amp)
        #fft_phase_unwrapped = np.unwrap(fft_phase)
    else:
        fft_phase_unwrapped = fft_phase

    #  Plot 2D FFT map
    vmin = fft_amp.min() if clim is None else clim[0]
    vmax = fft_amp.max() if clim is None else clim[1]
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.figure()
    im = plt.imshow(
        fft_amp,
        extent=[kx.min(), kx.max(), E_plot.min(), E_plot.max()],
        aspect="auto",
        origin="lower",
        cmap=cmap
    )
    plt.xlabel("Frequency (1/fs)")
    plt.ylabel("Energy (eV)")
    plt.title("Rabitt FFT heat map")
    plt.colorbar(im).set_label("FFT Amplitude")
    plt.tight_layout()
    plt.xlim(0,1.5)
    plt.savefig(output_file, dpi=500)
    plt.close()

    #  1D slices
    if plot_2d:
        idx = np.argmax(fft_amp)
        _, kx_idx = np.unravel_index(idx, fft_amp.shape)
        amp_slice = fft_amp[:, kx_idx] / np.max(fft_amp[:, kx_idx])
        plt.figure(figsize=(6, 4))
        plt.plot(E_plot, amp_slice, 'r', lw=1.5)
        plt.title(f"Amplitude at kx={kx[kx_idx]:.3f}")
        plt.xlabel("Energy (eV)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file_2d, dpi=500)
        plt.close()

    if unwrap_phase:
        phase_slice = fft_phase_unwrapped[:, kx_idx]
        plt.figure(figsize=(6, 4))
        plt.scatter(E_plot, phase_slice, s=8, color='k')
        plt.title(f"Phase at kx={kx[kx_idx]:.3f}")
        plt.xlabel("Energy (eV)")
        plt.grid(True)
        plt.ylim(-37,-32)
        plt.tight_layout()
        plt.savefig(output_file_phase, dpi=500)
        plt.close()
    
    #return kx, fft_amp

    # Find global maximum in FFT amplitude
    idx = np.argmax(fft_amp)
    E_idx, kx_idx = np.unravel_index(idx, fft_amp.shape)
    kx_at_max = kx[kx_idx]
    E_at_max = E_plot[E_idx]

    return kx, fft_amp, kx_at_max, E_at_max





# --- Example synthetic data generator and test harness ---
def test_fft_surface_with_sine_wave():
    # Define axes
    nE = 200         # number of energy points
    nPos = 59       # number of spatial/position points
    E = np.linspace(0, 5, nE)          # energy axis (eV)
    positions = np.linspace(-2, 6, nPos)  # position axis (arbitrary units)

    # --- Create predictable test data ---
    # A single spatial frequency sine wave that varies with energy
    spatial_freq = 1.395e-5  # cycles per 10 position units → expect FFT peak at ±0.2
    phase_shift = 0.5 * np.pi  # optional phase offset
    amplitude = 1.0
    AA = 2.5

    # Generate Y(E, x) = sin(2π * f * x + φ(E))
    Y_matrix = AA + amplitude * np.sin(2 * np.pi * spatial_freq * positions[None, :] / (positions[-1] - positions[0]) + phase_shift)
    print(Y_matrix.shape)

    # Optionally modulate amplitude or phase with energy for more complex patterns
    # e.g., Y_matrix *= np.cos(2 * np.pi * 0.1 * E[:, None])

    # --- Run your FFT surface plot ---
    kx, fft_amp = PlotSurfaceFFT_2D(
        E, Y_matrix, positions,
        output_file="test_FFT_2D_Map.png",
        output_file_2d="test_FFT_1D.png",
        output_file_phase="test_FFT_Phase.png",
        normalize=True,
        log_scale=False,
        unwrap_phase=False,
        plot_2d=False
    )

    # --- Diagnostic: print expected vs found frequency peak ---
    expected_peak = spatial_freq / (positions[-1] - positions[0])
    kx_peak = kx[np.argmax(np.mean(fft_amp, axis=0))]
    print(f"Expected frequency ≈ {expected_peak:.3f}, detected peak ≈ {kx_peak:.3f}")

def unwrap2d(wrapped, weight=None):
    """
    2D phase unwrapping using reliability-based method.
    Translated from David Kroon's MATLAB code (Appl. Opt. 41, 7437–7444, 2002).
    
    Parameters
    ----------
    wrapped : 2D np.ndarray
        Wrapped phase map in radians (values between -π and π).
    weight : 2D np.ndarray or None
        Weight matrix (reliability). If None, defaults to ones.
        
    Returns
    -------
    unwrapped : 2D np.ndarray
        Unwrapped phase map.
    """
    wrapped = np.array(wrapped, dtype=float)
    nrow, ncol = wrapped.shape
    if weight is None:
        weight = np.ones_like(wrapped)

    # === Compute reliability R ===
    R = np.zeros_like(wrapped)
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            H = np.angle(np.exp(1j*(wrapped[i-1,j]-wrapped[i,j]))) - np.angle(np.exp(1j*(wrapped[i,j]-wrapped[i+1,j])))
            V = np.angle(np.exp(1j*(wrapped[i,j-1]-wrapped[i,j]))) - np.angle(np.exp(1j*(wrapped[i,j]-wrapped[i,j+1])))
            D1 = np.angle(np.exp(1j*(wrapped[i-1,j-1]-wrapped[i,j]))) - np.angle(np.exp(1j*(wrapped[i,j]-wrapped[i+1,j+1])))
            D2 = np.angle(np.exp(1j*(wrapped[i-1,j+1]-wrapped[i,j]))) - np.angle(np.exp(1j*(wrapped[i,j]-wrapped[i+1,j-1])))
            D = np.sqrt(H**2 + V**2 + D1**2 + D2**2)
            if D != 0:
                R[i, j] = weight[i, j] / D
            else:
                R[i, j] = 0

    # === Build edge list (horizontal + vertical) ===
    edges = []
    for i in range(1, nrow-1):
        for j in range(1, ncol-1):
            if j != ncol-1:
                edges.append((R[i, j] + R[i, j+1], i, j, i, j+1))
            if i != nrow-1:
                edges.append((R[i, j] + R[i+1, j], i, j, i+1, j))
    edges.sort(reverse=True, key=lambda x: x[0])

    # === Group tracking ===
    group_id = np.zeros((nrow, ncol), dtype=int)
    groups = []
    group_count = 0

    # === Unwrapping process ===
    for _, r1, c1, r2, c2 in edges:
        g1 = group_id[r1, c1]
        g2 = group_id[r2, c2]
        if g1 == 0 and g2 == 0:
            corr = np.round((wrapped[r2, c2] - wrapped[r1, c1]) / (2*np.pi)) * 2*np.pi
            wrapped[r2, c2] -= corr
            group_count += 1
            groups.append([(r1, c1), (r2, c2)])
            group_id[r1, c1] = group_count
            group_id[r2, c2] = group_count
        elif g1 != 0 and g2 == 0:
            corr = np.round((wrapped[r2, c2] - wrapped[r1, c1]) / (2*np.pi)) * 2*np.pi
            wrapped[r2, c2] -= corr
            groups[g1-1].append((r2, c2))
            group_id[r2, c2] = g1
        elif g1 == 0 and g2 != 0:
            corr = np.round((wrapped[r1, c1] - wrapped[r2, c2]) / (2*np.pi)) * 2*np.pi
            wrapped[r1, c1] -= corr
            groups[g2-1].append((r1, c1))
            group_id[r1, c1] = g2
        elif g1 != g2:
            # merge groups
            corr = np.round((wrapped[r2, c2] - wrapped[r1, c1]) / (2*np.pi)) * 2*np.pi
            wrapped[np.array([r for r, _ in groups[g2-1]]), np.array([c for _, c in groups[g2-1]])] -= corr
            for (r, c) in groups[g2-1]:
                group_id[r, c] = g1
            groups[g1-1].extend(groups[g2-1])
            groups[g2-1] = []

    return wrapped


