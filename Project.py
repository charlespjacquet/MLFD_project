# === RBF-based velocity magnitude and vorticity interpolation ===
# Includes separated functions for velocity and vorticity interpolation using RBFs.

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
from tqdm import tqdm


PATH = r"C:\Users\charl\OneDrive\VKI\Research_Project\TEST_CAMPAIGN_1\Grouped_images\L2B_24022025_3_pitch_85_5_1_GROUPIMAGE_UP\split_images\filtered\out_PaIRS"





# --- Gaussian RBF and its derivatives ---
def rbf_2d_gaussian(x, y, xc, yc, epsilon):
    """
    Gaussian radial basis function in 2D.

    Parameters
    ----------
    x, y : array_like
        Evaluation coordinates.
    xc, yc : float
        Center of the RBF.
    epsilon : float
        Shape parameter controlling the width.

    Returns
    -------
    array_like
        RBF values at each (x, y).
    """
    r2 = (x - xc)**2 + (y - yc)**2
    return np.exp(-epsilon**2 * r2)

def drbf_dx(x, y, xc, yc, epsilon):
    """
    Derivative of the 2D Gaussian RBF with respect to x.
    """
    return -2 * epsilon**2 * (x - xc) * rbf_2d_gaussian(x, y, xc, yc, epsilon)

def drbf_dy(x, y, xc, yc, epsilon):
    """
    Derivative of the 2D Gaussian RBF with respect to y.
    """
    return -2 * epsilon**2 * (y - yc) * rbf_2d_gaussian(x, y, xc, yc, epsilon)

# --- Load and average a single PIV case ---
def process_piv_case(folder_path):
    """
    Process and average all .mat files in a folder for a single PIV case.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing .mat files.

    Returns
    -------
    dict
        Dictionary with fields x, y, mean velocity components, magnitude, etc.
    """
    SN = Mag = Umean = Vmean = None
    countImg = 0

    mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat') and f != 'out.mat']
    mat_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f.split('_')[1]))))

    print(f"Processing folder: {folder_path} with {len(mat_files)} files...")

    for filename in tqdm(mat_files, desc=f"Processing {os.path.basename(folder_path)}"):
        file_path = os.path.join(folder_path, filename)
        mat_data = scipy.io.loadmat(file_path)
        U, V = mat_data['U'], mat_data['V']
        x, y = mat_data['x'], mat_data['y']
        sn = mat_data['SN']

        if countImg == 0:
            nx, ny = U.shape
            SN = np.zeros((nx, ny))
            Mag = np.zeros((nx, ny))
            Umean = np.zeros((nx, ny))
            Vmean = np.zeros((nx, ny))

        Mag += np.sqrt(U**2 + V**2)
        Umean += U
        Vmean += V
        SN += sn

        countImg += 1

    Mag /= countImg
    Umean /= countImg
    Vmean /= countImg
    SN /= countImg

    return {
        "folder": folder_path,
        "x": x,
        "y": y,
        "Mag": Mag,
        "Umean": Umean,
        "Vmean": Vmean,
        "SN": SN,
    }

# --- Load and process multiple PIV cases ---
def process_selected_cases(folder_paths):
    """
    Load and process multiple PIV cases.

    Parameters
    ----------
    folder_paths : list of str
        Paths to the folders containing PIV .mat data.

    Returns
    -------
    dict
        Dictionary mapping each case to its processed data.
    """
    results_dict = {}
    for folder_path in folder_paths:
        result = process_piv_case(folder_path)
        if result:
            case_name = os.path.basename(folder_path)
            results_dict[case_name] = result
    return results_dict

# --- Interpolate velocity magnitude only ---
def rbf_interpolate_velocity_magnitude(all_results, epsilon=20.0, alpha=1e-6, n_grid=200, subsample_step=5):
    """
    Interpolates the mean velocity magnitude field using RBF interpolation.

    Parameters
    ----------
    all_results : dict
        Dictionary of processed PIV cases.
    epsilon : float
        Shape parameter of the Gaussian RBF.
    alpha : float
        Regularization parameter for solving the linear system.
    n_grid : int
        Resolution of the interpolation grid.
    subsample_step : int
        Subsampling step for RBF centers.

    Returns
    -------
    None
    """
    print("--- RBF Interpolation: Velocity Magnitude ---")
    x_all, y_all, u_all, v_all = [], [], [], []
    for result in all_results.values():
        x_all.append(result['x'].flatten())
        y_all.append(result['y'].flatten())
        u_all.append(result['Umean'].flatten())
        v_all.append(result['Vmean'].flatten())

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)
    u_all = np.concatenate(u_all)
    v_all = np.concatenate(v_all)

    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    x_norm = 2 * (x_all - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (y_all - y_min) / (y_max - y_min) - 1

    idx = np.arange(0, len(x_norm), subsample_step)
    xc, yc = x_norm[idx], y_norm[idx]

    Phi = np.zeros((len(x_norm), len(xc)))
    for i in tqdm(range(len(xc)), desc="Building RBF matrix"):
        Phi[:, i] = rbf_2d_gaussian(x_norm, y_norm, xc[i], yc[i], epsilon)

    wu = np.linalg.solve(Phi.T @ Phi + alpha * np.eye(len(xc)), Phi.T @ u_all)
    wv = np.linalg.solve(Phi.T @ Phi + alpha * np.eye(len(xc)), Phi.T @ v_all)

    xi = np.linspace(-1, 1, n_grid)
    yi = np.linspace(-1, 1, n_grid)
    XI, YI = np.meshgrid(xi, yi)
    Xg, Yg = XI.flatten(), YI.flatten()

    U_interp = np.zeros(len(Xg))
    V_interp = np.zeros(len(Xg))
    for i in tqdm(range(len(xc)), desc="Interpolating velocity"):
        U_interp += wu[i] * rbf_2d_gaussian(Xg, Yg, xc[i], yc[i], epsilon)
        V_interp += wv[i] * rbf_2d_gaussian(Xg, Yg, xc[i], yc[i], epsilon)

    MAG_interp_grid = np.sqrt(U_interp**2 + V_interp**2).reshape(XI.shape)
    XI_phys = 0.5 * (XI + 1) * (x_max - x_min) + x_min
    YI_phys = 0.5 * (YI + 1) * (y_max - y_min) + y_min

    x_origin = 150
    y_origin = 57
    z_hub = 160
    D = 150
    XI_corr = (XI_phys + x_origin) / D
    YI_corr = (y_origin + (y_max - YI_phys)) / z_hub

    plt.figure(figsize=(8, 6))
    plt.contourf(XI_corr, YI_corr, MAG_interp_grid / np.max(MAG_interp_grid), 100, cmap='jet')
    plt.colorbar(label=r"$U/\bar{u}_{hub}$ [-]")
    plt.title(f'Interpolated Velocity Magnitude (ε={epsilon}, α={alpha})')
    plt.xlabel(r"$x/D$")
    plt.ylabel(r"$z/z_{hub}$")
    plt.axis('equal')
    plt.tight_layout()
    filename = f"velocity_mag_eps{epsilon}_alpha{alpha}.png"
    plt.savefig(filename)
    print(f"Velocity magnitude figure saved as {filename}")
    #plt.close()

# --- Interpolate vorticity only ---
def rbf_interpolate_vorticity(all_results, epsilon=20.0, alpha=1e-6, n_grid=200, subsample_step=5):
    """
    Interpolates the vorticity field using RBF and analytical derivatives.

    Parameters
    ----------
    all_results : dict
        Dictionary of processed PIV cases.
    epsilon : float
        Shape parameter of the Gaussian RBF.
    alpha : float
        Regularization parameter for solving the linear system.
    n_grid : int
        Resolution of the interpolation grid.
    subsample_step : int
        Subsampling step for RBF centers.

    Returns
    -------
    None
    """
    print("--- RBF Interpolation: Vorticity ---")
    x_all, y_all, u_all, v_all = [], [], [], []
    for result in all_results.values():
        x_all.append(result['x'].flatten())
        y_all.append(result['y'].flatten())
        u_all.append(result['Umean'].flatten())
        v_all.append(result['Vmean'].flatten())

    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)
    u_all = np.concatenate(u_all)
    v_all = np.concatenate(v_all)

    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    x_norm = 2 * (x_all - x_min) / (x_max - x_min) - 1
    y_norm = 2 * (y_all - y_min) / (y_max - y_min) - 1

    idx = np.arange(0, len(x_norm), subsample_step)
    xc, yc = x_norm[idx], y_norm[idx]

    Phi = np.zeros((len(x_norm), len(xc)))
    for i in tqdm(range(len(xc)), desc="Building RBF matrix"):
        Phi[:, i] = rbf_2d_gaussian(x_norm, y_norm, xc[i], yc[i], epsilon)

    wu = np.linalg.solve(Phi.T @ Phi + alpha * np.eye(len(xc)), Phi.T @ u_all)
    wv = np.linalg.solve(Phi.T @ Phi + alpha * np.eye(len(xc)), Phi.T @ v_all)

    xi = np.linspace(-1, 1, n_grid)
    yi = np.linspace(-1, 1, n_grid)
    XI, YI = np.meshgrid(xi, yi)
    Xg, Yg = XI.flatten(), YI.flatten()

    dVdx = np.zeros(len(Xg))
    dUdy = np.zeros(len(Xg))
    for i in tqdm(range(len(xc)), desc="Computing derivatives"):
        dVdx += wu[i] * drbf_dy(Xg, Yg, xc[i], yc[i], epsilon)
        dUdy += wv[i] * drbf_dx(Xg, Yg, xc[i], yc[i], epsilon)
    vorticity_interp = dVdx - dUdy
    VORT_interp_grid = vorticity_interp.reshape(XI.shape)

    XI_phys = 0.5 * (XI + 1) * (x_max - x_min) + x_min
    YI_phys = 0.5 * (YI + 1) * (y_max - y_min) + y_min

    x_origin = 150
    y_origin = 57
    z_hub = 160
    D = 150
    XI_corr = (XI_phys + x_origin) / D
    YI_corr = (y_origin + (y_max - YI_phys)) / z_hub

    plt.figure(figsize=(8, 6))
    plt.contourf(XI_corr, YI_corr, VORT_interp_grid, 100, cmap='seismic')
    plt.colorbar(label=r"Vorticity [1/s]")
    plt.title(f'Interpolated Vorticity Field (ε={epsilon}, α={alpha})')
    plt.xlabel(r"$x/D$")
    plt.ylabel(r"$z/z_{hub}$")
    plt.axis('equal')
    plt.tight_layout()
    filename = f"vorticity_eps{epsilon}_alpha{alpha}.png"
    plt.savefig(filename)
    print(f"Vorticity figure saved as {filename}")
    #plt.close()

# === MAIN ===
if __name__ == "__main__":
    folder_list = [PATH]

    all_results = process_selected_cases(folder_list)
    rbf_interpolate_velocity_magnitude(all_results, epsilon=20, alpha=1e-6)
    rbf_interpolate_vorticity(all_results, epsilon=15, alpha=1e-6)

    param_study = 0
    if param_study == 1:
        epsilons = [10, 20]
        alphas = [1e-6, 1e-4]
        for eps in epsilons:
            for alpha in alphas:
                rbf_interpolate_velocity_magnitude(all_results, epsilon=eps, alpha=alpha)
                rbf_interpolate_vorticity(all_results, epsilon=eps, alpha=alpha)
