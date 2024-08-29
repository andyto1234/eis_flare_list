import numpy as np
from scipy import io
from tqdm import tqdm
from typing import Tuple
from demregpy import dn2dem
import matplotlib.pyplot as plt
import astropy.time as time
import numpy as np

class DEMCalculator:
    def __init__(self, response_file: str = 'aia_tresp_en.dat'):
        self.trin = io.readsav(f"demregpy/tresp/{response_file}")
        self.tresp_logt = np.array(self.trin['logt'])
        self.trmatrix = self.trin['tr'].T
        
        self.t_space = 0.05
        self.t_min, self.t_max = 5.625, 7.625

        self.logtemps = np.arange(self.t_min, self.t_max + self.t_space, self.t_space) - self.t_space / 2
        self.temps = 10 ** self.logtemps
        self.mlogt=np.array([np.mean([(np.log10(self.temps[i])),np.log10((self.temps[i+1]))]) for i in np.arange(0,len(self.temps)-1)])

        
        self.tr_reduced = self._interp_tr(self.mlogt, self.tresp_logt, self.trmatrix)

    def _interp_emis_temp(self, original_array: np.ndarray) -> np.ndarray:
        new_size = 101
        new_indices = np.linspace(0, len(original_array) - 1, new_size)
        return np.interp(new_indices, np.arange(len(original_array)), original_array)

    def _interp_tr(self, logtemps: np.ndarray, tresp_logt: np.ndarray, trmatrix: np.ndarray) -> np.ndarray:
        return np.array([np.interp(logtemps, tresp_logt, trmatrix[:, i]) for i in range(trmatrix.shape[1])]).T

    def _pred_intensity_compact(self, emis: np.ndarray, logt: np.ndarray, dem: np.ndarray) -> float:
        integrand = emis * dem
        temp = logt[:len(integrand)]
        return np.trapz(integrand, temp)

    def calculate_dem(self, map_array: np.ndarray, err_array: np.ndarray) -> Tuple[np.ndarray, ...]:
        nx, ny, nf = map_array.shape
        n_temps = len(self.temps) - 1
        
        dem = np.zeros((nx, ny, n_temps))
        edem = np.zeros((nx, ny, n_temps))
        elogt = np.zeros((nx, ny, n_temps))
        chisq = np.zeros((nx, ny))
        dn_reg = np.zeros((nx, ny, nf))

        total_pixels = nx * ny
        with tqdm(total=total_pixels, desc="Calculating DEM", unit="pixel") as pbar:
            for i in range(nx):
                for j in range(ny):
                    pixel_data = map_array[i, j, :]
                    pixel_error = err_array[i, j, :]
                    
                    dem[i,j,:], edem[i,j,:], elogt[i,j,:], chisq[i,j], dn_reg[i,j,:] = dn2dem(
                        pixel_data, pixel_error, self.trmatrix, self.tresp_logt, self.temps, 
                        emd_int=True, emd_ret=False, gloci=1, max_iter=200
                    )
                    
                    percentage_difference = np.abs(pixel_data - np.array([
                        self._pred_intensity_compact(self.tr_reduced[:,k], 10**self.mlogt, dem[i,j,:])
                        for k in range(self.tr_reduced.shape[1])
                    ])) / pixel_data * 100
                    
                    pbar.update(1)
                    pbar.set_postfix({"Pixel": f"({i},{j})", "Chi^2": f"{chisq[i,j]:.2f}"})
        
        return dem, edem, elogt, chisq, dn_reg, self.mlogt, self.logtemps

    def plot_dem_images(self, submap, dem: np.ndarray, img_arr_tit: str):
        """
        Plot DEM images.

        Args:
            submap: The submap containing date information
            dem (np.ndarray): The Differential Emission Measure array
            img_arr_tit (str): The title for the saved image
        """
        nt = dem.shape[2]
        nt_new = int(nt / 2)
        nc, nr = 3, 3

        plt.rcParams.update({
            'font.size': 12,
            'font.family': "sans-serif",
            'font.sans-serif': "Arial",
            'mathtext.default': "regular"
        })

        fig, axes = plt.subplots(nrows=nr, ncols=nc, figsize=(12, 12), sharex=True, sharey=True)
        fig.suptitle('Image time = ' + time.Time.strftime(submap.date, "%Y-%m-%dT%H:%M:%S"))
        fig.supxlabel('Pixels')
        fig.supylabel('Pixels')
        cmap = plt.cm.get_cmap('cubehelix_r')

        for i, axi in enumerate(axes.flat):
            new_dem = (dem[:, :, i*2] + dem[:, :, i*2+1]) / 2
            im = axi.imshow(new_dem, vmin=1e19, vmax=1e21, origin='lower', cmap=cmap, aspect='auto')
            axi.set_title(f'{self.logtemps[i*2]:.2f} - {self.logtemps[i*2+2]:.2f}')

        plt.tight_layout()
        plt.colorbar(im, ax=axes.ravel().tolist(), label='$\mathrm{DEM\;[cm^{-5}\;K^{-1}]}$', fraction=0.03, pad=0.02)
        plt.savefig(img_arr_tit, bbox_inches='tight')
        plt.close(fig)

    def calculate_and_plot_dem(self, map_array: np.ndarray, err_array: np.ndarray, submap, img_arr_tit: str) -> Tuple[np.ndarray, ...]:
        """
        Calculate DEM and plot the results.

        Args:
            map_array (np.ndarray): Input map array
            err_array (np.ndarray): Error array
            submap: The submap containing date information
            img_arr_tit (str): The title for the saved image

        Returns:
            Tuple[np.ndarray, ...]: Results of DEM calculation
        """
        results = self.calculate_dem(map_array, err_array)
        dem = results[0]  # The DEM is the first item in the returned tuple
        self.plot_dem_images(submap, dem, img_arr_tit)
        return results

if __name__ == "__main__":
    # Example usage
    calculator = DEMCalculator()
    
    # Assuming you have map_array and err_array defined
    map_array = np.random.rand(10, 10, 6)  # Example shape
    err_array = np.random.rand(10, 10, 6)  # Example shape
    
    results = calculator.calculate_dem(map_array, err_array)
    
    # Process results as needed
    dem, edem, elogt, chisq, dn_reg, mlogt, logtemps = results
    print(f"DEM shape: {dem.shape}")
    print(f"Chi-square mean: {np.mean(chisq)}")