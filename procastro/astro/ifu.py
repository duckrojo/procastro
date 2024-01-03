import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from scipy.optimize import curve_fit


def closest(x, arr):
    """
    From an array and a value, the function finds the closest value in the array to that value.

    Args:
        x (float): Value that you are looking for.
        arr (array): Array where to look for the value or the closest one.

    Returns:
        index (int): Index where the value (or the closest one in the array) is.
    """

    difference_array = np.absolute(arr-x)
    index = difference_array.argmin()
    return index


def visualize(cube_path, save_plots=None, plots=True):
    """
    Provides a quick visualization of the data, allowing you to check the dimensions
    of the slit, data quality, and object center. Also returns important parameters.

    Args:
        cube_path (str): Path to the data cube.

    Returns:
        data (3D array): The raw data of the data cube.
        wave (array): An array representing the wavelength range of the data.
        pix_x (int): Number of pixels in the x-direction of the slit.
        pix_y (int): Number of pixels in the y-direction of the slit.
        dx (float): Arcseconds between pixels in the x-direction of the slit.
        dy (float): Arcseconds between pixels in the y-direction of the slit.
    """
    obs = get_pkg_data_filename(cube_path)
    hdul = fits.open(cube_path)
    header = hdul[0].header
    N = header["NAXIS3"]
    pix_x = header["NAXIS1"]
    pix_y = header["NAXIS2"]
    dx = np.abs(header["CDELT1"]) * 60 * 60
    dy = np.abs(header["CDELT2"]) * 60 * 60
    wave = np.zeros(N)
    for i in range(N):
        wave[i] = (i+header["CRPIX3"])*header["CDELT3"] + header["CRVAL3"]
    #obtain the data and wavelength
    data = fits.getdata(obs, ext=0)[:, :, :]
    medians = np.zeros((pix_y, pix_x))
    for i in range(pix_x):
        for j in range(pix_y):
            medians[j, i] = np.nanpercentile(data[:, j, i], 50)

    if plots==True:
        fig, axes = plt.subplots(1, 1, figsize=(6, 10))
        axes.set_title("Visualization of the data cube")
        axes.imshow(medians, origin="lower", aspect=dy/dx)
        axes.set_ylabel("y-spaxels")
        axes.set_xlabel("x-spaxels")
        if save_plots != None:
            plt.savefig(save_plots + "/Visualization_RawData.png")

    print(" ")
    print("Size of the data: (",N ,", ", pix_y, ", ", pix_x, ")")
    return data, wave, int(pix_x), int(pix_y), dx, dy


def Atmospheric_dispersion_correction(cube_path, 
                                      data=np.array([]), 
                                      wave=np.array([]),
                                      center_x=True, 
                                      center_y=True, 
                                      range_x = None, 
                                      range_y = None, 
                                      plots = True, 
                                      max_plots=3,
                                      save_plots=None):
    """
    Corrects atmospheric dispersion over a data cube, returning a modified data cube
    depending on the observed object's center movement. Also provides the new center
    calculated for the object in the corrected data cube.

    Args:
        cube_path (str): Path to the data cube.
        data (3D array): 3D array containing the data cube.
        wave (array): Array containing the wavelengths of the data.
        center_x (bool): True to correct atmospheric dispersion in the x-direction.
        center_y (bool): True to correct atmospheric dispersion in the y-direction.
        range_x (tuple): Tuple specifying the range for parabolic fit in the x-direction.
        range_y (tuple): Tuple specifying the range for parabolic fit in the y-direction.
        plots (bool): True to display plots.
        max_plots (float): Factor to set vertical plot limits. ylim = max_plots * data_median.
        save_plots (str): If a path is provided, the images are save in this directory.

    Returns:
        corrected_data (3D array): Data cube with atmospheric dispersion correction.
        center (tuple): Calculated center of the new data cube.
    """
    if len(data) == 0:
        obs = get_pkg_data_filename(cube_path)
        hdul = fits.open(cube_path)
        header = hdul[0].header
        N = header["NAXIS3"]
        wave = np.zeros(N)
        #obtain the data and wavelength
        data = fits.getdata(obs, ext=0)[:, :, :]

        for i in range(N):
            wave[i] = (i+header["CRPIX3"])*header["CDELT3"] + header["CRVAL3"]
    
    if (len(data) > 0) and (len(wave) == 0):
        print(" ")
        print("wave = empty")
        print("Error: If you provide the data, also should provide the wavelength")

    if len(data) > 0:
        N = len(data)

    pix_x = len(data[0, 0, :])
    pix_y = len(data[0, :, 0])

    x_coords_center = np.zeros(N)
    for l in range(N):
        suma_x = 0
        mass_center_x = 0

        for i in range(pix_x):
            mass_center_x += (i*np.sum(data[l, :, i]))
            suma_x += np.sum(data[l, :, i])

        x_coords_center[l] = mass_center_x / suma_x

    if center_x == False:
        x_coords_center = np.ones(N)*np.nanmedian(x_coords_center)


    y_coords_center = np.zeros(N)
    for l in range(N):
        suma_y = 0
        mass_center_y = 0

        for j in range(pix_y):
            mass_center_y += (j*np.sum(data[l, j, :]))
            suma_y += np.sum(data[l, j, :])


        y_coords_center[l] = mass_center_y / suma_y

    if center_y == False:
        y_coords_center = np.ones(N)*np.nanmedian(y_coords_center)

    if plots == True:
        fig, axes = plt.subplots(2, 1, figsize=(18, 6))

    # Define the parabola function
    def parabola(x, a, b, c):
        return a*x**2 + b*x + c

    if range_y == None:
        # Fit a parabola to the data using curve_fit()
        popt_y, pcov_y = curve_fit(parabola, wave[:], y_coords_center[:] - y_coords_center[0])
    if range_y != None:
        # Fit a parabola to the data using curve_fit()
        lower = closest(range_y[0], wave)
        upper = closest(range_y[1], wave)
        popt_y, pcov_y = curve_fit(parabola, wave[lower:upper], y_coords_center[lower:upper] - y_coords_center[0])

    # Plot the data and the parabola

    if plots == True:
        axes[0].set_title("Movement of the center in y of the object through wavelength")
        axes[0].plot(wave, y_coords_center,  linewidth=0.3, c="k", alpha=1, label='center in y-spaxels')
        axes[0].plot(wave, y_coords_center[0] + parabola(wave, *popt_y), c="red", label='parabola')
        axes[0].plot(wave, y_coords_center[0] + np.round(parabola(wave, *popt_y)), linestyle=":", c="purple", label='round parabola', zorder=10)
        axes[0].set_ylim(-1, pix_y + 1)
        axes[0].set_xlabel("Wavelength [nm]", fontsize=14)
        axes[0].set_ylabel("y-spaxels", fontsize=14)
        axes[0].grid(False)
        axes[0].legend()

    if range_x == None:
        # Fit a parabola to the data using curve_fit()
        popt_x, pcov_x = curve_fit(parabola, wave[:], x_coords_center[:] - x_coords_center[0])

    if range_x != None:
        lower = closest(range_x[0], wave)
        upper = closest(range_x[1], wave)
        popt_x, pcov_x = curve_fit(parabola, wave[lower:upper], x_coords_center[lower:upper] - x_coords_center[0])

    # Plot the data and the parabola

    if plots == True:
        axes[1].set_title("Movement of the center in x of the object through wavelength")
        axes[1].plot(wave, x_coords_center,  linewidth=0.3, c="k", alpha=1, label='center in y-spaxels')
        axes[1].plot(wave, x_coords_center[0] + parabola(wave, *popt_x), c="red", label='parabola')
        axes[1].plot(wave, x_coords_center[0] + np.round(parabola(wave, *popt_x)), linestyle=":", c="purple", label='round parabola', zorder=10)
        axes[1].set_ylim(-1, pix_x + 1)
        axes[1].set_xlabel("Wavelength [nm]", fontsize=14)
        axes[1].set_ylabel("x-spaxels", fontsize=14)
        axes[1].grid(False)
        axes[1].legend()
        if save_plots != None:
            plt.savefig(save_plots + "/CenterMovement.png")
        plt.show()


    if center_y == True: 

        if np.max(y_coords_center[0] + parabola(wave, *popt_y)) > pix_y - 1:
            print("WARNING: There is too much displacement of the object. Moves outside the slit.")
            y_mov = np.zeros(N)
            y_mov_float = np.zeros(N)
            for i in range(N):
                if np.round(y_coords_center[0] + parabola(wave, *popt_y))[i] > pix_y - 1:
                    y_mov[i] == pix_y - 1
                    y_mov_float[i] == pix_y - 1

        if np.min(y_coords_center[0] + parabola(wave, *popt_y)) < 0:
            print("WARNING: There is too much displacement of the object. Moves outside the slit.")
            y_mov = np.zeros(N)
            y_mov_float = np.zeros(N)
            for i in range(N):
                if np.round(y_coords_center[0] + parabola(wave, *popt_y))[i] < 0:
                    y_mov[i] == 0
                    y_mov_float[i] == 0

        else:
            y_mov = np.round(y_coords_center[0] + parabola(wave, *popt_y))
            y_mov_float = y_coords_center[0] + parabola(wave, *popt_y)


    if center_y == False:

        y_mov = np.round(y_coords_center)
        y_mov_float = np.round(y_coords_center)


    y_variation = y_mov_float - y_mov
    y_dif_center = int(np.max(y_mov) - np.min(y_mov))
    long_y = int(pix_y + y_dif_center)
    y_offset = y_mov - y_mov[0] 


    if center_x == True:

        if np.max(x_coords_center[0] + parabola(wave, *popt_x)) > pix_x - 1:
            print("WARNING: There is too much displacement of the object. Moves outside the slit.")
            x_mov = np.zeros(N)
            x_mov_float = np.zeros(N)
            for i in range(N):
                if np.round(x_coords_center[0] + parabola(wave, *popt_x))[i] > pix_x - 1:
                    x_mov[i] == pix_x - 1
                    x_mov_float[i] == pix_x - 1

        if np.min(x_coords_center[0] + parabola(wave, *popt_x)) < 0:
            print("WARNING: There is too much displacement of the object. Moves outside the slit.")
            x_mov = np.zeros(N)
            x_mov_float = np.zeros(N)
            for i in range(N):
                if np.round(x_coords_center[0] + parabola(wave, *popt_x))[i] < 0:
                    x_mov[i] == 0
                    x_mov_float[i] == 0

        else:
            x_mov = np.round(x_coords_center[0] + parabola(wave, *popt_x))
            x_mov_float = x_coords_center[0] + parabola(wave, *popt_x)


    if center_x == False:

        x_mov = np.round(x_coords_center)
        x_mov_float = np.round(x_coords_center)


    x_variation = x_mov_float - x_mov
    x_dif_center = int(np.max(x_mov) - np.min(x_mov))
    long_x = int(pix_x + x_dif_center)
    x_offset = x_mov - x_mov[0] 


    medianas = np.zeros((pix_y, pix_x))
    for i in range(pix_x):
        for j in range(pix_y):
            medianas[j, i] = np.nanpercentile(data[:, j, i], 50)

    if center_y:
        new_long_y = int(long_y - 2*y_dif_center)
        corrected_ydata = np.zeros((N, new_long_y, pix_x))
        for i in range(pix_x):
            for j in range(new_long_y):
                for l in range(N):
                    if y_variation[l] > 0:
                        movement = int(y_dif_center + y_offset[l])
                        if (j + movement + 1) == pix_y:
                            corrected_ydata[l, j, i] = data[l, j + movement, i]
                        else:
                            corrected_ydata[l, j, i] = data[l, j + movement, i]*(1 - y_variation[l]) + data[l, j + movement + 1, i]*(y_variation[l])
                    if y_variation[l] < 0:
                        movement = int(y_dif_center + y_offset[l])
                        corrected_ydata[l, j, i] = data[l, j + movement, i]*(1 + y_variation[l]) + data[l, j + movement - 1, i]*(y_variation[l])*(-1)
    else: 
        corrected_ydata = data

    if center_x:
        new_long_x = int(long_x - 2*x_dif_center)
        corrected_data = np.zeros((N, new_long_y, new_long_x))
        for i in range(new_long_x):
            for j in range(new_long_y):
                for l in range(N):
                    if x_variation[l] > 0:
                        movement = int(x_dif_center + x_offset[l])
                        if (i + movement + 1) == pix_x:
                            corrected_data[l, j, i] = corrected_ydata[l, j, i+movement]
                        else:
                            corrected_data[l, j, i] = corrected_ydata[l, j, i+movement]*(1 - x_variation[l]) + corrected_ydata[l, j, i + movement + 1]*(x_variation[l])
                    if x_variation[l] < 0:
                        movement = int(x_dif_center + x_offset[l])
                        corrected_data[l, j, i] = corrected_ydata[l, j, i+movement]*(1 + x_variation[l]) + corrected_ydata[l, j, i+movement-1]*(x_variation[l])*(-1)

    else:
        corrected_data = corrected_ydata


    x_center = int(np.round(x_coords_center[0] + parabola(wave, *popt_x)[0] - x_dif_center))
    y_center = int(np.round(y_coords_center[0] + parabola(wave, *popt_y)[0] - y_dif_center))

    if x_center > pix_x - 1:
        x_center = pix_x - 1
    if y_center > pix_y - 1:
        y_center = pix_y - 1
    if x_center < 0:
        x_center = 0
    if y_center < 0:
        y_center = 0 

    print(" ")
    print("Center of the object: (y, x) = (", y_center,", ", x_center, ")")
    print("previous center (before the correction): (y, x) = (", y_center + y_dif_center, ", ", x_center + x_dif_center, ")")

    if plots == True:
        fig, axes = plt.subplots(1, 1, figsize=(18, 10))
        median_data = np.nanmedian(data[:, y_center, x_center])
        median_corrected_data = np.nanmedian(corrected_data[:, y_center, x_center])
        mean_both = median_data*0.5 + median_corrected_data*0.5
        axes.plot(wave, data[:, y_center + y_dif_center, x_center + x_dif_center] , c="red", label="Original data", linewidth=0.5)
        axes.plot(wave, corrected_data[:, y_center, x_center] , c="k", label="Corrected data", linewidth=0.5)
        axes.set_title("Spectra after the atmospheric dispersion correction", fontsize=21)
        axes.set_xlabel("Wavelenght", fontsize=18)
        axes.set_ylabel("Count", fontsize=18)
        axes.set_ylim(0, mean_both*max_plots)
        axes.legend()
        axes.grid(False)
        if save_plots != None:
            plt.savefig(save_plots + "/AtmosphericDispersion_Correction.png")
        plt.show()

    return corrected_data, (y_center, x_center)



def Sigma_clipping_adapted_for_IFU(cube_path, data=np.array([]), wave=np.array([]), A=5, window=100):
    """
    Identifies outliers using an adaptation of the sigma-clipping algorithm and replaces
    them with the median of their neighboring wavelengths. Returns a cleaner data cube.

    Args:
        cube_path (str): Path to the data cube.
        data (3D array): 3D array containing the data cube.
        wave (array): Array containing the wavelengths of the data.
        A (float): Amount of sigma away from the median to be considered an outlier.
        window (float): Width of the window for comparison and leveling of the data.

    Returns:
        clean_data (3D array): Data cube with outliers replaced.
    """
    if len(data) == 0:
        obs = get_pkg_data_filename(cube_path)
        hdul = fits.open(cube_path)
        header = hdul[0].header
        N = header["NAXIS3"]
        wave = np.zeros(N)
        #obtain the data and wavelength
        data = fits.getdata(obs, ext=0)[:, :, :]

        for i in range(N):
            wave[i] = (i+header["CRPIX3"])*header["CDELT3"] + header["CRVAL3"]
    
    if (len(data) > 0) and (len(wave) == 0):
        print("wave= ")
        print("Error: If you provide the data, also should provide the wavelength")

    if len(data) > 0:
        N = len(data)

    clean_data = data.copy()

    n = N - window//2 
    pix_x = len(data[0, 0, :])
    pix_y = len(data[0, :, 0])

    #dx = dim_x / (pix_x + 1)
    #dy = dim_y / (pix_y + 1)

    #distance_matrix = np.zeros((pix_y, pix_x))
    #for i in range(pix_x):
    #    for j in range(pix_y):
    #        distance_matrix[j, i] = np.sqrt( (dy*(j-centro[0]))**2 + (dx*(i-centro[1]))**2)

    #max_distance = np.max(distance_matrix)
    #radius = np.linspace(0, max_distance, 100)



    medianas = np.zeros((pix_y, pix_x))
    IQR = np.zeros(N)

    for lambda_wave in range(window//2, n):
        for i in range(pix_x):
            for j in range(pix_y):
            
                medianas[j, i] = np.nanpercentile(clean_data[lambda_wave-window//2:lambda_wave+window//2, j, i], 50)
        
        variable_y = (clean_data[lambda_wave]/medianas).reshape(pix_x*pix_y)
        #variable_x = distance_matrix.reshape(pix_x*pix_y)
        mediana = np.nanpercentile(variable_y, 50)
        deviation = variable_y - mediana
        upper_limit = np.nanpercentile(deviation, 75)
        lower_limit = np.nanpercentile(deviation, 25)
        IQR[lambda_wave] = upper_limit - lower_limit
        deviation = deviation.reshape(pix_y, pix_x)

        for lambda_x in range(pix_x):
            for lambda_y in range(pix_y):
                if (deviation[lambda_y, lambda_x] > A*upper_limit) or (deviation[lambda_y, lambda_x] < A*lower_limit):
                    clean_data[lambda_wave, lambda_y, lambda_x] = medianas[lambda_y, lambda_x]

    return clean_data
    

def optimal_radius_selection_IFU(cube_path, 
                                 center, 
                                 lower_lam, 
                                 upper_lam, 
                                 data=np.array([]), 
                                 wave=np.array([]), 
                                 dim_y=4, 
                                 dim_x=1.8, 
                                 error=3, 
                                 plots=True, 
                                 percentage=20,
                                 save_plots=None,
                                 N_max=None):
    """
    Determines the optimal radius for disk integration of spectra by analyzing a small, flat range
    of the spectra and observing its behavior as the radius of the integrated area increases.

    Args:
        cube_path (str): Path to the data cube.
        center (tuple): Central pixel coordinates of the object in the format (y-center, x-center).
        lower_lam (float): Lower limit in wavelength for the spectra to study.
        upper_lam (float): Upper limit in wavelength for the spectra to study.
        data (3D array): 3D array containing the data cube.
        wave (array): Array containing the wavelengths of the data.
        dim_x (float): Dimension in the x-direction of the slit in arcseconds.
        dim_y (float): Dimension in the y-direction of the slit in arcseconds.
        error (float): Percentage of error to consider as a deviation from the theoretical
                       signal-to-noise increase.
        plots (bool): True if you want to visualize plots.
        percentage (float): Percentage of the initial data to consider for fitting.
        save_plots (str): If a path is provided, the images are save in this directory.
        N_max (int): If a value is provided, the algorithm uses this amount of spaxels as the upper limit

    Returns:
        radius (float): Optimal radius for disk integration in arcseconds.
        radius_spaxel (int): Number of pixels within the optimal radius.
    """
    if len(data) == 0:
        obs = get_pkg_data_filename(cube_path)
        hdul = fits.open(cube_path)
        header = hdul[0].header
        N = header["NAXIS3"]
        wave = np.zeros(N)
        #obtain the data and wavelength
        data = fits.getdata(obs, ext=0)[:, :, :]
        dx = header["CDELT1"] * 60 * 60
        dy = header["CDELT1"] * 60 * 60
        pix_x = header["NAXIS1"]
        pix_y = header["NAXIS2"]

        for i in range(N):
            wave[i] = (i+header["CRPIX3"])*header["CDELT3"] + header["CRVAL3"]
    
    if (len(data) > 0) and (len(wave) == 0):
        print(" ")
        print("wave= ")
        print("Error: If you provide the data, also should provide the wavelength")

    if len(data) > 0:
        N = len(data)
        pix_x = len(data[0, 0, :])
        pix_y = len(data[0, :, 0])
        dx = dim_x / (pix_x + 1)
        dy = dim_y / (pix_y + 1)

    distance_matrix = np.zeros((pix_y, pix_x))
    for i in range(pix_x):
        for j in range(pix_y):
            distance_matrix[j, i] = np.sqrt( (dy*(j-center[0]))**2 + (dx*(i-center[1]))**2)

    max_distance = np.max(distance_matrix)
    r_N = 500
    radius = np.linspace(0, max_distance, r_N)

    upper_value = closest(upper_lam, wave)
    lower_value = closest(lower_lam, wave)

    n = len(radius)
    lambda_values = wave[lower_value:upper_value]
    StoN_radius = np.zeros(n)
    signal_r = np.zeros(n)
    noise_r = np.zeros(n)
    radius_spaxel = np.zeros(n)
    spec_r = np.zeros((n, len(lambda_values)))

    def linear(x, m, c):
        return m*x + c

    for r in range(1, n):
        flujo_sumado = np.zeros(len(lambda_values))
        pixel_count = 0
        for i in range(pix_x):
            for j in range(pix_y):
                ventana = data[lower_value:upper_value, j, i]
                if distance_matrix[j, i] < radius[r]:
                    flujo_sumado = flujo_sumado + ventana
                    pixel_count += 1
        radius_spaxel[r] = pixel_count
        signal_r[r] = np.sum(flujo_sumado) 

        popt, pcov = curve_fit(linear, lambda_values, flujo_sumado)
        m, c = popt
        noise_r[r] = np.std((flujo_sumado - linear(lambda_values, m, c)))
        spec_r[r] =  flujo_sumado
        StoN_radius[r] = signal_r[r] / noise_r[r]

    indiceRadio = np.int(percentage/100*r_N)
        

    def f(x, alpha, cte):
        return alpha*np.sqrt(x + cte) 

    # Fit a line to the spectrum
    popt, pcov = curve_fit(f, signal_r[1:indiceRadio], StoN_radius[1:indiceRadio])
    alpha, cte = popt

    for i in range(1, len(radius)):
        if np.abs(f(signal_r[i], alpha, cte) - StoN_radius[i]) > f(signal_r[i], alpha, cte)*(error/100):
            indiceRadio = i - 1
            break

    if plots==True:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))

        axes.set_title("SNR as function of the signal increase")
        axes.plot(signal_r[1:], StoN_radius[1:], c="k", label = "Real S/N")
        axes.set_ylabel("S/N", fontsize=18)
        axes.set_xlabel("log(Signal)", fontsize=18)
        axes.plot(signal_r[1:], f(signal_r[1:], alpha, cte), c="red", label="Teoritical S/N")
        axes.fill_between(signal_r[1:], (1-error/100)*f(signal_r[1:], alpha, cte), (1 + error/100)*f(signal_r[1:], alpha, cte), color="red", alpha=0.1)
        axes.plot(signal_r[indiceRadio], StoN_radius[indiceRadio],  ".", c="blue", markersize=10, label="Optimal Radius: "+ str(np.round(radius_spaxel[indiceRadio]))+ ' spaxels')
        axes.legend()
        axes.grid(False)
        if save_plots != None:
            plt.savefig(save_plots + "/SignalToNoiseIncrease.png")
        plt.show()

    if N_max != None:
        if radius_spaxel[indiceRadio] > N_max:
            N_max = min(radius_spaxel, key=lambda x: abs(N_max - x))
            indiceRadio = np.where(radius_spaxel == N_max)[0][0]
            #indiceRadio = radius_spaxel.index(N_max)

    print(" ")
    print("Optimal radius (arcsec): ", radius[indiceRadio])
    print("Number of spaxels inside the optimal radius: ", radius_spaxel[indiceRadio])

    return radius[indiceRadio], int(radius_spaxel[indiceRadio])

def Disk_integrate(cube_path, 
                   center, 
                   rad,
                   data=np.array([]), 
                   wave=np.array([]),
                   dim_x = 1.5,
                   dim_y = 4):
    """
    Performs disk integration on IFU observations using the provided radius.

    Args:
        cube_path (str): Path to the data cube.
        center (tuple): Central pixel coordinates of the object in the format (y-center, x-center).
        rad (float): Limit for the disk integration.
        data (3D array): 3D array containing the data cube.
        wave (array): Array containing the wavelengths of the data.
        dim_x (float): Dimension in the x-direction of the slit in arcseconds.
        dim_y (float): Dimension in the y-direction of the slit in arcseconds.

    Returns:
        flux_final (array): The final disk-integrated spectra.
        wave (array): The wavelength array for the final spectra.
    """
    if len(data) == 0:
        obs = get_pkg_data_filename(cube_path)
        hdul = fits.open(cube_path)
        header = hdul[0].header
        N = header["NAXIS3"]
        wave = np.zeros(N)
        dx = header["CDELT1"] * 60 * 60
        dy = header["CDELT1"] * 60 * 60
        pix_x = header["NAXIS1"]
        pix_y = header["NAXIS2"]
        #obtain the data and wavelength
        data = fits.getdata(obs, ext=0)[:, :, :]

        for i in range(N):
            wave[i] = (i+header["CRPIX3"])*header["CDELT3"] + header["CRVAL3"]
    
    if (len(data) > 0) and (len(wave) == 0):
        print(" ")
        print("wave= ")
        print("Error: If you provide the data, also should provide the wavelength")

    if len(data) > 0:
        N = len(data)
        pix_x = len(data[0, 0, :])
        pix_y = len(data[0, :, 0])
        dx = dim_x / (pix_x + 1)
        dy = dim_y / (pix_y + 1)

    pix_x = len(data[0, 0, :])
    pix_y = len(data[0, :, 0])

    distance_matrix = np.zeros((pix_y, pix_x))

    for i in range(pix_x):
        for j in range(pix_y):
            distance_matrix[j, i] = np.sqrt( (dy*(j-center[0]))**2 + (dx*(i-center[1]))**2)

    max_distance = np.max(distance_matrix)
    radius = np.linspace(0, max_distance, 100)

    flux_final= np.zeros(N)
    for i in range(pix_x):
        for j in range(pix_y):
            ventana = data[:, j, i]
            if distance_matrix[j, i] < rad:
                flux_final = flux_final + ventana

    return flux_final, wave

def integrate_extended(cube_path,
                       data=np.array([]), 
                       wave=np.array([]),
                       mode="all", 
                       discard=[], 
                       A=1.5, 
                       lower=None,
                       upper=None, 
                       save_plots=None, 
                       aspect=0.2564):
    """
    Integrates the pixels of the data cube to create a single final spectrum. 
    This function operates in two modes: "all", which integrates all pixels except those specified
    in the discard parameter, and "drop," which analyzes if there are pixels too different from the others.
    In the "drop" mode, any pixel detected as an outlier is not considered in the integration. 
    "drop fitting is a variation of "drop" mode but uses a line fitting to get rid of the linear tendency of the data. 
    For this last one is important to choose a linear section of the spectra with the parameters upper ad lower.


    Args:
        cube_path (str): Path to the data cube.
        data (3D array): 3D array containing the data cube.
        wave (array): Array containing the wavelengths of the data.
        mode (str): Mode of integration to be used, which can be "all" or "drop."
        discard (array): Array of tuples indicating the pixels that should not be considered in the integration.
        A (float): Parameter to determine if a pixel is an outlier (recommended: 1.5).
        lower (float): Lower limit in wavelength for studying the dispersion in the pixels.
        upper (float): Upper limit in wavelength for studying the dispersion in the pixels.
        save_plots (str): Path where the generated images should be saved.
        aspect (float): Value to control the aspect ratio of the displayed images.

    Returns:
        flux_final (array): The final integrated spectrum.
        wave (array): The wavelength array for the final spectrum.
    """

    if len(data) == 0:
        obs = get_pkg_data_filename(cube_path)
        hdul = fits.open(cube_path)
        header = hdul[0].header
        N = header["NAXIS3"]
        wave = np.zeros(N)
        pix_x = header["NAXIS1"]
        pix_y = header["NAXIS2"]
        #obtain the data and wavelength
        data = fits.getdata(obs, ext=0)[:, :, :]

        for i in range(N):
            wave[i] = (i+header["CRPIX3"])*header["CDELT3"] + header["CRVAL3"]

    if lower != None:
        lower = closest(lower, wave)
    if upper != None:
        upper = closest(upper, wave)
    
    if (len(data) > 0) and (len(wave) == 0):
        print(" ")
        print("wave= ")
        print("Error: If you provide the data, also should provide the wavelength")

    if len(data) > 0:
        N = len(data)
        pix_x = len(data[0, 0, :])
        pix_y = len(data[0, :, 0])


    pix_x = len(data[0, 0, :])
    pix_y = len(data[0, :, 0])

    if mode =="all":
        flux_final= np.zeros(N)
        for i in range(pix_x):
            for j in range(pix_y):
                if (j, i) not in discard:
                    flux_final = flux_final + data[:, j, i]

        return flux_final, wave

    if mode == "drop":

        medianas = np.zeros((pix_y, pix_x))
        for i in range(pix_x):
            for j in range(pix_y):
                medianas[j, i] = np.nanpercentile(data[lower:upper, j, i], 50)
        medianas = medianas / np.max(medianas)
        medianas_raw = medianas.copy()

        leveled_data = data / medianas

        median_flux = np.zeros(N)
        for l in range(N):
            median_flux[l] = np.nanpercentile(leveled_data[l, :, :], 50)
        flux_final= np.zeros(N)
        deviations = np.zeros((pix_y, pix_x))
        for i in range(pix_x):
            for j in range(pix_y):
                flux = leveled_data[:, j, i]
                dif = flux[lower:upper] - median_flux[lower:upper]
                deviations[j, i] = np.std(dif)
        
        iqr = np.nanpercentile(deviations, 75) - np.nanpercentile(deviations, 25)
        limit = np.nanpercentile(deviations, 75) + A*iqr
        #deviations = deviations / iqr

        for i in range(pix_x):
            for j in range(pix_y):
                if (j, i) not in discard:
                    if deviations[j, i] > limit:
                        print("pixel (y="+str(j)+", x="+str(i)+") considered noisy")
                        medianas[j, i] = None
                        pass
                    elif deviations[j, i] < limit:
                        flux_final = flux_final + data[:, j, i]
                if (j, i) in discard:
                    print("pixel (y="+str(j)+", x="+str(i)+") discarded")
                    medianas[j, i] = None



        fig, axes = plt.subplots(1, 3, figsize=(6, 21))
        axes[0].set_title("Visualization")
        im0 = axes[0].imshow(medianas_raw, origin="lower", aspect=aspect)
        #bar0 = plt.colorbar(im0)
        axes[0].set_ylabel("y-spaxels")
        axes[0].set_xlabel("x-spaxels")

        axes[2].set_title("Integrated pixels")
        im2 = axes[2].imshow(medianas, origin="lower", aspect=aspect)
        #bar2 = plt.colorbar(im2)
        axes[2].set_ylabel("y-spaxels")
        axes[2].set_xlabel("x-spaxels")

        axes[1].set_title("Noise per pixel")
        im1 = axes[1].imshow(deviations, origin="lower", aspect=aspect)
        #bar1 = plt.colorbar(im1)
        axes[1].set_ylabel("y-spaxels")
        axes[1].set_xlabel("x-spaxels")
        if save_plots != None:
            plt.savefig(save_plots + "/Visualization_RawData.png")

        return flux_final, wave
    

    if mode == "drop fitting":

        def linear(x, m, c):
            return x*m+c

        n = int(upper-lower)
        centered_data = np.zeros((n, pix_y, pix_x))

        
        for j in range(pix_y):
            for i in range(pix_x):
                popt, pcov = curve_fit(linear, wave[lower:upper], data[lower:upper, j, i])
                m, c = popt
                centered_data[:, j, i] = data[lower:upper, j, i] - linear(wave[lower:upper], m, c)

        medianas = np.zeros((pix_y, pix_x))
        for i in range(pix_x):
            for j in range(pix_y):
                medianas[j, i] = np.nanpercentile(data[lower:upper, j, i], 50)
        medianas = medianas / np.max(medianas)
        medianas_raw = medianas.copy()

        median_centered = np.zeros(n)
        leveled_data = centered_data / medianas
        for l in range(n):
            median_centered[l] = np.nanpercentile(leveled_data[l], 50)

        flux_final= np.zeros(N)
        deviations = np.zeros((pix_y, pix_x))
        for i in range(pix_x):
            for j in range(pix_y):
                flux = leveled_data[:, j, i]
                dif = flux - median_centered
                deviations[j, i] = np.std(dif)

        iqr = np.nanpercentile(deviations, 75) - np.nanpercentile(deviations, 25)
        limit = np.nanpercentile(deviations, 75) + A*iqr

        for i in range(pix_x):
            for j in range(pix_y):
                if (j, i) not in discard:
                    if deviations[j, i] > limit:
                        print("pixel (y="+str(j)+", x="+str(i)+") considered noisy")
                        medianas[j, i] = None
                        pass
                    elif deviations[j, i] < limit:
                        flux_final = flux_final + data[:, j, i]
                if (j, i) in discard:
                    print("pixel (y="+str(j)+", x="+str(i)+") discarded")
                    medianas[j, i] = None

        fig, axes = plt.subplots(1, 3, figsize=(6, 21))
        axes[0].set_title("Visualization")
        im0 = axes[0].imshow(medianas_raw, origin="lower", aspect=aspect)
        #bar0 = plt.colorbar(im0)
        axes[0].set_ylabel("y-spaxels")
        axes[0].set_xlabel("x-spaxels")

        axes[2].set_title("Integrated pixels")
        im2 = axes[2].imshow(medianas, origin="lower", aspect=aspect)
        #bar2 = plt.colorbar(im2)
        axes[2].set_ylabel("y-spaxels")
        axes[2].set_xlabel("x-spaxels")

        axes[1].set_title("Noise per pixel")
        im1 = axes[1].imshow(deviations, origin="lower", aspect=aspect)
        #bar1 = plt.colorbar(im1)
        axes[1].set_ylabel("y-spaxels")
        axes[1].set_xlabel("x-spaxels")
        if save_plots != None:
            plt.savefig(save_plots + "/Visualization_RawData.png")

        return flux_final, wave

    

def save_file(path_to_save, 
              header, 
              data,
              radius = None, 
              radius_spaxel = None, 
              center = None, 
              comment = None, 
              lower_limit=None,
              upper_limit=None,
              correct_center_x=None, 
              correct_center_y=None,
              look_center_x=None,
              look_center_y=None,
              A_sc=None,  
              window_sc=None, 
              percentage=None, 
              error=None):
    """
    Save the final data into a FITS file. Also writes in the header all the important information about the final data.
    This is dessign for disk-integrated spectra.

    Args:
        path_to_save (str): Path where the data will be saved.
        header (str): Header of the raw data.
        data (float): Final data that needs to be saved.
        radius (float): Optimal radius for disk integration in arcseconds.
        radius_spaxel (int): Number of pixels within the optimal radius.
        center (tuple): Calculated center of the new data cube.
        lower_limit (float): Lower limit in wavelength for optimal radius selection.
        upper_limit (float): Upper limit in wavelength for optimal radius selection.
        corrected_center_x (bool): True if atmospheric correction is needed in the x-direction.
        corrected_center_y (bool): True if atmospheric correction is needed in the y-direction.
        look_center_x (tuple): Higher and lower values for parabolic fit in the x-direction.
        look_center_y (tuple): Higher and lower values for parabolic fit in the y-direction.
        plots (bool): True to visualize plots.
        max_plots (float): Factor to set vertical plot limits. ylim = max_plots * data_median.
        A_sc (float): Amount of sigma away from the median to consider an outlier.
        window_sc (float): Width of the window for comparison and data leveling.
        percentage (float): Percentage of initial data to consider for fitting.
        error (float): Percentage of error to consider as a deviation from the theoretical signal-to-noise increase.

    Returns:
        None
    """

    hdr = header

    if radius_spaxel != None:
        hdr["SH SPAXELS R"] = radius_spaxel
    if radius_spaxel == None:
        hdr["SH SPAXELS R"] = "No information"

    if radius != None:
        hdr["SH ANGULAR R"] = radius
    if radius == None:
        hdr["SH ANGULAR R"] = "No information"

    if A_sc != None:
        hdr["SH A SIGMA CLIPPING"] = A_sc
        hdr["SH WINDOW SIGMA CLIPPING"] = window_sc
    if A_sc == None:
        hdr["SH A SIGMA CLIPPING"] = "No information"
        hdr["SH WINDOW SIGMA CLIPPING"] = "No information"

    if center != None:
        hdr["SH OBJECT CENTER Y"] = center[0]
        hdr["SH OBJECT CENTER X"] = center[1]
    if center == None:
        hdr["SH OBJECT CENTER Y"] = "No information"
        hdr["SH OBJECT CENTER X"] = "No information"

    if comment != None:
        hdr["SH COMMENT"] = comment
    if comment == None:
        hdr["SH COMMENT"] = "No comments"

    if lower_limit != None:
        hdr["SH LOWER LIMIT"] = lower_limit
        hdr["SH UPPER LIMIT"] = upper_limit
    if lower_limit == None:
        hdr["SH LOWER LIMIT"] = "No information"
        hdr["SH UPPER LIMIT"] = "No information"

    if correct_center_x != None:
        hdr["SH X CENTER CORRECTION"] = correct_center_x
    if correct_center_y != None:
        hdr["SH Y CENTER CORRECTION"] = correct_center_y
    if correct_center_x == None:
        hdr["SH X CENTER CORRECTION"] = "No information"
    if correct_center_y == None:
        hdr["SH Y CENTER CORRECTION"] = "No information"

    if look_center_x != None:
        hdr["SH X CENTER CORRECTION RANGE"] = look_center_x
    if look_center_y != None:
        hdr["SH Y CENTER CORRECTION RANGE"] = look_center_y
    if look_center_x == None:
        hdr["SH X CENTER CORRECTION RANGE"] = "No information"
    if look_center_y == None:
        hdr["SH Y CENTER CORRECTION RANGE"] = "No information"

    if percentage != None:
        hdr["SH PERCENTAGE FOR FITTING"] = look_center_x
    if percentage == None:
        hdr["SH PERCENTAGE FOR FITTING"] = "No information"
    
    if error != None:
        hdr["SH ERROR ACEPTED"] = error
    if percentage == None:
        hdr["SH ERROR ACEPTED"] = "No information"
    
    empty_primary = fits.PrimaryHDU(data, header=hdr)

    hdul = fits.HDUList([empty_primary])
    hdul.writeto(path_to_save, overwrite=True)

def save_file_extended(path_to_save, 
                       header, 
                       data, 
                       mode_used=None,
                       discard_pixels=None,
                       comment=None,
                       lower=None,
                       upper=None):
    """
    Save the final data into a FITS file and write important information into the header.
    This function is designed for cases where the observation is entirely within the studied object.

    Args:
        path_to_save (str): Path where the data will be saved.
        header (str): Header of the raw data.
        data (float): Final data to be saved.
        mode_used (str): Mode of integration to be used, which can be "all" or "drop."
        discard_pixels (list): List of tuples indicating the pixels that should not be considered in the integration.
        comment (str): Special comment to be saved in the header of the final FITS file.
        lower (float): Lower limit in wavelength for studying the dispersion in the pixels.
        upper (float): Upper limit in wavelength for studying the dispersion in the pixels.

    Returns:
        None    
    """

    hdr = header

    if mode_used != None:
        hdr["SH DI MODE USED"] = mode_used
    if mode_used == None:
        hdr["SH DI MODE USED"] = "No information"
    if discard_pixels != None:
        hdr["SH DISCARD PIXELS FOR DI"] = str(discard_pixels)
    if discard_pixels == None:
        hdr["SH DISCARD PIXELS FOR DI"] = "No information"
    if comment != None:
        hdr["SH COMMENT"] = comment
    if comment == None:
        hdr["SH COMMENT"] = "No comments"
    if lower != None:
        hdr["SH LOWER LIMIT"] = lower
    if lower == None:
        hdr["SH LOWER LIMIT"] = "No comments"
    if upper != None:
        hdr["SH UPPER LIMIT"] = upper
    if upper == None:
        hdr["SH UPPER LIMIT"] = "No comments"
    
    empty_primary = fits.PrimaryHDU(data, header=hdr)

    hdul = fits.HDUList([empty_primary])
    hdul.writeto(path_to_save, overwrite=True)


def process_my_ifu_obs(fits_path,
                       lower_limit, 
                       upper_limit, 
                       correct_center_x=True, 
                       correct_center_y=True, 
                       look_center_x=None, 
                       look_center_y=None, 
                       plots=True, 
                       max_plots=3, 
                       A_sc=3, 
                       window_sc=100, 
                       percentage=25, 
                       error=1, 
                       path_to_save = None, 
                       comment = None, 
                       save_plots = None,
                       N_max=None):
    """
    Computes a single disk-integrated spectrum from observations with IFUs. The algorithm involves three steps:
    1. Corrects atmospheric dispersion (optional for x and y directions).
    2. Identifies and replaces outliers using an adapted sigma clipping algorithm.
    3. Selects the optimal radius for disk integration based on central pixel analysis.

    Args:
        fits_path (str): Path to the data cube.
        lower_limit (float): Lower limit in wavelength for optimal radius selection.
        upper_limit (float): Upper limit in wavelength for optimal radius selection.
        corrected_center_x (bool): True if atmospheric correction is needed in the x-direction.
        corrected_center_y (bool): True if atmospheric correction is needed in the y-direction.
        look_center_x (tuple): Higher and lower values for parabolic fit in the x-direction.
        look_center_y (tuple): Higher and lower values for parabolic fit in the y-direction.
        plots (bool): True to visualize plots.
        max_plots (float): Factor to set vertical plot limits. ylim = max_plots * data_median.
        A_sc (float): Amount of sigma away from the median to consider an outlier.
        window_sc (float): Width of the window for comparison and data leveling.
        percentage (float): Percentage of initial data to consider for fitting.
        error (float): Percentage of error to consider as a deviation from the theoretical signal-to-noise increase.
        path_to_save (str): Path where the final data will be saved.
        comment (str): Special comment that will be saved in the header of the final FITS file.
        save_plots (str): If a path is provided, the images are save in this directory.
        N_max (int): If a value is provided, the algorithm uses this amount of spaxels as the upper limit

    Returns:
        final_data (array): Disk-integrated spectra of the data-cube
        wave (array): Wavelength of the final spectra
    """
    
    data, wave, pix_x, pix_y, dx, dy = visualize(fits_path, save_plots)

    corrected_data, center = Atmospheric_dispersion_correction(" ", data, wave, 
                                                               center_x=correct_center_x, 
                                                               center_y=correct_center_y, 
                                                               range_x=look_center_x, 
                                                               range_y=look_center_y, 
                                                               plots=plots, 
                                                               max_plots=max_plots,
                                                               save_plots=save_plots)
    
    clean_data = Sigma_clipping_adapted_for_IFU("", 
                                                data=corrected_data, 
                                                wave=wave, 
                                                A=A_sc, 
                                                window=window_sc)

    fig, axes = plt.subplots(1, 1, figsize=(18, 10))
    median = np.median(corrected_data[:, center[0], center[1]])
    axes.vlines(lower_limit, 0, median*3, linestyle="--", color="orange", label="Limits for finding the best radius.")
    axes.vlines(upper_limit, 0, median*3, linestyle="--", color="orange")
    axes.plot(wave, corrected_data[:, center[0], center[1]], c="red", linewidth=0.5, label="Raw data")
    axes.plot(wave, clean_data[:, center[0], center[1]], c="k", linewidth=0.5, label="Data with Sigma-Clipping")
    axes.set_title("Data with and without Sigma clipping", fontsize=22)
    axes.set_xlabel("Wavelength", fontsize=18)
    axes.set_ylabel("Count", fontsize=18)
    axes.legend()
    axes.set_ylim(0, median*3)
    if save_plots != None:
        plt.savefig(save_plots + "/SigmaClipping.png")

    radius, radius_spaxels = optimal_radius_selection_IFU(" ", 
                                                          center, 
                                                          lower_limit, 
                                                          upper_limit, 
                                                          data=clean_data, 
                                                          wave=wave, 
                                                          percentage=percentage, 
                                                          error=error,
                                                          dim_x=dx*(pix_x + 1), 
                                                          dim_y=dy*(pix_y + 1),
                                                          save_plots=save_plots,
                                                          N_max=N_max)
    
    final_data, wave = Disk_integrate(" ", 
                                center, 
                                radius, 
                                data=clean_data, 
                                wave=wave, 
                                dim_x=dx*(pix_x + 1), 
                                dim_y=dy*(pix_y + 1))
    
    fig, axes = plt.subplots(1, 1, figsize=(18, 10))
    median = np.median(final_data)
    axes.plot(wave, final_data, c="k", linewidth=0.5)
    axes.set_title("Final data after the Integration", fontsize=22)
    axes.set_xlabel("Wavelength", fontsize=18)
    axes.set_ylabel("Count", fontsize=18)
    axes.legend()
    axes.set_ylim(0, median*3)
    if save_plots != None:
        plt.savefig(save_plots + "/Final_DiskIntegrated_Spectra.png")

    if path_to_save != None:
        hdul = fits.open(fits_path)
        save_file(path_to_save, 
                  hdul[0].header, 
                  final_data, 
                  radius=radius, 
                  radius_spaxel=radius_spaxels,  
                  center=center, 
                  comment=comment, 
                  lower_limit=lower_limit,
                  upper_limit=upper_limit,
                  correct_center_x=correct_center_x, 
                  correct_center_y=correct_center_y,
                  look_center_x=look_center_x,
                  look_center_y=look_center_y,
                  A_sc=A_sc,  
                  window_sc=window_sc, 
                  percentage=percentage, 
                  error=error)
    
    return final_data, wave

def process_ifu_extended(fits_path,
                         plots=True,
                         max_plots=3,
                         A_sc=3,
                         window_sc=100, 
                         discard=np.array([]),
                         mode_di = "drop",
                         A_di=3,
                         path_to_save = None,
                         comment = None,
                         save_plots = None,
                         lower=None,
                         upper=None):
    """
    Compute a single integrated spectrum from observations with IFUs. The algorithm involves two steps:
    1. Identifies and replaces outliers using an adapted sigma-clipping algorithm.
    2. Integrates the data cube with the integration mode provided. 

    Args:
        fits_path (str): Path to the data cube.
        plots (bool): True to visualize plots.
        max_plots (float): Factor to set vertical plot limits. ylim = max_plots * data_median.
        A_sc (float): The number of standard deviations away from the median to consider as an outlier.
        window_sc (float): Width of the window for comparison and data leveling.
        mode_di (str): Mode of integration to be used, which can be "all" or "drop."
        discard (list): List of tuples indicating the pixels that should not be considered in the integration.
        A_di (float): Parameter to consider a pixel an outlier (recommended: 1.5).
        path_to_save (str): Path where the final data will be saved.
        comment (str): Special comment to be saved in the header of the final FITS file.
        save_plots (str): If a path is provided, the images will be saved in this directory.
        lower (float): Lower limit in wavelength for studying the dispersion in the pixels.
        upper (float): Upper limit in wavelength for studying the dispersion in the pixels.

    Returns:
        final_data (array): Integrated spectra of the data cube.
        wave (array): Wavelength of the final spectra.
    """
    
    data, wave, pix_x, pix_y, dx, dy = visualize(fits_path, save_plots, plots=False)

    clean_data = Sigma_clipping_adapted_for_IFU("", 
                                                data=data, 
                                                wave=wave, 
                                                A=A_sc, 
                                                window=window_sc)
    
    X = pix_x//2
    Y = pix_y//2 
    fig, axes = plt.subplots(1, 1, figsize=(18, 10))
    median = np.median(data[:, Y, X])
    axes.plot(wave, data[:, Y, X], c="red", linewidth=0.5, label="Raw data")
    axes.plot(wave, clean_data[:, Y, X], c="k", linewidth=0.5, label="Data with Sigma-Clipping")
    axes.set_title("Data with and without Sigma clipping", fontsize=22)
    axes.set_xlabel("Wavelength", fontsize=18)
    axes.set_ylabel("Count", fontsize=18)
    axes.legend()
    axes.set_ylim(0, median*max_plots)
    if save_plots != None:
        plt.savefig(save_plots + "/SigmaClipping.png")

    final_data, wave = integrate_extended("", 
                                          data=clean_data, 
                                          wave=wave,
                                          mode=mode_di,
                                          discard=discard,
                                          A=A_di,
                                          lower=lower,
                                          upper=upper)
    
    fig, axes = plt.subplots(1, 1, figsize=(18, 10))
    median = np.median(final_data)
    axes.plot(wave, final_data, c="k", linewidth=0.5)
    axes.set_title("Final data after the Integration", fontsize=22)
    axes.set_xlabel("Wavelength", fontsize=18)
    axes.set_ylabel("Count", fontsize=18)
    axes.legend()
    axes.set_ylim(0, median*max_plots)
    if save_plots != None:
        plt.savefig(save_plots + "/Final_DiskIntegrated_Spectra.png")

    if path_to_save != None:
        hdul = fits.open(fits_path)
        save_file_extended(path_to_save, 
                                    hdul[0].header, 
                                    final_data,
                                    mode_used=mode_di,
                                    discard_pixels=discard,
                                    comment=comment,
                                    lower=lower,
                                    upper=upper)
    return final_data, wave