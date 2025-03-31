# -*- coding: utf-8 -*-
"""

Analysis of motion heterogeneity, based on a density-colored scatter plot of 
track-wise standard deviations of X and Y displacements, normalized (or unnormalized) to the standard deviation 
of X and Y displacements across all tracks, for experimental data.

"""

import csv 
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import gaussian_kde
from scipy.integrate import simps

delta_x = 0.005 #time interval in sec for linear interpolation of collected tracks
min_trajectory_length = 10 #minimum lag time for MSD calculation, interpolation
r2 = 0.9 # filter treshhold for power law fit to MSD
scale = (1/0.0586000) #scale to receive x and y in um 
time_scale = 1 #cale to convert into sec 
save = False
experiment_condition = 'r6cf0p1_l1d5'
step = 2 # 10 ms
colorbar = 'plasma'


ExperDirectory = 'D:/082124_082524/'
experiments = [f for f in os.listdir(ExperDirectory) if os.path.isdir(os.path.join(ExperDirectory, f))]
experiments = [['0.24ugml doxycycline 4pm 35oC', '0.24 ugml doxycycline 5pm repeat 2'], ['0.24ugml doxycycline 12_30pm  2d 35oC', '0.24 ugml doxycycline 11am 2day repeat 2'], ['0.5ugml doxycycline 6pm 35oC', '0.5ugml doxycycline 12am repeat2 35oC'], ['0.5ugml doxycycline 2pm 2d 35oC','0.5ugml doxycycline 19_30m day2 repeat2 35oC'], ['1ugml doxycycline 7_20pm 35oC', '1ugml doxycycline 10_30am repeat2 35oC'], ['1ugml doxycycline 3pm 2d 35oC', '1ugml doxycycline 8_30am 2day repeat2 35oC'], ['GEM 3d 35oC', 'GEM 35 repeat', 'GEM 35 repeat 3'], [ 'GEM High Expression','GEM High Expression repeat 2', 'GEM High Expression repeat 3'], [ 'GEM Low Expression', 'GEM Low Expression repeat 2', 'GEM Low Expression repeat 3']]

#parameters for figures
plt.rcParams['figure.figsize'] = [16, 10]  # Adjust based on your needs
plt.rcParams.update({
    'font.family': 'Times New Roman', 
    'font.weight': 'bold', 
    'axes.labelweight': 'bold', 
    'axes.titleweight': 'bold', 
    'font.size': 50,
    'axes.titlesize': 50,
    'axes.labelsize': 50,
    'legend.fontsize': 40,
    'xtick.labelsize': 40,  
    'ytick.labelsize': 40,
}) 

def plot(x, y, xname = None, yname = None, std = None, plot_number = 'None', title = None):
    plt.figure(num = int(plot_number) if plot_number != 'None' else plt.figure())
    plt.plot(x, y, marker='.', linestyle = '-', linewidth = 2, markersize = 2)
    
    if std is not None:
        plt.errorbar(x, y, yerr=std, ecolor='green', label=f'Standard Deviation {yname}', alpha = 0.5)
    
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.title(title)
    plt.grid(True, zorder=1)


#function for interpolation np.interpolation: 
#x ~ time, delta_x ~ time displacement, y ~ data to interpolate
def interpolation(delta_x, x, y, xname, yname, plot_number):
    x_interp = [x[0]]
    #y_interp = []
    point = x[0]
    while point <= x[-1]:
        point = point + delta_x
        x_interp.append(point)
    y_interp = np.interp(x_interp, x, y)
    #plt.figure(int(plot_number)) if plot_number != 'None' else plt.figure()
    #plt.scatter(x, y, s = 5)
    #plt.plot(x_interp, y_interp, color = 'red', linewidth = 1)
    #plt.ylabel(yname)
    #plt.xlabel(xname)
    #plt.grid(True, zorder=1)
    return (x_interp, y_interp) 

def PDF(data):
    # Create histogram
    counts, bin_edges = np.histogram(data, bins=30, density=False)
    
    # Calculate bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate the integral using the trapezoidal rule
    integral_value = np.trapz(counts, bin_centers)
    
    # Normalize the histogram
    pdf = counts / integral_value
    
    # Verify the integral of the normalized histogram
    #normalized_integral_value = np.trapz(pdf, bin_centers)
    #print(f"Integral of the histogram: {integral_value}")
    #print(f"Integral of the normalized histogram (should be 1): {normalized_integral_value}")

    return bin_centers, pdf


def density_colored_scatter_plot(data_x, data_y, density_min = 0,  density_max = 200, coord_lim = [0, 0.4], mode = ''):
    # Example data (use your sigmaX_cells and sigmaY_cells)
    x = np.array(data_x)
    y = np.array(data_y)
    
    # Calculate the point density
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
        
    # Sort the points by density (optional, for nicer visualization)
    idx = sorted(range(len(density)), key=lambda i: density[i])
    x, y, density = [x[i] for i in idx], [y[i] for i in idx], [density[i] for i in idx]
    
    plt.figure()
    sc = plt.scatter(x, y, c=density, cmap='plasma', s=50, vmin=density_min, vmax=density_max)
    plt.xlim(coord_lim[0], coord_lim[1])
    plt.ylim(coord_lim[0], coord_lim[1])
    if mode == 'unnormalized':
        plt.xlabel(r'$\sigma(\Delta X)$, $\mu$m')
        plt.ylabel(r'$\sigma(\Delta Y)$, $\mu$m')
    if mode == 'normalized':
        plt.xlabel('$\sigma$($\Delta$X) / $\sigma$($\Delta$X)$_{cell}$')
        plt.ylabel('$\sigma$($\Delta$Y) / $\sigma$($\Delta$Y)$_{cell}$') 
    # Add the colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('PDF')
    if save:
        plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure5_2D_{mode}_{experiment[0]}.png', dpi=300)
        plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure5_2D_{mode}_{experiment[0]}.pdf', dpi=300)
    plt.show()

def PDF_KDE_3D(data, kde_min_filter, kde_max_filter, zlim = [0,300], Value='Value', name_PDF = ''):
    kde = gaussian_kde(data)  # Perform KDE on transposed data (x, y)
    
    # Define the range for the PDF
    x_vals = np.linspace(kde_min_filter, kde_max_filter, 500)  # Range for x-axis
    y_vals = np.linspace(kde_min_filter, kde_max_filter, 500)  # Range for y-axis
    X, Y = np.meshgrid(x_vals, y_vals)  # Create a meshgrid for the 2D space
    
    positions = np.vstack([X.ravel(), Y.ravel()])  # Stack X, Y into a 2D array
    
    Z = kde(positions).reshape(X.shape)  # Evaluate KDE and reshape to grid dimensions
    
    # Plot the result
    fig = plt.figure(figsize=(20, 16))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.grid(color='gray')
    
    # Plot 3D surface
    ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')
    
    # Labels and title
    if 'normalized' in name_PDF:
        ax.set_xlabel('$\sigma$($\Delta$X) / $\sigma$($\Delta$X)$_{cell}$', labelpad=40)
        ax.set_ylabel('$\sigma$($\Delta$Y) / $\sigma$(Y)$_{cell}$', labelpad=40)
    else:
        ax.set_xlabel('$\sigma$(\Delta$X), $\mu$m', labelpad=40)
        ax.set_ylabel('$\sigma$(\Delta$Y), $\mu$m', labelpad=40)
        
    ax.set_zlabel('Probability Density', labelpad=40)
    ax.set_zlim(zlim[0],zlim[1])
    
    if save:
        plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure5_3D_{experiment[0]}_{name_PDF}.png', dpi=300)
        plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure5_3D_{experiment[0]}_{name_PDF}.pdf', dpi=300)

    plt.show()
    
    # Calculate the integral of the KDE using Simpson's Rule
    integral = simps(simps(Z, x_vals), y_vals)  # Nested Simpson's rule for 2D
    
    print(f"Integral of the KDE: {integral:.6f} (should be close to 1)")


def skewness(data, normalization_status = ''):
    # Calculate skewness
    x_skew = skew(data[0])
    y_skew = skew(data[1])
    
    # Determine symmetry status
    symmetry_status = "Approximately symmetric" if abs(x_skew) < 0.05 and abs(y_skew) < 0.05 else "Asymmetric"
    
    # Print results
    print({experiment[0]}, normalization_status)
    print(f"Skewness (X-axis): {x_skew:.6f}")
    print(f"Skewness (Y-axis): {y_skew:.6f}")
    print(f"The KDE is {symmetry_status.lower()}.")
    
    # # Define CSV file name
    # csv_filename = "skewness_results.csv"
    
    # # Write to CSV (append mode)
    # with open(csv_filename, mode='a', newline='') as file:
    #     writer = csv.writer(file)
        
    #     # Write header if file is empty
    #     if file.tell() == 0:
    #         writer.writerow([f'{experiment[0]}', "X_skew", "Y_skew", "Symmetry Status"])
        
    #     # Write data
    #     writer.writerow([normalization_status, x_skew, y_skew, symmetry_status])


for experiment in experiments: 
    
    #list to collect track-wise standard deviations of X and Y displacements from the experiment
    sigma_X_displacement_experiment = [] 
    sigma_Y_displacement_experiment = []
    
    #track-wise standard deviations of X and Y displacements, normalized to the standard deviation 
    #of X and Y displacements across all tracks.
    
    sigma_X_displacement_normalized_experiment = [] 
    sigma_Y_displacement_normalized_experiment = []
    
    for repeat in experiment: 
        # Specify the directory you want to list folders from
        parent_dir = f'{ExperDirectory}{repeat}/'
        files = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
        
        #Van Gogh distribution sigma (std dv)
    
        
        for file in files:
            #print(f'{repeat} {file}')
            
            sigma_X_displacement_cell_averaged = [] 
            sigma_Y_displacement_cell_averaged = []
            
            sigma_X_displacement_cell = [] 
            sigma_Y_displacement_cell = []
            
            trajectories = [] # list with trajectories for plotting
            
            # Check if the directory path exists
            if not os.path.exists(f'{parent_dir}/{file}/{experiment_condition}/'):
                continue
            
            #Convertation of converted_data to the list ['trajectory №', [time_o], [x], [y], [time_frame]] for plotting
            with open(f'{parent_dir}{file}/{experiment_condition}/converted_data_length10.csv') as data:
                data_file = csv.reader(data, delimiter=',')
            
                check = False
                for row in data_file:
                    if row[0] == 'Trajectory':
                        if check == True:
                            one_tr = (name, time, x, y, time_frame)
                            if len(x) >=  min_trajectory_length:
                                trajectories.append(one_tr)
                        check = True
                        name = (row[0]+' '+row[1])
                        x = [] # time if every trajectory starts from 0sec
                        y = []
                        time = []
                        time_frame = [] #real time of every trajectory  
                    if row[0] != 'frame' and row[0] != 'Trajectory':
                        time.append(float(row[3])/time_scale) 
                        x.append(float(row[1])/scale) 
                        y.append(float(row[2])/scale) 
                        time_frame.append(float(row[4])/time_scale)
                        
                one_tr = (name, time, x, y, time_frame)
                if len(x) >=  min_trajectory_length:
                    trajectories.append(one_tr)

            #collection data for Heterogenity check
            for trajectory in trajectories:
                
                X_displacement = []
                Y_displacement = []
                
                name = trajectory[0]
                t = trajectory[1]
                x = trajectory[2]
                y = trajectory[3]
                t_frame = trajectory[4]
                
                #Collection of displacements and itrs standard deviations 
                for i in range(0, len(t)-2, step):
                    time_interval = t[i+1] - t[i] 
                    x_displacement = x[i+1] - x[i]
                    y_displacement = y[i+1] - y[i]
                    
                    X_displacement.append(x_displacement)
                    Y_displacement.append(y_displacement)
                    
                    sigma_X_displacement_cell_averaged.append(x_displacement)
                    sigma_Y_displacement_cell_averaged.append(y_displacement)
                    
                sigma_Y_displacement_experiment.append(np.std(Y_displacement, ddof = 1))
                sigma_X_displacement_experiment.append(np.std(X_displacement, ddof = 1))
                
                sigma_Y_displacement_cell.append(np.std(Y_displacement, ddof = 1))
                sigma_X_displacement_cell.append(np.std(X_displacement, ddof = 1))
                
            sigma_X_displacement_normalized_experiment += [_/(np.std(sigma_X_displacement_cell_averaged, ddof = 1)) for _ in sigma_X_displacement_cell]
            sigma_Y_displacement_normalized_experiment += [_/(np.std(sigma_Y_displacement_cell_averaged, ddof = 1)) for _ in sigma_Y_displacement_cell]
    
    density_colored_scatter_plot(sigma_X_displacement_experiment, sigma_Y_displacement_experiment, 0, 100, [0, 0.4], 'unnormalized')
    density_colored_scatter_plot(sigma_X_displacement_normalized_experiment, sigma_Y_displacement_normalized_experiment, 0, 1.5, [0, 2.5], 'normalized')
    
    #skewness([sigma_X_displacement_experiment, sigma_Y_displacement_experiment])
    
    # data = np.vstack([sigma_X_displacement_experiment, sigma_Y_displacement_experiment])
    # PDF_KDE_3D(data, 0, 0.2)
    
    # data = np.vstack([sigma_X_displacement_normalized_experiment, sigma_Y_displacement_normalized_experiment])
    # PDF_KDE_3D(data, 0, 2.5, name_PDF = 'normalized')
    
    #skewness([sigma_X_displacement_normalized_experiment, sigma_Y_displacement_normalized_experiment], normalization_status= ' normalized')
    

        
    


    

                        
        



        
