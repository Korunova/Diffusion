# -*- coding: utf-8 -*-
"""
Fractional Brownian motion modeling of 2D tracks to analyze motion heterogeneity, based on a density-colored scatter plot of 
track-wise standard deviations of X and Y displacements, normalized (or unnormalized) to the standard deviation 
of X and Y displacements across all tracks.
"""

#include size of particles

import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
import random
import csv
from scipy.stats import gaussian_kde
from scipy.integrate import simps
from collections import defaultdict
from scipy.optimize import curve_fit

save = False
save_name = '0.2_0.5HbetweenCells_CellPopulation (3)'

#parameters for model
particle_number = 500
tau = 0.005 #time between frames in sec
length = 0.050  # length of the whole track in sec
groups = 30 #number of groups in one simulation

step = 1 #minimum lag time for ensemble- and time- averaged mean squared displacement (MSD) (step = 1 ~ 5 ms)
step_disp = 2 #time displacement to collect coordinate displacement (step = 2 ~ 10 ms)

#Calculation of theoretical diffusion coefficient in water to calculate scale factor
T = 273 + 37
nw = 0.6913*10**(-3)
Kb = 1.38*10**(-23) #J/K (Pa*m3/K)
r = 20*10**(-9) #m
L = 1*10**(-6) #m size of field
nc = nw*50 # our viscosity
D = Kb*T/(6*np.pi*nc*r) #m2/sec

scale_factor = np.sqrt(2 * D) #scale factor to scale track 


#parameters for saved figures
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


#functions to measure diffusion coefficient and parameter a from power-law describing relationship between MSD and time lag
def anomalius_diffusion(x, a, Da):
    return np.log(4*Da)+a*x

# Brownian motion case: a = 1 (fixed)
def brownian_diffusion(x, D):
    return np.log(4*D) + x

def anomalius_diffusion_parameters(x, y, ysigma = None, points = 10):
    # Fit the linear function to the data with uncertainties
    
    params, covariance = curve_fit(anomalius_diffusion, x[0:points], y[0:points], sigma = ysigma[0:points])
    
    #extract, param
    fitted_a, fitted_Da = params
    
    # Extract the diagonal elements of the covariance matrix as the squared errors (standard errors of the slope and intercept)
    a_error,  Da_error = np.sqrt(np.diag(covariance))
        
    #R2 determination
    predicted_msd = fitted_a * x[0:points] + np.log(4*fitted_Da)
    r_squared = 1 - np.sum((y[0:points] - predicted_msd) ** 2) / np.sum((y[0:points] - np.mean(y[0:points])) ** 2)
    
    return float(r_squared), predicted_msd, float(fitted_Da), float(Da_error), float(fitted_a), float(a_error)

def brownian_diffusion_parameters(x, y, ysigma = None, points = 10):
    # Fit the linear function to the data with uncertainties
    
    params, covariance = curve_fit(brownian_diffusion, x[0:points], y[0:points], sigma = ysigma[0:points])
    
    #extract, param
    fitted_D = params
    
    # Extract the diagonal elements of the covariance matrix as the squared errors (standard errors of the slope and intercept)
    D_error = np.sqrt(np.diag(covariance))
        
    #R2 determination
    predicted_msd = x[0:points] + np.log(4*fitted_D)
    r_squared = 1 - np.sum((y[0:points] - predicted_msd) ** 2) / np.sum((y[0:points] - np.mean(y[0:points])) ** 2)
    
    return float(r_squared), predicted_msd, float(fitted_D), float(D_error)


def diffusion_fit(x, y, ysigma=None, points=10, Figure=False, mode='anomalius'):
    
    # Convert to numpy arrays
    x, y, ysigma = map(np.array, (x, y, ysigma))
    
    # Log-transform
    x_log = np.log(x)
    y_log = np.log(y)
    ysigma_log = ysigma / (y * np.log(10))
    
    # Choose the appropriate fit function
    if mode == 'anomalius':
        fit_function = anomalius_diffusion_parameters
        plot_title = 'Anomalous Diffusion (aeMSD)'
    else:
        fit_function = brownian_diffusion_parameters
        plot_title = 'Brownian Diffusion (aeMSD)'
    
    # Initial fit for the first 4 points
    result = fit_function(x_log, y_log, ysigma=ysigma_log, points=4)
    
    # Increment points until the R-squared drops below 0.9 or reach 10 points
    for p in range(4, points + 1):
        result = fit_function(x_log, y_log, ysigma=ysigma_log, points=p)
        if result[0] < 0.90 or p == 10:
            result = fit_function(x_log, y_log, ysigma=ysigma_log, points=p-1)
            break
    
    # If the R-squared is still acceptable, optionally plot and return the results
    if result[0] > 0.9:
        if Figure:
            plt.figure(figsize=(8, 8))
            plt.errorbar(x_log, y_log, yerr=ysigma_log)
            plt.errorbar(x_log[:p-1], result[1])
            plt.title(plot_title)
            
            Da = result[2] * 10**12
            Da_sigma = result[3] * 10**12
            y_position = min(y_log) + (min(y_log) - min(y_log)) * 0.25  # Adjust 0.25 for "slightly lower"
            
            #plt.xlabel('time, s')
            if mode == 'anomalius':
                plt.text(-4.8, y_position-0.2, f'$R^2$ = {result[0]:.2f} \nDα = {Da:.3f} ± {Da_sigma:.3f} $\mu$m$^2$/$sec^α$ \nα = {result[4]:.2f} ± {result[5]:.2f}', fontsize=30, color='black', ha='left', va='center')
            else:
                plt.text(-4.8, y_position-0.2, f'$R^2$ = {result[0]:.2f} \nDα = {Da:.3f} ± {Da_sigma:.3f} $\mu$m$^2$/$sec^2$', fontsize=30, color='black', ha='left', va='center')
            #plt.ylabel('log(MSD), $m^2$')
            plt.grid(True)
            
            if save:
                plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\MSDate_{save_name}.png', dpi=300)
                plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\MSDate_{save_name}.pdf', dpi=300) 
                
            plt.show()
        
        return result[2], result[3], result[4] if mode == 'anomalius' else None, result[5] if mode == 'anomalius' else None
    
    return None    

#density colored scatter plot 
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
        plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\Figure3A_2D_{mode}_{save_name}.png', dpi=300)
        plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\Figure3A_2D_{mode}_{save_name}.pdf', dpi=300)
    plt.show()

# Function for 3D Kernel Density Estimation Plot
def PDF_KDE_3D(data, kde_min_filter, kde_max_filter, Value='Value', name_PDF = ''):
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
        ax.set_ylabel('$\sigma$($\Delta$Y) / $\sigma$($\Delta$Y)$_{cell}$', labelpad=40)
    else:
        ax.set_xlabel('$\sigma$($\Delta$X), $\mu$m', labelpad=40)
        ax.set_ylabel('$\sigma$($\Delta$Y), $\mu$m', labelpad=40)
        
    ax.set_zlabel('Probability Density', labelpad=40)
    ax.set_zlim(0, 3000)
    
    if save:
        plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\Figure3A_3D_{name_PDF}.png', dpi=300)
        plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\Figure3A_3D_{name_PDF}.pdf', dpi=300)

    plt.show()
    
    # Calculate the integral of the KDE using Simpson's Rule
    integral = simps(simps(Z, x_vals), y_vals)  # Nested Simpson's rule for 2D
    


#lists to collect lists for heterogeinity analyses
sigmaX_normalized_cells = []
sigmaY_normalized_cells = []

sigmaX_cells = []
sigmaY_cells = []

#Data to collect
H_list = [] #collect input Hurst parameter
aeMSD = defaultdict(list) #list for ansemble and time averaged MSD values 
Da_list = [] #collect output diffusion coefficient estimated from MSD ~ t^a
a_list = [] #collect output a coefficient estimated from MSD ~ t^a


for group in range(0, groups):
    sigmaX_averaged = []
    sigmaY_averaged = []
    
    sigmaX = []
    sigmaY = []
    
    #H = 0.5
    H = random.uniform(0.2,0.5)
    
    for N in range(0, particle_number):
        
        #SIMULATION PART: fractional Brownian motion 
        
        # Define parameters for fractional Brownian motion
        #H = random.uniform(0.2,0.5)
        n = int(length/tau)  # Number of steps
        H_list.append(H)
        
        # Generate fractional Brownian motion for x and y directions
        f_x = FBM(n=n, hurst=H, length=length, method='hosking') #hosking, cholesky and daviesharte
        f_y = FBM(n=n, hurst=H, length=length, method='hosking') 
        
        #scaled x and y
        fbm_path_x = f_x.fbm() * np.sqrt(2 * D) 
        fbm_path_y = f_y.fbm() * np.sqrt(2 * D) 
        
        # Create time
        time = np.linspace(0, length, int(length/tau))
        
        #Collection of standard deviation of X or Y coordinate displacement across one track for motion type heterogeinity analyses 
        X_displacement = []
        Y_displacement = []
        for i in range(0, len(time)-1, step_disp):
            x_displacement = fbm_path_x[i+1] - fbm_path_x[i]
            y_displacement = fbm_path_y[i+1] - fbm_path_y[i]
            X_displacement.append(x_displacement)
            Y_displacement.append(y_displacement)
            
            sigmaX_averaged.append(x_displacement)
            sigmaY_averaged.append(y_displacement)
            
        sigmaX.append(np.std(X_displacement, ddof = 1))
        sigmaY.append(np.std(Y_displacement, ddof = 1))
        
        if save:
            with open(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\trajectories_{save_name}.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['group'] + [group])
                writer.writerow(['time (s)'] + time.tolist())
                writer.writerow(['x (m)'] + fbm_path_x.tolist())
                writer.writerow(['y (m)'] + fbm_path_y.tolist())
              
        #estimation of diffusion parameters from time-averaged MSD and collection of data 
        teMSD_x = []
        teMSD_y = []
        teMSD_ysigma = []
        for t in range(step, len(time)-1, step):
            # tau = 5, tau = 10, 15
            msd_list = []
            time_ds = time[t+1]-time[1]
            
            for delta in range(0, len(time)-t, step):
                msd = (fbm_path_x[delta+t]-fbm_path_x[delta])**2 + (fbm_path_y[delta+t]-fbm_path_y[delta])**2
                msd_list.append(msd)
                tau2 = time[delta+t] - time[delta]
                aeMSD[round(tau2,3)].append(msd)
            
            teMSD_x.append(t)
            teMSD_y.append(np.mean(msd_list))
            teMSD_ysigma.append(np.std(msd_list, ddof = 1))
        
        result_teMSD = diffusion_fit(teMSD_x, teMSD_y, teMSD_ysigma, points = 3)
        if result_teMSD != None:
            Da_list.append(result_teMSD[0]*10**12)
            a_list.append(result_teMSD[2])
    
    #List of single tracks' standard deviations of displacements (sigmaX and sigmaY) collected from one group
    sigmaX = [_*10**6 for _ in sigmaX] # Convertion from m to um
    sigmaY = [_*10**6 for _ in sigmaY]
    
    #Normalization of collected single tracks' standard deviations of displacements to standard deviation of displacemnts across one group
    sigmaX_normalized = [_/(np.std(sigmaX_averaged, ddof = 1)*10**6) for _ in sigmaX]
    sigmaY_normalized = [_/(np.std(sigmaY_averaged, ddof = 1)*10**6) for _ in sigmaY]
    
    #list of normalized single tracks' standard deviations of displacements collected from all groups in one simulation
    sigmaX_normalized_cells += sigmaX_normalized
    sigmaY_normalized_cells += sigmaY_normalized
    
    #list of single tracks' standard deviations of displacements collected from all groups in one simulation
    sigmaX_cells += sigmaX
    sigmaY_cells += sigmaY

#Visualisation of single tracks' standard deviations (normalized and unnormalized) 
plt.figure()
plt.errorbar(sigmaX_normalized_cells, sigmaY_normalized_cells, fmt='o', elinewidth=2, color = 'orange', capsize=4)
plt.xlim(0, 2.5)
plt.ylim(0, 2.5)
plt.xlabel('$\sigma$($\Delta$X) / $\sigma$($\Delta$X)$_{cell}$')
plt.ylabel('$\sigma$($\Delta$Y) / $\sigma$($\Delta$Y)$_{cell}$') 

#plt.yticks([tick for tick in plt.gca().get_yticks() if tick != 0])
plt.legend()
if save:
    plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\Figure3A_normalized_{save_name}.png', dpi=300)
    plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\Figure3A_normalized_{save_name}.pdf', dpi=300)
plt.show()

density_colored_scatter_plot(sigmaX_normalized_cells, sigmaY_normalized_cells, 0, 1.5, [0, 2.5], 'normalized')

# data = np.vstack([sigmaX_normalized_cells, sigmaY_normalized_cells])
# PDF_KDE_3D(data, 0, 2.5, name_PDF = 'normalized')

plt.figure()
plt.errorbar(sigmaX_cells, sigmaY_cells, fmt='o', elinewidth=2, color = 'orange', capsize=4)
plt.xlim(0, 0.4)
plt.ylim(0, 0.4)
plt.xlabel('$\sigma$($\Delta$X), $\mu$m')
plt.ylabel('$\sigma$($\Delta$Y), $\mu$m') 
#plt.yticks([tick for tick in plt.gca().get_yticks() if tick != 0])
plt.legend()
if save:
    plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\Figure3A_2D_{save_name}.png', dpi=300)
    plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\Figure3A_2D_{save_name}.pdf', dpi=300)
plt.show()

density_colored_scatter_plot(sigmaX_cells, sigmaY_cells, 0, 100, [0, 0.4], 'unnormalized')

# data = np.vstack([sigmaX_cells, sigmaY_cells])
# PDF_KDE_3D(data, 0, 0.5)

#Estimation of diffusion parameters from ansemble- and time-averaged MSD
tau_x = []
msd_y = []
sigma_y = []
for key, values in aeMSD.items():
    tau_x.append(key)
    msd_y.append(np.mean(values))
    sigma_y.append(np.std(values, ddof = 1))
    #print(key)
    
result = diffusion_fit(tau_x, msd_y, sigma_y, points=10, Figure = True)
D2 = result[0]
a = result[2]
print(D*10**12, D2*10**12, a)
plt.figure(figsize=(8, 8))
plt.errorbar(tau_x, msd_y, yerr=sigma_y)

#Real H_list
plt.figure()
plt.hist(H_list, bins = 50)
plt.xlim(0, 1)
plt.ylabel('counts')
plt.title('Hurst parameter distribution')
if save:
    plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\H_list_{save_name}.png', dpi=300)
    plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\H_list_{save_name}.pdf', dpi=300)
plt.show()

#Data from Brownian diffusion
plt.figure()
plt.hist(Da_list, bins = 25)
plt.title('$D_{α}$ distribution')
plt.ylabel('counts')
plt.xlabel('$um^2$/$sec^α$')
if save:
    plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\Da_list_{save_name}.png', dpi=300)
    plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\Da_list_{save_name}.pdf', dpi=300)
plt.show()

plt.figure()
plt.hist(a_list, bins = 25)
plt.xlim(0, 2)
plt.ylabel('counts')
plt.title('α distribution')
if save:
    plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\a_list_{save_name}.png', dpi=300)
    plt.savefig(fr'D:\article_GEM_U2OS_expression\Main Revision\pictures\Figure 2 fmb model\{save_name}\a_list_{save_name}.pdf', dpi=300)
plt.show()
