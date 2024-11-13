# -*- coding: utf-8 -*-
"""
MSD calculationa from fBM modeling
"""

#include size of particles

import numpy as np
import matplotlib.pyplot as plt
from fbm import FBM
import random
from collections import defaultdict
from scipy.optimize import curve_fit
import sympy as sp


particle_number = 1000
tau = 0.005
length = 0.400  # Length of time (optional)
step = 2 #for 5 ms lag_time


T = 273 + 37
nw = 0.6913*10**(-3)
Kb = 1.38*10**(-23) #J/K (Pa*m3/K)
r = 20*10**(-9) #m
nc = nw*np.exp(1.5)
D_theoretical = Kb*T/(6*np.pi*nc*r)*10**12 #um2/sec
Dw =  Kb*T/(6*np.pi*r)*10**12 #um2/sec
save = False
file_path = 'C:/Users/korunova/Desktop/Random walks in Biology/Hsuper'


# Set Times New Roman and font sizes globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [16, 10]  # Adjust based on your needs
plt.rcParams['font.size'] = 35  # Set global font size
plt.rcParams['axes.titlesize'] = 35  # Title font size
plt.rcParams['axes.labelsize'] = 30  # Axis labels font size
plt.rcParams['xtick.labelsize'] = 30  # X-axis tick labels font size
plt.rcParams['ytick.labelsize'] = 30  # Y-axis tick labels font size
plt.rcParams['legend.fontsize'] = 30  # Legend font size
 

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
            plt.xlabel(f'time, ms {result[0]:.2f}\nDa = {result[2]:.3f} ± {result[3]:.3f}')
            if mode == 'anomalius':
                plt.xlabel(f'time, ms {result[0]:.2f}\nDa = {result[2]:.3f} ± {result[3]:.3f}, a = {result[4]:.3f} ± {result[5]:.3f}')
            plt.ylabel('MSD, um²')
            plt.grid(True)
            if save == True:
                plt.savefig(f'{file_path}/aeMSD.pdf', dpi=300)
            plt.show()
        
        return result[2], result[3], result[4] if mode == 'anomalius' else None, result[5] if mode == 'anomalius' else None
    
    return None    

def anomalius_diffusion_var(x, a, Da):
    return np.log(2*Da)+a*x

def anomalius_diffusion_variance(x, y, points = 10):
    # Fit the linear function to the data with uncertainties
    
    params, covariance = curve_fit(anomalius_diffusion_var, x[0:points], y[0:points])
    
    #extract, param
    fitted_a, fitted_Da = params
    
    # Extract the diagonal elements of the covariance matrix as the squared errors (standard errors of the slope and intercept)
    a_error,  Da_error = np.sqrt(np.diag(covariance))
        
    #R2 determination
    predicted_msd = fitted_a * x[0:points] + np.log(2*fitted_Da)
    r_squared = 1 - np.sum((y[0:points] - predicted_msd) ** 2) / np.sum((y[0:points] - np.mean(y[0:points])) ** 2)
    
    return float(r_squared), predicted_msd, float(fitted_Da), float(Da_error), float(fitted_a), float(a_error)

aeMSD = defaultdict(list)

#anomaius power law
Da_list = []
a_list = []
Deff_list = []
Deff_sigma_list = []

#Brownian parameters
D_list = [] 
D_sigma_list = []

#parameters calculated from variance
Da_var_list = []
a_var_list = []

#Hurst parameter list
H_list = []

count_Brownian = 0
count_anomalius = 0

#standard deviationa of coordinates calculated from raw data
sigmaX = []
sigmaY = []
#from data passed power law
sigmaX_powerlaw = []
sigmaY_powerlaw = []
#from data passede Brownian law
sigmaX_Brownian = []
sigmaY_Brownian = []

H_test = [0.2, 0.4, 0.5]
#H = random.uniform(0.4,0.5)   # Hurst parameter for fractional Brownian motion
#H = random.uniform(0.5,1)   # Hurst parameter for fractional Brownian motion

n=0

for H in H_test:
    for N in range(0, particle_number):
        # Define parameters for fractional Brownian motion
        n = int(length/tau)  # Number of steps
        H_list.append(H)
        
        
        # Generate fractional Brownian motion for x and y directions
        f_x = FBM(n=n, hurst=H, length=length, method='hosking') #hosking, cholesky and daviesharte
        f_y = FBM(n=n, hurst=H, length=length, method='hosking') 
        
        #scaled x and y
        fbm_path_x = f_x.fbm() 
        fbm_path_y = f_y.fbm() 
        
        # Create time
        time = np.linspace(0, length, int(length/tau))
        
        # Plot the 2D fBm path
        #plt.figure(figsize=(8, 8))
        #plt.plot(fbm_path_x, fbm_path_y, marker='o')
        #plt.title(f'2D Fractional Brownian Motion (H = {H})')
        #plt.xlabel('X')
        #plt.ylabel('Y')
        #plt.grid(True)
        #plt.show()
        
        #displacement 
        #X and Y analysis of heterogenity of raw data
        X_displacement = []
        Y_displacement = []
        for i in range(0, len(fbm_path_y)-2, step):
            x_displacement = fbm_path_x[i+1] - fbm_path_x[i]
            y_displacement = fbm_path_y[i+1] - fbm_path_y[i]
            X_displacement.append(x_displacement)
            Y_displacement.append(y_displacement)
        sigmaX.append(np.std(X_displacement, ddof = 1))
        sigmaY.append(np.std(Y_displacement, ddof = 1))
        
    
        #displacement 
        #MSD from displacement variance
        # sigmaX_overtime = []
        # sigmaY_overtime = []
        # t_var = []
        # for step in range(1,10):
        #     X_displacement = []
        #     Y_displacement = []
        #     for i in range(0, len(fbm_path_y)-2-step, step):
        #         x_displacement = fbm_path_x[i+step] - fbm_path_x[i]
        #         y_displacement = fbm_path_y[i+step] - fbm_path_y[i]
        #         t_displacement = time[i+step] - time[i]
        #         X_displacement.append(x_displacement)
        #         Y_displacement.append(y_displacement)
        #     sigmaX_overtime.append(np.var(X_displacement, ddof = 1))
        #     sigmaY_overtime.append(np.var(Y_displacement, ddof = 1))
        #     t_var.append(t_displacement)
        
         
        # sigmaX_overtime = np.array(np.log(sigmaX_overtime))
        # sigmaY_overtime = np.array(np.log(sigmaY_overtime))
        # t_var = np.array(np.log(t_var))
        # result_varMSD = anomalius_diffusion_variance(t_var, sigmaY_overtime, points = 4)
        # if result_varMSD[0]>0.9:
        #     Da_var_list.append(result_varMSD[2])
        #     a_var_list.append(result_varMSD[4])
        
            
        #calculate aeMSD (ansemble-averaged time-averaged MSD) and teMSD (time-averaged MSD))
        teMSD_x = []
        teMSD_y = []
        teMSD_ysigma = []
        for t in range(step, len(fbm_path_y), step):
            # tau = 5, tau = 10, 15
            msd_list = []
            for delta in range(0, len(fbm_path_y)-t, step):
                msd = (fbm_path_x[delta+t]-fbm_path_x[delta])**2 + (fbm_path_y[delta+t]-fbm_path_y[delta])**2
                msd_list.append(msd)
                aeMSD[t].append(msd)
                
            teMSD_x.append(t)
            teMSD_y.append(np.mean(msd_list))
            teMSD_ysigma.append(np.std(msd_list, ddof = 1))
        
        #plt.figure(figsize=(8, 8), num = 1)
        #plt.errorbar(teMSD_x, teMSD_y, yerr = teMSD_ysigma)
        #plt.title('aeMSD')
        #plt.xlabel('time, ms')
        #plt.ylabel('MSD, um2')
        #plt.grid(True)
        
        result_teMSD = diffusion_fit(teMSD_x, teMSD_y, teMSD_ysigma, points = 10)
        
        if result_teMSD != None:
            Da_value = result_teMSD[0]
            Da_sigma = result_teMSD[1]
            a_value= result_teMSD[2]
            a_sigma = result_teMSD[3]
            
            
            Da_list.append(result_teMSD[0])
            a_list.append(result_teMSD[2])
            count_anomalius += 1
            
            
            #calculation of Deff and its standard deviation using the propagated error formula 
            Da, a = sp.symbols('Da a')
            Deff = Da * 2**(2 * (a - 1)) / (Dw**a)
            partial_Da = sp.diff(Deff, Da) 
            partial_a = sp.diff(Deff, a)
            
            partial_Da_value = float(partial_Da.subs({Da: Da_value, a: a_value}))
            partial_a_value = float(partial_a.subs({Da: Da_value, a: a_value}))
            
            Deff_sigma = np.sqrt((partial_Da_value * Da_sigma)**2 + (partial_a_value * a_sigma)**2)
            Deff_value = float(Deff.subs({Da: Da_value, a: a_value}))
            
            Deff_list.append(Deff_value)
            Deff_sigma_list.append(Deff_sigma)
            
            
            #X and Y analysis of heterogenity data passed anomalius law
            X_displacement = []
            Y_displacement = []
            for i in range(0, len(fbm_path_y)-2, step):
                x_displacement = fbm_path_x[i+1] - fbm_path_x[i]
                y_displacement = fbm_path_y[i+1] - fbm_path_y[i]
                X_displacement.append(x_displacement)
                Y_displacement.append(y_displacement)
            sigmaX_powerlaw.append(np.std(X_displacement, ddof = 1))
            sigmaY_powerlaw.append(np.std(Y_displacement, ddof = 1))
            
            
        result_teMSD = diffusion_fit(teMSD_x, teMSD_y, teMSD_ysigma, points = 10, mode = 'Brownian')
        if result_teMSD != None:
            D_list.append(result_teMSD[0])
            D_sigma_list.append(result_teMSD[1])
            count_Brownian += 1 
            
            #X and Y analysis of heterogenity data passed Brownian law
            X_displacement = []
            Y_displacement = []
            for i in range(0, len(fbm_path_y)-2, step):
                x_displacement = fbm_path_x[i+1] - fbm_path_x[i]
                y_displacement = fbm_path_y[i+1] - fbm_path_y[i]
                X_displacement.append(x_displacement)
                Y_displacement.append(y_displacement)
            sigmaX_Brownian.append(np.std(X_displacement, ddof = 1))
            sigmaY_Brownian.append(np.std(Y_displacement, ddof = 1))
        
        
    aeMSD_y = []
    aeMSD_ysigma = []
    aeMSD_x = []
    for key, values in aeMSD.items():
        aeMSD_y.append(np.mean(values))
        aeMSD_ysigma.append(np.std(values, ddof = 1))
        aeMSD_x.append(key)
    
    result = diffusion_fit(aeMSD_x, aeMSD_y, aeMSD_ysigma, points=10, Figure = False)
    
    #Data from anomalius power law
    plt.figure(1)
    plt.hist(Da_list, bins = 50, label = H)
    plt.ylabel ('counts')
    plt.xlabel(f'{count_anomalius} trajectories')
    plt.title('distribution of Da calculated from power-law')
    plt.legend()
    #plt.xlim(0, 0.008)
    #plt.xlim(0,5)
    if save == True:
        plt.savefig(f'{file_path}/Da_dist.pdf', dpi=300)
    
    plt.figure(2)
    plt.hist(a_list, bins = 50)
    plt.ylabel ('counts')
    plt.xlim(0, 2)
    plt.title('distribution of a calculated from power-law')
    if save == True:
        plt.savefig(f'{file_path}/a_dist.pdf', dpi=300)
    
    plt.figure(3)
    plt.hist(Deff_list, bins = 50)
    plt.xlabel(f'{count_anomalius} trajectories')
    plt.title('Deff from an. law')
    #plt.xlim(0,5)
    
    
    #Data from variance of coordinates
    # plt.figure()
    # plt.hist(Da_var_list, bins = 50)
    # plt.xlabel(f'{count_anomalius} trajectories')
    # plt.title('Da from an. law and var')
    # #plt.xlim(0,5)
    
    # plt.figure()
    # plt.hist(a_var_list, bins = 50)
    # plt.title('a from an. law and var')
    
    
    #Real H_list
    plt.figure(4)
    plt.hist(H_list, bins = 50)
    plt.ylabel ('counts')
    plt.xlim(0, 1)
    plt.title('distribution of Hurst parameter')
    if save == True:
        plt.savefig(f'{file_path}/H_dist.pdf', dpi=300)
    
    
    #Data from Brownian diffusion
    plt.figure(5)
    plt.hist(D_list, bins = 25)
    plt.xlabel(f'{count_Brownian} trajectories')
    plt.title('distribution of D calculated from Brownian law')
    #plt.xlim(0, 0.008)
    if save == True:
        plt.savefig(f'{file_path}/D_dist.pdf', dpi=300)
    
    #plt.figure()
    #plt.errorbar(D_sigma_list, D_list, fmt='o')
    #plt.xlim(0,0.0018)
    #plt.ylim(0,0.025)
    #plt.xlabel('standard deviation of D')
    #plt.ylabel('mean D')

    #Heterogenity data from a) raw data b) power law c) Brownian
    plt.figure(6)
    plt.errorbar(sigmaX, sigmaY, fmt='o', elinewidth=2, color = f'C{n}', capsize=4, label = H)
    plt.xlabel('standard deviation of X displacement')
    plt.ylabel('standard deviation of Y displacement') 
    plt.legend()
    if save == True:
        plt.savefig(f'{file_path}/StandardDeviation.pdf', dpi=300)
    
    
    # plt.figure(7)
    # plt.errorbar(sigmaX_powerlaw, sigmaY_powerlaw, fmt='o', elinewidth=2, capsize=4, label = H)
    # plt.xlabel('standard deviation of X displacement')
    # plt.ylabel('standard deviation of Y displacement') 
    # plt.legend()
    
    # plt.figure(8)
    # plt.errorbar(sigmaX_Brownian, sigmaY_Brownian, fmt='o', elinewidth=2, capsize=4, label = H)
    # plt.xlabel('standard deviation of X displacement')
    # plt.ylabel('standard deviation of Y displacement') 
    # plt.legend()
     
    n += 1
