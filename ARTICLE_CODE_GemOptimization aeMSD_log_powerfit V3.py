# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:43:32 2024

@author: Huawei
"""

import csv 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import sympy as sp
import os
import pandas as pd
from matplotlib import cm

delta_x = 0.005 #sec _interval of linear interpolation
taus = [0.01] #time displacement for MSD calculation, interpolation
save = False
time_scale = 1
#scale = (1/0.0621481)
scale = (1/0.0586000)
min_trajectory_length = 10
remove_static_error = True
error_file = 'D:082124_082524 folder for not using files in coding/gem FIXED 35Oc/MSD_lagtime5.0ms.csv'
step = 2

ExperDirectory = 'D:082124_082524/'
experiments = [f for f in os.listdir(ExperDirectory) if os.path.isdir(os.path.join(ExperDirectory, f))]
#experiments = ['GEM 3d 35oC']

experiment_condition = 'r6cf0p1_l1d5'

exp_cond = experiments

T = 308 #K #variable!
nw = 0.7195*10**(-3) #Pa s #variable!

#T = 273 + 37
#nw = 0.6913*10**(-3)

Kb = 1.38*10**(-23) #J/K (Pa*m3/K)
r = 20*10**(-9) #m
Dw = Kb*T/(6*np.pi*nw*r)*10**12 #um2/sec


# Set Times New Roman and font sizes globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [16, 10]  # Adjust based on your needs
plt.rcParams['font.size'] = 35  # Set global font size
plt.rcParams['axes.titlesize'] = 35  # Title font size
plt.rcParams['axes.labelsize'] = 30  # Axis labels font size
plt.rcParams['xtick.labelsize'] = 30  # X-axis tick labels font size
plt.rcParams['ytick.labelsize'] = 30  # Y-axis tick labels font size
plt.rcParams['legend.fontsize'] = 30  # Legend font size

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


def plot(x, y, xname = None, yname = None, std = None, plot_number = 'None'):
    plt.figure(num = int(plot_number), figsize=(12,8)) if plot_number != 'None' else plt.figure(figsize = (12,8))
    plt.plot(x, y, marker='.', linestyle = '-', linewidth = 2, markersize = 2)
    
    if std is not None:
        plt.errorbar(x, y, yerr=std, ecolor='green', label=f'Standard Deviation {yname}', alpha = 0.5)
    
    plt.ylabel(yname)
    plt.xlabel(xname)
    plt.grid(True, zorder=1)

def FitLinearFunction(name, x, y, label = None, n = None, points=10, R2=0.9, sigma = None, check_plot = False):
    # Define the linear function
    def linear_function(x, a, Da):
        return a*x + np.log(4*Da)
    
    # Fit the linear function to the data with uncertainties
    params, covariance = curve_fit(linear_function, x[0:points], y[0:points], sigma = sigma[0:points])
    
    # Extract the fitted parameters
    fitted_a, fitted_Da = params
    
    # Extract the diagonal elements of the covariance matrix as the squared errors (standard errors of the slope and intercept)
    a_error, Da_error = np.sqrt(np.diag(covariance))
        
    #R2 determination
    predicted_msd = linear_function(x[0:points], fitted_a, fitted_Da)
    r_squared = 1 - np.sum((y[0:points] - predicted_msd) ** 2) / np.sum((y[0:points] - np.mean(y[0:points])) ** 2)
    if check_plot == True:
        colour = f'C{n}'
        #colors = cm.coolwarm(np.linspace(0, 1, len(experiments)))
        plt.figure(figsize=(18, 14), num= 13)
        plt.errorbar(x, y, yerr = sigma, fmt = 'o', color = colour, alpha = 0.2)
        plt.plot(x[0:points], predicted_msd, marker='.', linestyle = '-', linewidth = 2, markersize = 2, color = colour, label = f'{label}')
        plt.ylabel('log(<MSD>), um2')
        plt.xlabel('log(τ), sec')
        plt.title(f'tracks over 50 ms with power-law fit with removed static error')
        plt.xticks()
        plt.yticks()
        plt.legend()
        
        plt.figure(figsize=(18, 14), num= 14)
        plt.errorbar(x, y, yerr = sigma, fmt = 'o', alpha = 0.2, color = colour, label = f'{label}')
        plt.ylabel('log(<MSD>), um2')
        plt.xlabel('log(τ), sec')
        plt.title(f'tracks over 50 ms')
        plt.xticks()
        plt.yticks()
        plt.legend()
        
    return name, float(fitted_a), float(a_error), float(fitted_Da), float(Da_error), r_squared

color_number = 0
results = []
for tau in taus: 

    condition = 0    
    
    for experiment in experiments: 
        sigma_X_displacement = [] 
        sigma_Y_displacement = []
        
        print(f'{experiment}')
        
        parent_dir = f'{ExperDirectory}{experiment}/'
        files = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
        
    
        lagtime_data = {}
        for file in files: 
            
            if not os.path.exists(f'{parent_dir}{file}/{experiment_condition}/converted_data_length10.csv'):
                continue
            
            #Convertation of converted_data to the list ['trajectory №', [time_o], [x], [y], [time_frame]] for plotting
            with open(f'{parent_dir}{file}/{experiment_condition}/converted_data_length10.csv') as data:
                data_file = csv.reader(data, delimiter=',')
                
                trajectories = [] # list with trajectories for plotting
                
                check = False
                for row in data_file:
                    if row[0] == 'Trajectory':
                        if check == True:
                            one_tr = (name, time, x, y, time_frame)
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
                trajectories.append(one_tr)
    
            trajectories_filtered = []
            for trajectory in trajectories:
                name = trajectory[0]
                t = trajectory[1]
                x = trajectory[2]
                y = trajectory[3]
                t_frame = trajectory[4]
                
                #Trajectory filter 
                if len(x)> min_trajectory_length:
                    one_tr = (name, t, x, y, t_frame)
                    trajectories_filtered.append(one_tr)
                                
            trajectories_interpolated = []
            for trajectory in trajectories_filtered:
                
                name = trajectory[0]
                t = trajectory[1]
                x = trajectory[2]
                y = trajectory[3]
                time_frame = trajectory[4]
                
                
                #Interpolation of trajectories data and drift correction
                result_x = interpolation(delta_x, t, x, 'sec_function', 'x, um', 6) 
                result_y = interpolation(delta_x, t, y, 'sec_function', 'y, um', 7)
                
                one_tr = (name, result_x[0], result_x[1], result_y[1])
                trajectories_interpolated.append(one_tr)
                
            #heterogenity 
            for trajectory in trajectories_interpolated:
                X_displacement = []
                Y_displacement = []
                
                name = trajectory[0]
                t = trajectory[1]
                x = trajectory[2]
                y = trajectory[3]
                
                for i in range(0, len(t)-2, step):
                    time_interval = t[i+1] - t[i] 
                    x_displacement = x[i+1] - x[i]
                    y_displacement = y[i+1] - y[i]
                    X_displacement.append(x_displacement)
                    Y_displacement.append(y_displacement)
                    
                sigma_Y_displacement.append(np.std(Y_displacement))
                sigma_X_displacement.append(np.std(X_displacement))
                
            tau_list = []
            msd_list = []
            msd_std_list = []
            msd_se_list = []
            
            
                
            for trajectory in trajectories_interpolated:
                
                name = trajectory[0]
                t = trajectory[1]
                x = trajectory[2]
                y = trajectory[3]
                
                min_step = int(tau/delta_x)
                
                for delta_t in range(min_step, len(t), min_step):
                   
                    for i in range(0, len(t) - delta_t, min_step):
                        time_interval = round(t[i+delta_t]-t[i],3) 
                        #t[10] - t[0], t[20] - t[10] = 10
                        #t[20] - t[0], t[30] - t[10] = 20 --- delta_t = 2
                        squared_displacement = (x[i + delta_t] - x[i])**2 + (y[i + delta_t] - y[i])**2
                        if time_interval not in lagtime_data.keys():
                            lagtime_data[time_interval] = [squared_displacement]
                        else:
                            lagtime_data[time_interval].append(squared_displacement)
                            
                            
                            
        sigma_X_displacement = np.array(sigma_X_displacement)
        sigma_Y_displacement= np.array(sigma_Y_displacement)
        combined_color = (sigma_X_displacement  + sigma_Y_displacement)/2
        
        # Normalize the color values
        norm = plt.Normalize(vmin=min(combined_color), vmax=0.1)  # Normalize to the range of combined_color
        cmap = cm.get_cmap('plasma')  # Use the 'coolwarm' colormap    

        plt.figure()
        sc = plt.scatter(sigma_X_displacement, sigma_Y_displacement, c=combined_color, cmap=cmap, norm=norm)
        plt.xlabel('standard deviation of X')
        plt.ylabel('standard deviation of X') 
        plt.title(f'{exp_cond[condition]}')
        plt.xlim(0, 0.14)
        plt.ylim(0, 0.14)
        cbar = plt.colorbar(sc)
        cbar.set_label('Standard Deviation of Displacement')
        if True: 
            plt.savefig(f'{ExperDirectory}Figure 9 ARTICLEV3_sigma_XY_displacement_lagtima{step*5}ms_{T}T_{experiment}.pdf', dpi=300)
        
        # If remove_static_error is True, load the error values from the CSV file
        if remove_static_error:
            # Read MSD errors from CSV and store in a dictionary {time_interval: error_value}
            msd_errors = {}
            with open(error_file, mode='r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                for row in reader:
                    time_interval = float(row[0])
                    msd_static = float(row[1])
                    sigma_static = float(row[2])
                    msd_errors[time_interval] = [msd_static, sigma_static]
                    
        for key, values in lagtime_data.items():
            mean_msd = np.mean(values)
            sigma_msd = np.std(values, ddof = 1)

            # If static error removal is enabled, subtract the static error from the mean MSD
            if remove_static_error and key in msd_errors:
                mean_msd -= msd_errors[key][0]
                sigma_msd = np.sqrt(sigma_msd**2 + sigma_static**2)
                                       
            tau_list.append(key)
            msd_list.append(mean_msd )
            msd_std_list.append(np.std(values, ddof = 1))
                    
        #plot(tau_list, msd_list, xname = 'tau', yname= 'MSD', std=msd_std_list, plot_number=11) #change color
        #plot(tau_list, msd_list, xname = 'tau', yname= 'MSD', std=msd_se_list, plot_number=12) #change color
                    
        time_data = np.array(tau_list)
        msd_data = np.array(msd_list)
        msd_std_data = np.array(msd_std_list)
                    
        #Mode of Motion Calulation
        time_data_log10 = np.log10(time_data)
        msd_data_log10 = np.log10(msd_data)
        msd_std_data_log10 = msd_std_data / (msd_data * np.log(10))  # Corrected error calculation log10x' = 1/(<x>*log10)
        
        limit = round(len(msd_data_log10)/10)
            
        for p in range(5,11):
            result = FitLinearFunction(name, time_data_log10[0:limit], msd_data_log10[0:limit], exp_cond[condition], n = color_number, points = p, sigma = msd_std_data_log10[0:limit], check_plot = False)
            if result[5] < 0.98 or p == 10:
                result = FitLinearFunction(name, time_data_log10[0:limit], msd_data_log10[0:limit], label = experiment, n = color_number, points = p-1, sigma = msd_std_data_log10[0:limit], check_plot = True)
                break

        color_number += 1 
        condition +=1 
                
    
        if result != None:
            print('Result is not None')
            Da_value = result[3]
            a_value = result[1]
            Da_sigma = result[4] #Calculate the error in <D>: propagation of uncertainty fo functions of single veriables
            a_sigma = result[2]
            
            Da, a = sp.symbols('Da a')
            Deff = Da * 2**(2 * (a - 1)) / (Dw**a)
            partial_Da = sp.diff(Deff, Da) 
            partial_a = sp.diff(Deff, a)
            
            partial_Da_value = float(partial_Da.subs({Da: Da_value, a: a_value}))
            partial_a_value = float(partial_a.subs({Da: Da_value, a: a_value}))
            
            Deff_sigma = np.sqrt((partial_Da_value * Da_sigma)**2 + (partial_a_value * a_sigma)**2)
            Deff_value = float(Deff.subs({Da: Da_value, a: a_value}))
            print('window = ', tau, 'no static error')
            print(result[5])
            print(f'Da = {Da_value:.3f} ± {Da_sigma:.3f}, a = {a_value:.2f} ± {a_sigma:.2f} \n Deff = {Deff_value:.3f} ± {Deff_sigma:.3f}')
            print()
            
            results.append({
                'Experiment': f'window = {tau*1000} ms',
                'Da_value': Da_value,
                'Da_sigma': Da_sigma,
                'a_value': a_value,
                'a_sigma': a_sigma,
                'Deff_value': Deff_value,
                'Deff_sigma': Deff_sigma
            })
        
        
if save == True: 
    plt.figure(num=13)
    plt.savefig(f'D:082124_082524/Figure 1B V3.pdf', dpi=300)
    #plt.show()
    
    plt.figure(num=14)
    plt.savefig(f'D:082124_082524/Figure 1D V3.pdf', dpi=300)
    #plt.show()

    # Convert the results list into a pandas DataFrame
    df = pd.DataFrame(results)
    
    # Save the DataFrame to a CSV file
    df.to_csv(f'D:082124_082524/table1_length10 V3.csv', index=False)
    
        
        
                
    
    
    
