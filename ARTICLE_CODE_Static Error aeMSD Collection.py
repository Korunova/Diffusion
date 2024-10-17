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
taus = [0.005, 0.01, 0.015, 0.02] #time displacement for MSD calculation, interpolation
save = False
time_scale = 1
#scale = (1/0.0621481)
scale = (1/0.0586000)
min_trajectory_length = 10


ExperDirectory = 'D:082124_082524 folder for not using files in coding/'
experiments = [f for f in os.listdir(ExperDirectory) if os.path.isdir(os.path.join(ExperDirectory, f))]
experiments = ['gem FIXED 35Oc']

experiment_condition = 'r6cf0p1_l1d5'

# Set Times New Roman and font sizes globally
plt.rcParams['font.family'] = 'Times New Roman'
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


color_number = 0
for tau in taus: 

    condition = 0  
    print(tau)
    for experiment in experiments: 
        
        parent_dir = f'{ExperDirectory}{experiment}/'
        files = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
        
    
        lagtime_data = {}
        for file in files: 
            print(file)
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
    
    
            tau_list = []
            msd_list = []
            msd_std_list = []
            msd_se_list = []
            
            #lagtime_data = {}
                
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
                                    
        for key, values in lagtime_data.items():
                                       
            tau_list.append(key)
            msd_list.append(np.mean(values))
            #msd_list.append(np.median(values))
            msd_std_list.append(np.std(values, ddof = 1))
            msd_se_list.append(np.std(values, ddof = 1)/np.sqrt(len(values)))
                    
        #plot(tau_list, msd_list, xname = 'tau', yname= 'MSD', std=msd_std_list, plot_number=11) #change color
        #plot(tau_list, msd_list, xname = 'tau', yname= 'MSD', std=msd_se_list, plot_number=12) #change color
                    
        time_data = np.array(tau_list)
        msd_data = np.array(msd_list)
        msd_std_data = np.array(msd_std_list)
        
        df = pd.DataFrame({
            'Time': time_data,
            'MSD': msd_data,
            'MSD_STD': msd_std_data
        })

        # Save to CSV
        df.to_csv(f'{ExperDirectory}{experiment}/MSD_lagtime{tau*1000}ms.csv', index=False)

        
        
    
        
        
                
    
    
    
