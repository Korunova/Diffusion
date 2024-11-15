# -*- coding: utf-8 -*-
"""

Plots of the standard deviation of X and Y displacements at 10 ms from tracks estimated by Mosaic Suite 


"""

import csv 
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
import numpy as np
import os
import random
import sympy as sp

delta_x = 0.005 #sec _interval of linear interpolation (0.0005 paper 1)
min_trajectory_lengths = [10] #trajectory filter
r2 = 0.9 # filter of diffusion plots try 0
scale = (1/0.0586000)
time_scale = 1 #convert into sec 
save = True
experiment_condition = 'r6cf0p1_l1d5'
step = 2 # 5 ms
colorbar = 'plasma'


ExperDirectory = 'D:/082124_082524/'
experiments = [f for f in os.listdir(ExperDirectory) if os.path.isdir(os.path.join(ExperDirectory, f))]
experiments = [ 'GEM 3d 35oC']


# Set Times New Roman and font sizes globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [16, 10]  # Adjust based on your needs
plt.rcParams['font.size'] = 35  # Set global font size
plt.rcParams['axes.titlesize'] = 35  # Title font size
plt.rcParams['axes.labelsize'] = 30  # Axis labels font size
plt.rcParams['xtick.labelsize'] = 30  # X-axis tick labels font size
plt.rcParams['ytick.labelsize'] = 30  # Y-axis tick labels font size
plt.rcParams['legend.fontsize'] = 30  # Legend font size


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



for experiment in experiments: 
    
    
    # Specify the directory you want to list folders from
    parent_dir = f'{ExperDirectory}{experiment}/'
    files = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    
    #Van Gogh distribution sigma (std dv)


    sigma_X_displacement_experiment = [] 
    sigma_Y_displacement_experiment = []

    for min_trajectory_length in min_trajectory_lengths:
        
        
        for file in files:
            print(f'{experiment} {file}')
            
            
            # Check if the directory path exists
            if not os.path.exists(f'{parent_dir}/{file}/{experiment_condition}/'):
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
                            if len(x) >  min_trajectory_length:
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
                if len(x) >  min_trajectory_length:
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
                
                #X and Y analysis of heterogenity
                for i in range(0, len(t)-2, step):
                    time_interval = t[i+1] - t[i] 
                    x_displacement = x[i+1] - x[i]
                    y_displacement = y[i+1] - y[i]
                    X_displacement.append(x_displacement)
                    Y_displacement.append(y_displacement)
                    
                sigma_Y_displacement_experiment.append(np.std(Y_displacement))
                sigma_X_displacement_experiment.append(np.std(X_displacement))
                    
        sigma_X_displacement_experiment = np.array(sigma_X_displacement_experiment)
        sigma_Y_displacement_experiment = np.array(sigma_Y_displacement_experiment)
        combined_color = (sigma_X_displacement_experiment  + sigma_Y_displacement_experiment)/2
        
        # Normalize the color values
        norm = plt.Normalize(vmin=min(combined_color), vmax=max(combined_color))  # Normalize to the range of combined_color
        cmap = cm.get_cmap(colorbar)  # Use the 'coolwarm' colormap
        
        plt.figure()
        #plt.scatter(sigma_X_displacement_experiment, sigma_Y_displacement_experiment, c = combined_color,  cmap = colorbar)
        sc = plt.scatter(sigma_X_displacement_experiment, sigma_Y_displacement_experiment, c=combined_color, cmap=cmap, norm=norm)
        plt.xlabel('standard deviation of X displacement')
        plt.ylabel('standard deviation of Y displacement') 
        #plt.xlim(0,0.17)
        #plt.ylim(0,0.17)
        cbar = plt.colorbar(sc)
        cbar.set_label('Standard Deviation of Displacement')
        if save == True: 
            plt.savefig(f'{parent_dir}sigma_XY_displacement.pdf', dpi=300)
        plt.show()
                    
        for file in files: 
            sigma_X_displacement = [] 
            sigma_Y_displacement = []
    
            print(f'{experiment} {file}')
            
            
            # Check if the directory path exists
            if not os.path.exists(f'{parent_dir}/{file}/{experiment_condition}/'):
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
                            if len(x) >  min_trajectory_length:
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
                if len(x) >  min_trajectory_length:
                    trajectories.append(one_tr)
                
                
            #collection data for Heterogenity check
            fig, ax = plt.subplots()
            for trajectory in trajectories:
                
                X_displacement = []
                Y_displacement = []
                
                name = trajectory[0]
                t = trajectory[1]
                x = trajectory[2]
                y = trajectory[3]
                t_frame = trajectory[4]


                #X and Y analysis of heterogenity
                for i in range(0, len(t)-2, step):
                    time_interval = t[i+1] - t[i] 
                    x_displacement = x[i+1] - x[i]
                    y_displacement = y[i+1] - y[i]
                    X_displacement.append(x_displacement)
                    Y_displacement.append(y_displacement)
                    
                sigma_Y_displacement.append(np.std(Y_displacement))
                sigma_X_displacement.append(np.std(X_displacement))
                
                c = (np.std(Y_displacement) + np.std(X_displacement))/2
                ax.plot(x, y, color=cmap(norm(c)))
                        
            # Добавление цветовой шкалы сбоку
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
            cbar.set_label('standard deviation of displacement')
    
            plt.xlabel('X, um')
            plt.ylabel('Y, um')
            plt.title(f' ({file})')
            if save == True: 
                 plt.savefig(f'{parent_dir}xy_heterogenity_map_{file}.pdf', dpi=300)
            plt.show()

            
            # #Van Gogh distribution for displacements
            # plt.figure()
            # plt.hist(X_displacement, bins=40, edgecolor='black', color='orange', alpha = 0.5)
            # plt.title('Distribution of X in U2OS colony')
            # plt.ylabel('Frequency')
            # plt.xlabel('x displacement, lag time = 5 ms')
            # if save == True: 
            #     plt.savefig(f'{parent_dir}{directory}/x_distribution_{file}.pdf', dpi=300)
            # plt.show()
            
            # plt.figure()
            # plt.hist(Y_displacement, bins=40, edgecolor='black', color='orange', alpha = 0.5)
            # plt.title('Distribution of Y in U2OS colony')
            # plt.ylabel('Frequency')
            # plt.xlabel('y displacement, lag time = 5 ms')
            # if save == True: 
            #     plt.savefig(f'{parent_dir}{directory}/y_distribution_{file}.pdf', dpi=300)
            # plt.show()
                                
            sigma_X_displacement = np.array(sigma_X_displacement)
            sigma_Y_displacement = np.array(sigma_Y_displacement)
            
            combined_color_cell = (sigma_X_displacement + sigma_Y_displacement)/2
            
            plt.figure()
            plt.scatter(sigma_X_displacement, sigma_Y_displacement, c = cmap(norm(combined_color_cell)),  cmap = colorbar)
            plt.xlabel('standard deviation of X displacement')
            plt.ylabel('standard deviation of Y displacement') 
            if save == True: 
                plt.savefig(f'{parent_dir}sigma_XY_displacement_{file}.pdf', dpi=300)
            plt.show()
            
                            
            

        


        
