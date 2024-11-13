# -*- coding: utf-8 -*-
"""
Plots of the standard deviation of X and Y displacements at 10 ms from tracks estimated by Mosaic Suite and passing SPT analysis
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
save = False
experiment_condition = 'r6cf0p1_l1d5'
step = 2 # 5 ms
colorbar = 'plasma'


ExperDirectory = 'D:/082124_082524/'
experiments = [f for f in os.listdir(ExperDirectory) if os.path.isdir(os.path.join(ExperDirectory, f))]
#experiments = [ 'GEM 3d 35oC']
directory = 'ARTICLE_V3_r6cf0p1_l1d5_R20.9_10trlength_5Derr_5aerr_lagtime10.0ms_nw0.00072T308'


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
            
            if not os.path.exists(f'{parent_dir}{directory}/Data_after_tracking_{file}.csv'):
                continue
            
            trajectories = []
            check = False
            with open(f'{parent_dir}{directory}/Data_after_tracking_{file}.csv') as data:
                data_file = csv.reader(data, delimiter=',')
                for row in data_file: 
                    if 'Trajectory' in row[0]:
                        
                        # Split the string by commas
                        split_string = row[0].split(',')
                        # Strip whitespace from each element
                        stripped_list = [element.strip() for element in split_string]
                        
                        
                        if check==True:
                            trajectory = (name, t, x, y)
                            #if float(stripped_list[1]) > 0.02:
                            trajectories.append(trajectory)
                        check = True
                        row = row[0].split(', ')
                        name = row[0]
                        x = []
                        y = []
                        t = []
                        
                    if row[0] != 'time' and 'Trajectory' not in row[0]:
                        t.append(float(row[0]))
                        x.append(float(row[1]))
                        y.append(float(row[2]))
                trajectory = (name, t, x, y)
                #if float(stripped_list[1]) > 0.02:
                trajectories.append(trajectory) 

                
            #collection data for Heterogenity check
            for trajectory in trajectories:
                X_displacement = []
                Y_displacement = []
                
                name = trajectory[0]
                t = trajectory[1]
                x = trajectory[2]
                y = trajectory[3]
                
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
        plt.xlim(0,0.17)
        plt.ylim(0,0.17)
        cbar = plt.colorbar(sc)
        cbar.set_label('Standard Deviation of Displacement')
        if save == True: 
            plt.savefig(f'{parent_dir}{directory}/sigma_XY_displacement.pdf', dpi=300)
        plt.show()
                    
        for file in files: 
            sigma_X_displacement = [] 
            sigma_Y_displacement = []
    
            print(f'{experiment} {file}')
            
                    
            if not os.path.exists(f'{parent_dir}{directory}/Data_after_tracking_{file}.csv'):
                continue
            
            trajectories = []
            check = False
            with open(f'{parent_dir}{directory}/Data_after_tracking_{file}.csv') as data:
                data_file = csv.reader(data, delimiter=',')
                for row in data_file: 
                    if 'Trajectory' in row[0]:
                        
                        # Split the string by commas
                        split_string = row[0].split(',')
                        # Strip whitespace from each element
                        stripped_list = [element.strip() for element in split_string]
                        
                        
                        if check==True:
                            trajectory = (name, t, x, y)
                            #if float(stripped_list[1]) > 0.02:
                            trajectories.append(trajectory)
                        check = True
                        row = row[0].split(', ')
                        name = row[0]
                        x = []
                        y = []
                        t = []
                        
                    if row[0] != 'time' and 'Trajectory' not in row[0]:
                        t.append(float(row[0]))
                        x.append(float(row[1]))
                        y.append(float(row[2]))
                trajectory = (name, t, x, y)
                #if float(stripped_list[1]) > 0.02:
                trajectories.append(trajectory)               
                
            #collection data for Heterogenity check

            fig, ax = plt.subplots()
            for trajectory in trajectories:
                X_displacement = []
                Y_displacement = []
                
                name = trajectory[0]
                t = trajectory[1]
                x = trajectory[2]
                y = trajectory[3]


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
            cbar.set_label('standard deviation of X displacement')
    
            plt.xlabel('X, um')
            plt.ylabel('Y, um')
            plt.title(f' Trajectory map colored by standard deviations ({file})')
            if save == True:
                plt.savefig(f'{parent_dir}{directory}/xy_heterogenity_map_{file}.pdf', dpi=300)
            #plt.show()

            
            #Van Gogh distribution for displacements
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
            #plt.title(f'Standard deviation Y from standard deviation X')
            plt.xlim(0,0.17)
            plt.ylim(0,0.17)
            if save == True: 
                #plt.savefig(f'{parent_dir}{directory}/sigma_XY_displacement_{file}.pdf', dpi=300)
                plt.savefig(f'{parent_dir}{directory}/sigma_XY_displacement.pdf', dpi=300)
            
            plt.show()
            
                            
            

        


        
