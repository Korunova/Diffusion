# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 21:43:32 2024

@author: Huawei
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
taus = [0.015] #min time displacement for MSD calculation 
r2 = 0.9 # filter of diffusion plots try 0
scale = (1/0.0586000)
time_scale = 1 #convert into sec 
save = True
drift_subdtract = False
experiment_condition = 'r6cf0p1_l1d5'
remove_static_error = True
error_file = 'D:082124_082524 folder for not using files in coding/gem FIXED 35Oc/MSD_lagtime5.0ms.csv'

#Theoretical GEM diffusion (Dw) in water calculated from Stokes-Einstein formula
T = 308 #K #variable!
nw = 0.7195*10**(-3) #Pa s #variable!

Kb = 1.38*10**(-23) #J/K (Pa*m3/K)
r = 20*10**(-9) #m
Dw = Kb*T/(6*np.pi*nw*r)*10**12 #um2/sec

ExperDirectory = 'D:/082124_082524/'
experiments = [f for f in os.listdir(ExperDirectory) if os.path.isdir(os.path.join(ExperDirectory, f))]


# Set Times New Roman and font sizes globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [16, 10]  # Adjust based on your needs
plt.rcParams['font.size'] = 35  # Set global font size
plt.rcParams['axes.titlesize'] = 35  # Title font size
plt.rcParams['axes.labelsize'] = 30  # Axis labels font size
plt.rcParams['xtick.labelsize'] = 30  # X-axis tick labels font size
plt.rcParams['ytick.labelsize'] = 30  # Y-axis tick labels font size
plt.rcParams['legend.fontsize'] = 30  # Legend font size

def save_file(data, trajectory_number, columns, filename):
    
    # Transpose the data to make columns as rows
    transposed_data = np.array(data).T.tolist()

    with open(f'{parent_dir}{directory}/{filename}', mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Writing the headers
        writer.writerow([trajectory_number])
        writer.writerow(columns)  # columns = ['column1', 'column2']
        
        # Writing the data
        for row in transposed_data:
            writer.writerow(row)

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


#Collection of velocities during the imaging of one cell
def VelocityAnalysis(trajectories, plot_number = None):
    velocities = {} # time of frame: [Vx], [Vy], delta_t
    Vx = [] #all velocities (x_axis) between frames through all trajectories
    Vy = [] #all velocities (y_axis) between frames through all trajectories
    for trajectory in trajectories: 
        
        name = trajectory[0] 
        
        t = trajectory[1]
        x = trajectory[2]
        y = trajectory[3]
        time_frame = trajectory[4]
    
                
        #Determination of particle velocities in every frame
        for i in range (1, len(t)):
            delta_t = time_frame[i]-time_frame[i-1] #time between current and previous frame
            vi_x = (x[i]-x[i-1])/delta_t #um/sec
            vi_y = (y[i]-y[i-1])/delta_t #um/sec
            Vx.append(vi_x) #
            Vy.append(vi_y) #
            if time_frame[i] not in velocities.keys():
                velocities[time_frame[i]] = [[vi_x], [vi_y], delta_t]
            else:
                velocities[time_frame[i]][0].append(vi_x)
                velocities[time_frame[i]][1].append(vi_y)
                
            
    #velocities = dict(sorted(velocities.items()))
    #for key, value in velocities.items():
    #    print(key, value)
    #    print()
            
    #
    plt.figure(num = plot_number)
    plt.hist(Vx, bins=50, edgecolor='black', color='orange', alpha = 0.5)
    plt.hist(Vy, bins=50, edgecolor='black', color='magenta', alpha = 0.5)
    plt.xlabel(f'Vx (orange) = {np.mean(Vx):.2f} ± {np.std(Vx):.2f} um/sec or Vy(magenta) = {np.mean(Vy):.2f} ± {np.std(Vy):.2f} um/sec')
    plt.ylabel('Frequency')
    plt.title('Distribution of velocities calculated calculated from coordinate displacement between frames for all trajectories')
    plt.grid()
    
    if save == True: 
        plt.savefig(f'{parent_dir}{directory}/velocities_{file}.pdf', dpi=300)
        plt.clf()

    return velocities
    
def FitLinearFunction(name, x, y, points=10, R2=0.9, yerr = None):
    # Define the linear function
    def linear_function(x, a, Da):
        return a*x + np.log(4*Da)
    
    
    # Fit the linear function to the data with uncertainties
    params, covariance = curve_fit(linear_function, x[0:points], y[0:points], sigma = yerr[0:points])
    
    # Extract the fitted parameters
    fitted_a, fitted_Da = params
    
    # Extract the diagonal elements of the covariance matrix as the squared errors (standard errors of the slope and intercept)
    a_error, Da_error = np.sqrt(np.diag(covariance))
        
    #R2 determination
    predicted_msd = linear_function(x[0:points], fitted_a, fitted_Da)
    r_squared = 1 - np.sum((y[0:points] - predicted_msd) ** 2) / np.sum((y[0:points] - np.mean(y[0:points])) ** 2)
    
    return name, float(fitted_a), float(a_error), float(fitted_Da), float(Da_error), r_squared, x[0:points], predicted_msd


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

for experiment in experiments: 
    
    
    # Specify the directory you want to list folders from
    parent_dir = f'{ExperDirectory}{experiment}/'
    files = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    
    #Van Gogh distribution sigma (std dv)
    sigma_X_displacement = [] 
    sigma_Y_displacement = []
    
    for tau in taus:
        
        for min_trajectory_length in min_trajectory_lengths:
            
            directory = f'ARTICLE_V3_{experiment_condition}_R2{r2}_{min_trajectory_length}trlength_5Derr_5aerr_lagtime{tau*1000}ms_nw{nw:.5f}T{T} static'
            
            path = os.path.join(parent_dir, directory)
            
            if save == True:
                os.mkdir(path)
            
            Alpha_AllCells = []
            Diffusion_AllCells = []
            Alpha_sigma_AllCells = []
            Diffusion_Sigma_AllCells = []
            Deff_AllCells = []
            Deff_sigma_AllCells = []
            delta_AllCells = []
            delta_sigma_AllCells = []
            
            
            File_list = []
            

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
                    
                    if save == True:
                        for trajectory in trajectories:
                            data = [trajectory[1], trajectory[2], trajectory[3], trajectory[4]]
                            name = trajectory[0]
                            save_file(data, name, ['time', 'x', 'y', 'time_frame'], f'Trajectory_data_{file}.csv')
                  
                    
                    #Trajectory map and trajectory filter
                    
                    
                    trajectories_filtered = []
                    
                    #collection data for Heterogenity check
                    X_displacement = []
                    Y_displacement = []
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
                            plot(x,y,'x', 'y', plot_number=1)
                         
                    
                        #X and Y analysis of heterogenity
                        step = 1 #for 5 ms lag_time
                        for i in range(0, len(t)-2, step):
                            time_interval = t[i+1] - t[i] 
                            x_displacement = x[i+1] - x[i]
                            y_displacement = y[i+1] - y[i]
                            X_displacement.append(x_displacement)
                            Y_displacement.append(y_displacement)
                            
                        sigma_Y_displacement.append(np.std(Y_displacement))
                        sigma_X_displacement.append(np.std(X_displacement))
                    
                    #Van Gogh distribution for displacements
                    plt.figure()
                    plt.hist(X_displacement, bins=40, edgecolor='black', color='orange', alpha = 0.5)
                    plt.title('Distribution of X in U2OS colony')
                    plt.ylabel('Frequency')
                    plt.xlabel('x displacement, lag time = 5 ms')
                    if save == True: 
                        plt.savefig(f'{parent_dir}{directory}/x_distribution_{file}.pdf', dpi=300)
                    plt.clf()
                    
                    
                    plt.figure()
                    plt.hist(Y_displacement, bins=40, edgecolor='black', color='orange', alpha = 0.5)
                    plt.title('Distribution of Y in U2OS colony')
                    plt.ylabel('Frequency')
                    plt.xlabel('y displacement, lag time = 5 ms')
                    if save == True: 
                        plt.savefig(f'{parent_dir}{directory}/y_distribution_{file}.pdf', dpi=300)
                    plt.clf()
                    
                    
                    #Drift_velicity Analysis
                    
                    #Collection of velocities during the imaging of one cell
                    velocities = VelocityAnalysis(trajectories_filtered, 2)
                    
                    #Determination of Xdrift and Ydrift
                    time = []
                     
                    x_drift_displacement = []
                    x_frame_displacement = []
                    x_drift_std = []
                    x_frame_std = []
                    
                    y_drift_displacement = []
                    y_frame_displacement = []
                    y_drift_std = []
                    y_frame_std = []
                    
                    x_add = 0
                    y_add = 0 
                    xerr_add = 0
                    yerr_add = 0
                    
                    for key, value in  velocities.items():
                        time.append(key)
                        delta_t = value[2]
                        
                        #Sample size check
                        if drift_subdtract == True:
                            if len(value[0]) < 50:
                                print(f'There are sample size of X < 50! Sample_size = {len(value[0])} for frame {key:.4f} sec' )
                        
                        x_add = x_add + np.mean(value[0])*delta_t
                        x_drift_displacement.append(x_add)
                        xerr_add = np.sqrt(xerr_add**2 + (np.std(value[0], ddof = 1)*delta_t)**2)
                        x_drift_std.append(xerr_add)
                        
                        x_frame_displacement.append(np.mean(value[0])*delta_t)
                        x_frame_std.append(np.std(value[0], ddof = 1)*delta_t)
                        
                        
                        y_add = y_add + np.mean(value[1])*delta_t
                        y_drift_displacement.append(y_add)
                        yerr_add = np.sqrt(yerr_add**2 + (np.std(value[1], ddof = 1)*delta_t)**2)
                        y_drift_std.append(yerr_add)
                        
                        y_frame_displacement.append(np.mean(value[1])*delta_t)
                        y_frame_std. append(np.std(value[1], ddof = 1)*delta_t)
                            
                    
                    #ensemble evaraged displacement between frames (black) with std (xy_frame_displacement)
                    #ensemble averaged trajectory (xy_drift_displacement) 
                    plt.figure(3)
                    plt.errorbar(time, x_frame_displacement, yerr = x_frame_std, ecolor='green', alpha = 0.5, color = 'black')
                    plot(time, x_drift_displacement, f'sec; <X> = {np.mean(x_frame_displacement):.4f}±{np.std(x_frame_displacement):.4f} um', 'um (X axis)', plot_number=3)
                    if save == True: 
                        plt.savefig(f'{parent_dir}{directory}/x_frame_displacement_{file}.pdf', dpi=300)
                        plt.clf()
                    plt.figure(4)
                    plt.errorbar(time, y_frame_displacement, yerr = y_frame_std, ecolor='green', alpha = 0.5, color = 'black')
                    plot(time, y_drift_displacement, f'sec; <Y> = {np.mean(y_frame_displacement):.4f}±{np.std(y_frame_displacement):.4f} um', 'um (Y axis)', plot_number=4)
                    if save == True: 
                        plt.savefig(f'{parent_dir}{directory}/y_frame_displacement_{file}.pdf', dpi=300)
                        plt.clf()
                    
                    #ensemble averaged trajectory (xy_drift_displacement) with std
                    plot(time, x_drift_displacement, 'sec', 'um (X axis)', std=x_drift_std, plot_number=6)
                    plot(time, y_drift_displacement, 'sec', 'um (Y axis)', std=y_drift_std, plot_number=7)
            
                    drift = (time, x_drift_displacement, y_drift_displacement)
                    
                    
                    
                    #Drift correction, trajectories map and accumulation of interpolated trajectories
                    trajectories_interpolated = []
                    
                    for trajectory in trajectories_filtered:
                        name = trajectory[0]
                        t = trajectory[1]
                        x = trajectory[2]
                        y = trajectory[3]
                        time_frame = trajectory[4]
                        
                        #Drift Correction and corrected trajectory map
                        if drift_subdtract == True: 
                            #drift data
                            time_drift = drift[0]
                            x_drift = drift[1]
                            y_drift = drift[2]
                            
                            #x and y drift correction
                            for i in range(0, len(time_drift)):
                                for j in range(0, len(time_frame)):
                                    if time_drift[i] == time_frame[j]:
                                        x[j] = x[j]-x_drift[i]
                                        y[j] =  y[j]-y_drift[i]
                                        
                            plot(x,y,'x', 'y', plot_number=5)
                        
                        #Interpolation of trajectories data and drift correction
                        result_x = interpolation(delta_x, t, x, 'sec_function', 'x, um', 6) 
                        result_y = interpolation(delta_x, t, y, 'sec_function', 'y, um', 7)
                    
                        
                        plot(result_x[0], result_x[1], 'sec', 'x um', plot_number=8)
                        plot(result_y[0], result_y[1], 'sec', 'y um', plot_number=9)
                        
                        #Vector Trajectory Plot
                        dx = np.diff([result_x[1][0], result_x[1][-1]])
                        dy = np.diff([result_y[1][0], result_y[1][-1]])
                        directions = np.arctan2(dy, dx) #angle between two arrays 
                        arrow_length = np.sqrt(dx**2 + dy**2) #hypothenuse 
                        colors = plt.cm.viridis(np.linspace(0, 1, 3)) #different colors of vectors
                        n = random.randint(0, 2)  #different colors of vectors
                        
                        plt.figure(num = 10, dpi=100)
                        plt.quiver(result_x[1][0], result_y[1][0], np.cos(directions)*arrow_length, np.sin(directions)*arrow_length, angles='xy', scale_units='xy', scale=1, width = 0.003, color = colors[n])
                        
                        #Accumulation of interpolated trajectories  filtered by length 
                        one_tr = (name, result_x[0], result_x[1], result_y[1])
                        trajectories_interpolated.append(one_tr)
                    
                    if save == True:       
                        for trajectory in trajectories_interpolated:
                            data = [trajectory[1], trajectory[2], trajectory[3]]
                            name = trajectory[0]
                            save_file(data, name, ['time', 'x', 'y'], f'Trajectory_data_Trajectory_Interpolated_data_{file}.csv')
               
                    plt.show()
                    
                    
                    #CALCULATION of Diffusion, mode of movement, total displacement and trajectory length
                    result_dictionary= {} # name: diffusion, diffusion err, alpha, alpha_errir, total displacement and trajectory length
                    Diffusion = [] #temporary to see the result from log plot
                    Diffusion_sigma = [] #standard deviation of D
                    Alpha = [] #temporary to see the result
                    Alpha_sigma = []
                    Tracking_data = [] #names 
                    
                    Deff_cell = []
                    Deff_sigma_cell = []
                    delta_cell = []
                    delta_sigma_cell = []
                    
                    k= 0
                    confined_trajectories = 0
                    all_trajectories = 0
                    for trajectory in trajectories_interpolated:
                        k += 0
                        
                        if k == 3:
                            break
                        
                        name = trajectory[0]
                        t = trajectory[1]
                        x = trajectory[2]
                        y = trajectory[3]
                        
                        
                        tau_list = []
                        msd_list = []
                        msd_std_list = []
                        msd_se_list = []
                        
                        min_step = int(tau/delta_x)
                        
                        for delta_t in range(min_step, len(t), min_step):
                            squared_displacements = []
                            time_intervals = []
                           
                            for i in range(0, len(t) - delta_t, min_step):
                                time_interval = t[i+delta_t]-t[i] 
                                #t[10] - t[0], t[20] - t[10] = 10
                                #t[20] - t[0], t[30] - t[10] = 20 --- delta_t = 2
                                squared_displacement = (x[i + delta_t] - x[i])**2 + (y[i + delta_t] - y[i])**2
                                squared_displacements.append(squared_displacement)
                            
                            mean_msd = np.mean(squared_displacements)
                            sigma_msd = np.std(squared_displacements, ddof=1)

                            # If static error removal is enabled, subtract the static error from the mean MSD
                            if remove_static_error and key in msd_errors:
                                mean_msd -= msd_errors[key][0]
                                sigma_msd = np.sqrt(sigma_msd**2 + sigma_static**2)
                            
                            tau_list.append(time_interval)
                            msd_list.append(mean_msd) #mean MSD from one time interval
                            msd_std_list.append(sigma_msd) #Standard Deviation of MSD
                            #msd_se_list.append(np.std(squared_displacements, ddof=1)/np.sqrt(len(squared_displacements))) #Standard error of MSD, the error from mean
            
                        
                        time_data = np.array(tau_list)
                        msd_data = np.array(msd_list)
                        msd_std_data = np.array(msd_std_list)
                        
                        #Mode of Motion Calulation
                        time_data_log10 = np.log10(time_data)
                        msd_data_log10 = np.log10(msd_data)
                        msd_std_data_log10 = msd_std_data / (msd_data * np.log(10))  # Corrected error calculation log10x' = 1/(<x>*log10)
            
            
                        #result = FitLinearFunction(name, time_data_log10, msd_data_log10, R2=r2, points = (int(round((len(time_data_log10)-1)/3))), yerr = msd_std_data_log10)
                        MSD_limit = (int(round((len(time_data_log10)-1)/2)))
                        p=0
                        for p in range(3, MSD_limit):
                            result = FitLinearFunction(name, time_data_log10[0:MSD_limit], msd_data_log10[0:MSD_limit], R2=r2, points = p, yerr = msd_std_data_log10[0:MSD_limit])
                            if result[5] < r2:
                                break
                        
                        if p>0 and result != None and result[5] > r2:
                            
                            #parameters of Brownian motion
                            Da_value = result[3]
                            a_value = result[1]
                            Da_err = result[4] #Calculate the error in <D>: propagation of uncertainty for functions of single veriables
                            a_err = result[2]
                            
                            #calculation of Deff and its standard deviation using the propagated error formula 
                            Da, a = sp.symbols('Da a')
                            Deff = Da * 2**(2 * (a - 1)) / (Dw**a)
                            partial_Da = sp.diff(Deff, Da) 
                            partial_a = sp.diff(Deff, a)
                            
                            partial_Da_value = float(partial_Da.subs({Da: Da_value, a: a_value}))
                            partial_a_value = float(partial_a.subs({Da: Da_value, a: a_value}))
                            
                            Deff_sigma = np.sqrt((partial_Da_value * Da_err)**2 + (partial_a_value * a_err)**2)
                            Deff_value = float(Deff.subs({Da: Da_value, a: a_value}))
                            
                            #estimation of delta and its standard deviation to show elasticity/viscosity behaviour 
                            delta = a_value*np.pi/2
                            delta_sigma = a_err*np.pi/2
                            
                            if Da_err*5<=Da_value and a_err*5<a_value: 
                                all_trajectories += 1 
                                plot(time_data_log10[0:MSD_limit], msd_data_log10[0:MSD_limit], std=msd_std_data_log10[0:MSD_limit], plot_number=11, title = 'loglog MSD vs tau') #change color
                               
                                #plt.figure()
                                #plt.errorbar(time_data_log10, msd_data_log10, yerr=msd_std_data_log10)
                                #plt.plot(result[6], result[7])
                                #plt.xlabel(f'R2 = {result[5]:.2f}, amount of points = {p-1} from {len(time_data_log10)} points')
                                
                                #Da and a data collection
                                
                                #collect data per one cell
                                Alpha.append(a_value) 
                                Alpha_sigma.append(a_err)
                                Diffusion.append(Da_value)
                                Diffusion_sigma.append(Da_err)
                                
                                #collect data  among all cells
                                Alpha_AllCells.append(a_value) 
                                Alpha_sigma_AllCells.append(a_err)
                                Diffusion_AllCells.append(Da_value)
                                Diffusion_Sigma_AllCells.append(Da_err)
                                
                                #Collection of Deff and delta
                                
                                #collect data per one cell
                                Deff_cell.append(Deff_value)
                                Deff_sigma_cell.append(Deff_sigma)
                                delta_cell.append(delta)
                                delta_sigma_cell.append(delta_sigma)
                                
                                #collect data  among all cells
                                Deff_AllCells.append(Deff_value)
                                Deff_sigma_AllCells.append(Deff_sigma)
                                delta_AllCells.append(delta)
                                delta_sigma_AllCells.append(delta_sigma)
                                
                                Tracking_data.append([t, x, y, Deff_value, Deff_sigma, delta, delta_sigma, result[0], time_data_log10, msd_data_log10, msd_std_data_log10, Da_value, Da_err, a_value, a_err]) # [time, x, y, Deff, Deff_sigma, delta, delta_sigma, name, tau_log10, msd_log10, msd_std_log10, Da_value, Da_err, a_value, a_err]
                                
                    plt.figure(num=11) 
                    plt.xlabel(f'log(tau) (sec), amount of tracks = {all_trajectories}')
                    plt.ylabel('log(MSD) (um2)')
                    if save == True: 
                        plt.savefig(f'{parent_dir}{directory}/MSDall_{file}.pdf', dpi=300)
                    plt.clf()
                        
                    
                    file_data = [file, np.mean(Deff_cell), np.std(Deff_cell), np.mean(delta_cell), np.std(delta_cell), all_trajectories, confined_trajectories]
                    File_list.append(file_data)
                                          
                    
                    plt.figure()
                    plt.hist(Diffusion, bins=40, edgecolor='black', color='orange', alpha = 0.5)
                    plt.ylabel('Frequency')
                    plt.xlabel('um2/sec, diffusion')
                    if save == True: 
                        plt.savefig(f'{parent_dir}{directory}/Diffusion_distribution_{file}.pdf', dpi=300)
                        plt.clf()
                    
                    plt.figure()
                    plt.hist(Alpha, bins=40, edgecolor='black', color='magenta', alpha = 0.5)
                    plt.ylabel('Frequency')
                    plt.xlabel('alpha, type of diffusion')
                    if save == True:
                        plt.savefig(f'{parent_dir}{directory}/Alpha_distribution_{file}.pdf', dpi=300)
                        plt.clf()
                    
                    plt.figure()
                    plt.errorbar(Diffusion, Alpha, xerr=Diffusion_sigma, yerr=Alpha_sigma, fmt='o', color='orange', ecolor='green', elinewidth=2, capsize=4,label='Diffusion from a')
                    plt.xlabel('Diffusion Da from a')
                    plt.ylabel('a')
                    if save == True: 
                        plt.savefig(f'{parent_dir}{directory}/AlphaFromDiffusion_{file}.pdf', dpi=300)
                        plt.clf()
                            
                    # Создание градиента цветов от красного до синего
                    #norm = plt.Normalize(vmin=min(Diffusion), vmax=max(Diffusion))
                    norm = plt.Normalize(vmin=0, vmax=0.1)
                    cmap = cm.get_cmap('coolwarm')
            
                    # Построение графика
                    fig, ax = plt.subplots()
                    for i in range(len(Tracking_data)):
                        ax.plot(Tracking_data[i][1], Tracking_data[i][2], color=cmap(norm(Tracking_data[i][3])))
                        
                    # Добавление цветовой шкалы сбоку
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
                    cbar.set_label('Deff')
            
                    plt.xlabel('X, um')
                    plt.ylabel('Y, um')
                    plt.title(f'Trajectories colored by diffusion ({file})')
                    
                    
                    if save == True:
                        plt.savefig(f'{parent_dir}{directory}/DiffusionMap_{file}.pdf', dpi=300)
                        plt.clf()
                        for trajectory in Tracking_data: # [time, x, y, Deff, Deff_sigma, delta, delta_sigma, name, tau_log10, msd_log10, msd_std_log10, Da_value, Da_err, a_value, a_err]
                            data = [trajectory[0], trajectory[1], trajectory[2]]
                            name = f'{trajectory[7]}, {trajectory[3]}, {(trajectory[4])}, Deff and sigma), {trajectory[5]}, {trajectory[6]}, delta and sigma, {trajectory[11]}, {(trajectory[12])}, um2/sec**a (diffusion and error), {trajectory[13]}, {trajectory[14]}, a and error'
                            save_file(data, name, ['time', 'x', 'y'], f'Data_after_tracking_{file}.csv')
             
                plt.figure()
                plt.errorbar(sigma_X_displacement, sigma_Y_displacement, fmt='o', color='orange', ecolor='green', elinewidth=2, capsize=4)
                plt.xlabel('sigma(delta(X))')
                plt.ylabel('sigma(delta(Y))') 
                if save == True: 
                    plt.savefig(f'{parent_dir}{directory}/sigma_XY_displacement.pdf', dpi=300)
                plt.clf()
             

            
            plt.figure()
            plt.errorbar(Diffusion_AllCells, Alpha_AllCells, xerr=Diffusion_Sigma_AllCells, yerr=Alpha_sigma_AllCells, fmt='o', color='orange', ecolor='green', elinewidth=2, capsize=4,label='Diffusion from a')
            plt.xlabel('Diffusion Da from a')
            plt.ylabel('a')
            if save == True: 
                plt.savefig(f'{parent_dir}{directory}/AlphaFromDiffusion_allcells.pdf', dpi=300)
            plt.show()
                 
            pdf_diffusion = PDF(Diffusion_AllCells)
                   
            # Plot the results
            plt.figure()
            plt.hist(Diffusion_AllCells, bins=30, density=True, alpha=0.5, label='Histogram (normalized)')
            plt.plot(pdf_diffusion[0], pdf_diffusion[1], label='PDF', color='red')
            plt.xlim(0,3)
            plt.ylim(0,3)
            plt.xticks()
            plt.yticks()
            plt.xlabel(f'Da values, <Da> = {np.mean(Diffusion_AllCells):.2f} ± {np.std(Diffusion_AllCells):.2f}, median = {np.median(Diffusion_AllCells):.2f}' , fontsize=20)
            plt.ylabel('Probability Density', fontsize=20)
            plt.title('Histogram and PDF of Da', fontsize=20)
            plt.legend(fontsize=15)
            if save == True: 
                plt.savefig(f'{parent_dir}{directory}/Histogram of Da.pdf', dpi=300)
            plt.show()
            
            pdf_alpha = PDF(Alpha_AllCells)
            
            # Plot the results
            plt.figure()
            plt.hist(Alpha_AllCells, bins=30, density=True, alpha=0.5, label='Histogram (normalized)')
            plt.plot(pdf_alpha[0], pdf_alpha[1], label='PDF', color='red')
            plt.xlim(0,2)
            plt.ylim(0,3)
            plt.xticks()
            plt.yticks()
            plt.xlabel(f'a values, <a> = {np.mean(Alpha_AllCells):.2f} ± {np.std(Alpha_AllCells):.2f}, median = {np.median(Alpha_AllCells):.2f}', fontsize=20)
            plt.ylabel('Probability Density')
            plt.title('Histogram and PDF of a')
            plt.legend()
            if save == True: 
                plt.savefig(f'{parent_dir}{directory}/Histogram of a.pdf', dpi=300)
            plt.show()
            
            print(experiment_condition)
            print(np.mean(Diffusion_AllCells), np.median(Diffusion_AllCells))
            print(np.mean(Alpha_AllCells), np.median(Alpha_AllCells))
            print()
            
            if save == True: 
                 
                with open(f'{parent_dir}{directory}/Cell_Averaged_Data.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                        
                    # Writing the headers
                    writer.writerow(['Name, Deff, Deff_sigma, gamma, gamma_sigma, All_trajectories, Confined trajectories'])

                    # Writing the data
                    for row in File_list:
                        writer.writerow(row)
            
