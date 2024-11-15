# -*- coding: utf-8 -*-
"""
The script processes outputs generated by 'Deff_calculation.py' or 'Deff_calculation_direct_dynamic_error.py', including Deff distributions, to estimate Spearman and Pearson correlation coefficients for the 
dependency of parameters on fluorescent intensity, which is separately collected using Fiji, across cell population.
"""

#Collect the distribution of lengths 

#remove np.mean Deff collection and take it from distribution

#add statistics

import csv 
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import os
import seaborn as sns


directory = 'ARTICLE_V3_r6cf0p1_l1d5_R20.9_10trlength_5Derr_5aerr_lagtime10.0ms_nw0.00072T308'
save = True
Deff_filter = False
Intenisities_file = 'Intenisties.csv'

ExperDirectory = 'D:082124_082524/'
experiments = [f for f in os.listdir(ExperDirectory) if os.path.isdir(os.path.join(ExperDirectory, f))]
#experiments = ['GEM 3d 35oC']

# Set Times New Roman and font sizes globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [16, 10]  # Adjust based on your needs
plt.rcParams['font.size'] = 35  # Set global font size
plt.rcParams['axes.titlesize'] = 35  # Title font size
plt.rcParams['axes.labelsize'] = 30  # Axis labels font size
plt.rcParams['xtick.labelsize'] = 30  # X-axis tick labels font size
plt.rcParams['ytick.labelsize'] = 30  # Y-axis tick labels font size
plt.rcParams['legend.fontsize'] = 30  # Legend font size




def Boxplot_And_Statistics(x_values, distributions, x_label = None, y_label = None, save_name = None, x_limit = None, y_limit = None):
    means = [np.mean(group) for group in distributions]
    std_dev = [np.std(group, ddof = 1) for group in distributions]
    medians = [np.median(group) for group in distributions]
    
    
    #Do std dev another color or alpha!
    plt.figure()
    plt.errorbar(x_values, means, yerr = std_dev, fmt='o', color = 'blue', label = 'mean',ecolor = 'blue', alpha = 0.3)
    plt.errorbar(x_values, medians, fmt='o', color = 'magenta', label= 'median', alpha = 0.3)
    plt.xlabel(f'{x_label}')
    plt.ylabel(f'mean of {y_label}')
    plt.ylim(y_limit)
    if y_label == 'Deff':
        plt.axvline(x=123-80, color='red', linestyle='--', linewidth=2)
    plt.legend()
    if save == True:
        plt.savefig(f'{parent_dir}{directory}/ARTICLE_V3_powerlaw_{save_name}_errorbar_DeffFilter{Deff_filter} scaled.pdf', dpi=300)
    plt.show()
    
    plt.figure()
    plt.errorbar(std_dev, means, fmt='o')
    plt.xlabel(f'standard deviation of {y_label}')
    plt.ylabel(f'mean of {y_label}')
    plt.show()
    

    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 10))
    # Plot boxplots at the positions specified by x_values
    for i, (x, y) in enumerate(zip(x_values, distributions)):
        bp = ax.boxplot(y, positions=[x], widths=3, patch_artist=True)  # You can adjust widths as needed
        # Set transparency for each component
        for box in bp['boxes']:
            box.set(facecolor='white', alpha=0.5)  # Adjust 'alpha' as needed for transparency
        for whisker in bp['whiskers']:
            whisker.set(color='black', alpha=0.5)
        for cap in bp['caps']:
            cap.set(color='black', alpha=0.5)
        for median in bp['medians']:
            median.set(color='red', alpha=1)
        for flier in bp['fliers']:
            flier.set(marker='o', color='red', alpha=0.5)

    # Adjust the axis
    #ax.set_xticks(np.arange(min(Int_sorted), max(Deff_boxplot_sorted) + 1, 0.5))  # Adjust tick frequency
    # Adjust the x-axis to include all ticks, including the last one
    tick_positions = x_values[::20]  # Show every 5th tick
    if x_values[-1] not in tick_positions:
        tick_positions = np.append(tick_positions, x_values[-1])

    ax.set_xticks(tick_positions)  # Set the tick positions including the last one
    ax.set_xticklabels(tick_positions, rotation=90, ha='right')  # Set the tick labels
    ax.set_xlabel(f'{x_label}')
    ax.set_ylabel(f'{y_label}')
    
    ax.set_xlim(x_limit)  # Set x-axis limits
    ax.set_ylim(y_limit)  # Set y-axis limits
    
    if y_label == 'Deff':
        plt.axvline(x=123-80, color='red', linestyle='--', linewidth=2)
    
    if save == True:
        plt.savefig(f'{parent_dir}{directory}/ARTICLE_V3_powerlaw{save_name}_boxplot_DeffFilter{Deff_filter}.pdf', dpi=300)

    plt.show()

    # Statistics

    # Spearman correlation (monotonic relationship)
    spearman_corr, spearman_p = spearmanr(x_values, medians)
    print(f"Spearman correlation: {spearman_corr}, p-value: {spearman_p}")

    # Pearson correlation (linear relationship)
    pearson_corr, pearson_p = pearsonr(x_values, medians)
    print(f"Pearson correlation: {pearson_corr}, p-value: {pearson_p}")
    
     #Create a DataFrame with the results
    results = {
        'Experiment': [experiment, experiment],
        'Statistic': ['Spearman Correlation', 'Pearson Correlation'],
        'Value': [spearman_corr, pearson_corr],
        'p-value': [spearman_p, pearson_p]
        }
    
    df = pd.DataFrame(results)
    
    # Save the DataFrame to an Excel file
    if save == True:
        output_file = f'{ExperDirectory}ARTICLE_V3_PowerLaw_{save_name}_statistics_results_DeffFilter{Deff_filter}.csv'
        
        # Check if the file exists to avoid writing the header multiple times
        try:
            # Append to the file and avoid writing header if the file already exists
            df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        except Exception as e:
            print(f"An error occurred while saving: {e}")
        else:
            print(f"Results for {experiment} have been saved to {output_file}")
            
def PDF(data, bins_value = 30):
    # Create histogram
    counts, bin_edges = np.histogram(data, bins=bins_value, density=False)
    
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
    print(f'{experiment}')
    
    parent_dir = f'{ExperDirectory}{experiment}/'
    files = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
     
    Deff_AllCells = []
    Deff_sigma_AllCells= []
    Delta_AllCells = []
    Alpha_AllCells = []
    
    Deff_boxplot = []
    Delta_boxplot = []
    Alpha_boxplot = []
    Trajectory_length_boxplot = []
    Files_count_Deff = [] #to check data from Data_after_tracking and Intensities
    
    #extract intensities and prepare data
    # Load the CSV file
    data = pd.read_csv(f'{parent_dir}{Intenisities_file}')  # Use delimiter for space or tab
    
    # Manually assign column names
    data.columns = ['file', 'Area', 'Mean', 'StdDev', 'IntDen', 'RawIntDen']
    
    # Ensure your CSV has 'x' and 'y' columns
    Files_count_Intensities = data['file']
    Int = data['Mean']
    Int_sigma = data['StdDev']
    
    Files_count_Intensities = list(Files_count_Intensities)
    Int = list(Int)
    Int_sigma = list(Int_sigma)
    
    if experiment == 'ShL1 35oC':
        Int = [round((x-80)*18.56/13.76,0)  for x in Int]
    else:
        Int = [round(x-80,0) for x in Int]
    
    Int_experiment = []
    
    for file in files: 
        
        if not os.path.exists(f'{parent_dir}{file}/r6cf0p1_l1d5/'):
            continue
        
        if not os.path.exists(f'{parent_dir}{directory}/Data_after_tracking_{file}.csv'):
            continue
        
        
        Diffusion = []
        Delta = []
        Alpha = []
        Diffusion_error = []
        check = False 
        Trajectory_length = [] 
        
        with open(f'{parent_dir}{directory}/Data_after_tracking_{file}.csv') as data:
            data_file = csv.reader(data, delimiter=',')
                        
            # Создание градиента цветов от красного до синего
            norm = plt.Normalize(vmin=0, vmax=0.05)
            #norm = plt.Normalize(vmin=0.8, vmax=np.pi/2)
            cmap = cm.get_cmap('plasma')
    
            # Построение графика
            fig, ax = plt.subplots()
            
            traj_count = 0
            for row in data_file:
                
                if 'Trajectory' in row[0]:
                    if check == True:
                        #space to plot
                        ax.plot(x, y, color=cmap(norm(float(stripped_list[1]))))
                        Trajectory_length.append(t[-1]*1000)
                        
                    # Split the string by commas
                    split_string = row[0].split(',')
                    # Strip whitespace from each element
                    stripped_list = [element.strip() for element in split_string]
                    
                    if Deff_filter == True: 
                        if float(stripped_list[1]) > 0.02:
                            Diffusion.append(float(stripped_list[1]))
                            Diffusion_error.append(float(stripped_list[2]))
                            Delta.append(float(stripped_list[4]))
                            Alpha.append(float(stripped_list[10]))
                        
                            Deff_AllCells.append(float(stripped_list[1]))
                            Deff_sigma_AllCells.append(float(stripped_list[2]))
                            Delta_AllCells.append(float(stripped_list[4]))
                            Alpha_AllCells.append(float(stripped_list[10]))
                    else: 
                            Diffusion.append(float(stripped_list[1]))
                            Diffusion_error.append(float(stripped_list[2]))
                            Delta.append(float(stripped_list[4]))
                            Alpha.append(float(stripped_list[10]))
                        
                            Deff_AllCells.append(float(stripped_list[1]))
                            Deff_sigma_AllCells.append(float(stripped_list[2]))
                            Delta_AllCells.append(float(stripped_list[4]))
                            Alpha_AllCells.append(float(stripped_list[10]))
                    
                    t = []
                    x = []
                    y = []
                    
                    check = True
                    
                if len(row)==3 and 'time' not in row:
                    t.append(float(row[0]))
                    x.append(float(row[1]))
                    y.append(float(row[2]))
            
        ax.plot(x, y, color=cmap(norm(float(stripped_list[1]))))
        Trajectory_length.append(t[-1]*1000)
        
        #color trajectorymap according to alpha parameter
        # Добавление цветовой шкалы сбоку
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Deff')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        #plt.title(f'Trajectories colored by a ({file})')
        plt.title('Trajectories colored by Deff, power-law fit')
        
        
        if save == True:
            plt.savefig(f'{parent_dir}{directory}/DiffusionMap_{file}.pdf', dpi=300)
        plt.show()
        
        if not Diffusion:
            print(f'no tracks, {file}')
        else:
            for file_index in range(0, len(Files_count_Intensities)):
                if float(file) == Files_count_Intensities[file_index]:
                    Int_experiment.append(Int[file_index])
        
        Diffusion = np.array(Diffusion)
        Delta = np.array(Delta)
        Alpha = np.array(Alpha)
        Diffusion_error = np.array(Diffusion_error)
        Trajectory_length = np.array(Trajectory_length)
        
        Deff_boxplot.append(Diffusion)
        Delta_boxplot.append(Delta)
        Alpha_boxplot.append(Alpha)
        Trajectory_length_boxplot.append(Trajectory_length)
        

        
        
    Deff_AllCells = np.array(Deff_AllCells)
    Deff_sigma_AllCells= np.array(Deff_sigma_AllCells)
    Delta_AllCells=np.array(Delta_AllCells)
    Alpha_AllCells=np.array(Alpha)
    
    #for distribution in Deff_boxplot:
                
    #    pdf_Deff_cell = PDF(distribution)
    #    plt.figure()
    #    plt.plot(pdf_Deff_cell[0], pdf_Deff_cell[1], label='PDF')
    #    plt.xlim(0,0.05)
    #    plt.ylim(0,80)
    #    plt.xticks()
    #    plt.yticks()
    #    plt.xlabel(f'Deff values by cell')
    #    plt.ylabel('Probability Density')
    #    plt.title('Histogram and PDF of Deff')
        
    #plt.show()
    
    pdf_Deff = PDF(Deff_AllCells, 30)
    
    # Plot the results
    
    Deff_median = np.median(Deff_AllCells)
    Deff_mean = np.mean(Deff_AllCells)
    Deff_mean_sigma = np.std(Deff_AllCells, ddof=1)

    plt.figure()
    plt.hist(Deff_AllCells, bins=30, density=True, alpha=0.5, label='Histogram (normalized)')
    plt.plot(pdf_Deff[0], pdf_Deff[1], label='PDF', color='red')
    plt.xlim(0,0.05)
    plt.ylim(0,80)
    plt.xticks()
    plt.yticks()
    plt.xlabel(f'Deff values, <Deff> = {Deff_mean:.2f} ± {Deff_mean_sigma:.2f}, median = {Deff_median:.2f}')
    plt.ylabel('Probability Density')
    plt.title('Histogram and PDF of Deff')
    plt.legend()
    if save == True:
        plt.savefig(f'{parent_dir}{directory}/Deff histogram_median_DeffFilter{Deff_filter}.pdf', dpi=300)
    plt.show()
    
    
    pdf_Delta = PDF(Delta_AllCells)
    
    # Plot the results
    
    Delta_median = np.median(Delta_AllCells)
    Delta_mean = np.mean(Delta_AllCells)
    Delta_mean_sigma = np.std(Delta_AllCells, ddof=1)

    plt.figure()
    plt.hist(Delta_AllCells, bins=30, density=True, alpha=0.5, label='Histogram (normalized)')
    plt.plot(pdf_Delta[0], pdf_Delta[1], label='PDF', color='red')
    plt.xlim(0,np.pi/2*1.5)
    plt.axvline(x=np.pi/2, color='red', linestyle='--', linewidth=2)
    plt.ylim(0,2.5)
    plt.xticks()
    plt.yticks()
    plt.xlabel(f'phase angle, <δ> = {Delta_mean:.1f} ± {Delta_mean_sigma:.1f}, median = {Delta_median:.1f}')
    plt.ylabel('Probability Density')
    plt.title('Histogram and PDF of phase angle')
    plt.legend()
    if save == True:
        plt.savefig(f'{parent_dir}{directory}/Phase Angle histogram_median_DeffFilter{Deff_filter}.pdf', dpi=300)
    plt.show
    
    #DEFF VIOLIN PLOT
    
    # Set up the figure and axis
    #fig, ax = plt.subplots(figsize=(18, 10))
    
    # Plotting distributions as violin plots
    #sns.violinplot(data=Deff_boxplot, ax=ax)
    
    # Adjust the axis
    #ax.set_xticks(range(len(list(Int_experiment))))
    #ax.set_xticklabels(Int_experiment)
    #ax.set_xlabel('X values')
    #ax.set_ylabel('Y values')
    #ax.set_title('Plot with Distributions at Each Data Point')
    #plt.xticks(fontsize=20,rotation=90, ha='right')
    #plt.xticks(rotation=90, ha='right')
    #plt.yticks(fontsize=20)
    #if save == True:
    #    plt.savefig(f'{parent_dir}{directory}/Deff violin plot_DeffFilter{Deff_filter}.pdf', dpi=300)
    #plt.show
    #plt.show()       
        
    
    #Boxplot_And_Statistics(Int_experiment, Deff_boxplot, 'Fluorescent Intensity', 'Deff', 'Deff',  x_limit = [0, 300], y_limit = [0, 0.06])
    Boxplot_And_Statistics(Int_experiment, Alpha_boxplot, 'Fluorescent Intensity', 'a' ,'a',  x_limit = [0,300], y_limit = [0, 1.4])
   
 #    Boxplot_And_Statistics(Int_experiment, Delta_boxplot, 'Fluorescent Intensity', 'phase angle', 'phase angle', x_limit = [0,300], y_limit = [0, np.pi/2*1.5])
 #    Boxplot_And_Statistics(Int_experiment, Trajectory_length_boxplot, 'Fluorescent Intensity', 'Trajector_length ms' ,'Trajectories_length',  x_limit = [0,300], y_limit = [50, 400])
           
    
