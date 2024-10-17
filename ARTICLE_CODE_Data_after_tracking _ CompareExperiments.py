# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:32:14 2024

@author: KORUNOVA
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:50:29 2024

@author: KORUNOVA

experiments = ['0.24ugml doxycycline 12_30pm  2d 35oC',
 '0.24ugml doxycycline 4pm 35oC',
 '0.5ugml doxycycline 2pm 2d 35oC',
 '0.5ugml doxycycline 6pm 35oC',
 '1ugml doxycycline 3pm 2d 35oC',
 '1ugml doxycycline 7_20pm 35oC',
 'GEM 3d 35oC',
 'ShL1 35oC']

"""
'''
exp_cond = ['0.24ug/ml dox 2 days',
 '0.24u/gml dox 1 day',
 '0.5u/gml dox 2 days',
 '0.5u/gml dox 1 day',
 '1u/gml dox 2 days',
 '1u/gml dox 1 day',
 'GEM',
 'fixed GEM',
 'GEM paGFP']
'''

#Collect the distribution of lengths 

#remove np.mean Deff collection and take it from distribution

#add statistics

import csv 
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


directory = 'ARTICLE_V3_r6cf0p1_l1d5_R20.9_10trlength_5Derr_5aerr_lagtime10.0ms_nw0.00072T308'
save = False
Deff_filter = False

ExperDirectory = 'D:082124_082524/'
experiments = [f for f in os.listdir(ExperDirectory) if os.path.isdir(os.path.join(ExperDirectory, f))]

exp_cond = ['0.24ug/ml dox 2 days',
 '0.24u/gml dox 1 day',
 '0.5u/gml dox 2 days',
 '0.5u/gml dox 1 day',
 '1u/gml dox 2 days',
 '1u/gml dox 1 day',
 'GEM high expression',
 'GEM low expression']

Deff_boxplot = []
Alpha_boxplot = []
Trajectory_length_boxplot = []
Particles_boxlot = []


# Set Times New Roman and font sizes globally
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.figsize'] = [20, 14]  # Adjust based on your needs
plt.rcParams['font.size'] = 35  # Set global font size
plt.rcParams['axes.titlesize'] = 35  # Title font size
plt.rcParams['axes.labelsize'] = 30  # Axis labels font size
plt.rcParams['xtick.labelsize'] = 30  # X-axis tick labels font size
plt.rcParams['ytick.labelsize'] = 30  # Y-axis tick labels font size
plt.rcParams['legend.fontsize'] = 20  # Legend font size




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


def Plot(labels, distributions, xlabel = None, ylabel = None, y_limits = None, round_number = 3):
    medians = [np.round(np.median(group), round_number) for group in distributions]
    means = [np.round(np.mean(group),  round_number) for group in distributions]
    stds = [np.round(np.std(group),  round_number) for group in distributions]
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize = (40,32))
    
    # Plotting distributions as violin plots
    sns.violinplot(data=distributions, ax=ax)
    
    # Adjust the axis
    ax.set_xticks(range(len(list(labels))))
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=90, ha='right')
    plt.xticks(rotation=90, ha='right')
    plt.yticks()
    plt.ylim(y_limits)
    
    
    if True:
        plt.savefig(f'{ExperDirectory}/Violin_Plot_{ylabel}_DeffFilter{Deff_filter}.pdf', dpi=300)
    
    plt.show()       
    
    
    # Create a box plot
    plt.figure(figsize = (40,32))
    plt.boxplot(distributions, vert=True, patch_artist=False, labels=labels)
    
    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} distributions Vs experimental condition')
    plt.xticks()
    plt.xticks(rotation=90, ha='right')
    plt.yticks()
    plt.ylim(y_limits)
    
    # Annotate with mean, median, and std
    for i in range(len(distributions)):
        x_pos = i + 1  # X position for each box plot
        # Text showing the mean, median, and std under each boxplot
        plt.text(x_pos, y_limits[1] - (0.05 * (y_limits[1] - y_limits[0])), 
                 f'Mean: {means[i]:.3f}\nStd: {stds[i]:.3f}\nMedian: {medians[i]:.3f}', 
                 ha='center', fontsize=20, rotation=0)
    
    
    if save == True:
        plt.savefig(f'{ExperDirectory}/ARTICLE_V3_Brownian_Article_l10_lagtime10_Boxplot_All_{ylabel}_DeffFilter{Deff_filter}.pdf', dpi=300)
    
    # Show the plot
    plt.show()


# Bootstrapping function
def bootstrap_compare(dist1, dist2, n_resamples=10000, statistic=np.mean):
    """
    Bootstrap resampling to compare two distributions.
    :param dist1: First distribution
    :param dist2: Second distribution
    :param n_resamples: Number of bootstrap samples
    :param statistic: Function to compute statistic (default: np.mean)
    :return: Resampled statistic differences
    """
    # Bootstrap resampling for each distribution
    #resample1 = np.random.choice(dist1, (n_resamples, len(dist1)), replace=True).mean(axis=1)
    #resample2 = np.random.choice(dist2, (n_resamples, len(dist2)), replace=True).mean(axis=1)
    
    resample1 = np.array([np.median(np.random.choice(dist1, len(dist1), replace=True)) for _ in range(n_resamples)])
    resample2 = np.array([np.median(np.random.choice(dist2, len(dist2), replace=True)) for _ in range(n_resamples)])
    
    
    # Calculate the difference between the statistics (e.g., means)
    diff_stat = resample1 - resample2
    
    return diff_stat


def distribution_comparison(distributions, param = None):
    
    diff_stat_6_7 = bootstrap_compare(distributions[7], distributions[6])
    sns.kdeplot(diff_stat_6_7, label=exp_cond[6])
    conf_interval_6_7 = np.percentile(diff_stat_6_7, [2.5, 97.5])
    
    diff_stat_7_7 = bootstrap_compare(distributions[7], distributions[7])
    sns.kdeplot(diff_stat_7_7,  label=exp_cond[7])
    conf_interval_7_7 = np.percentile(diff_stat_7_7, [2.5, 97.5])

    
    for d in range(0,len(Deff_boxplot)-3):
        # Compare first distribution with fifth
        diff_stat_d_6 = bootstrap_compare(distributions[7], distributions[d])
        # Compute confidence intervals for the differences
        conf_interval_d_6 = np.percentile(diff_stat_d_6, [2.5, 97.5])
        
        sns.kdeplot(diff_stat_d_6, label=exp_cond[d])
        
        plt.title(f'Bootstrap Comparison of {param} Distributions')
        plt.xlabel('Difference in Medians')
        plt.ylabel('Density')    
        plt.legend()
        #plt.show()
        

        
        # Output the confidence intervals
       
        print(f"95% CI for difference between {exp_cond[7]} and {exp_cond[d]}: {conf_interval_d_6}")
    print(f"95% CI for difference between {exp_cond[7]} and {exp_cond[6]}: {conf_interval_6_7}")
    print(f"95% CI for difference between {exp_cond[7]} and {exp_cond[7]}: {conf_interval_7_7}")
        
    if save == True:
        plt.savefig(f'{ExperDirectory}/ARTICLE_V3_powerlaw_dynamic_Article_Bootstrap_{param}_Deff_DeffFilter{Deff_filter}.pdf', dpi=300)
    
    plt.show()
    
    
    
    
    

for experiment in experiments: 
    print(f'{experiment}')
    
    
    parent_dir = f'{ExperDirectory}{experiment}/'
    files = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
    
    
    
    Diffusion = []
    Alpha = []
    Trajectory_length = [] 
    Particles = []

        
    for file in files: 
        
        if not os.path.exists(f'{parent_dir}{directory}/Data_after_tracking_{file}.csv'):
            continue
        
        check = False 
        number = 0
        with open(f'{parent_dir}{directory}/Data_after_tracking_{file}.csv') as data:
            
            data_file = csv.reader(data, delimiter=',')
            
            for row in data_file:
                
                if 'Trajectory' in row[0]:
                    
                    
                    if check == True:
                        Trajectory_length.append(t[-1]*1000)
                        
                    # Split the string by commas
                    split_string = row[0].split(',')
                    # Strip whitespace from each element
                    stripped_list = [element.strip() for element in split_string]
                    #print(stripped_list)
                    
                    if Deff_filter == True:
                        if float(stripped_list[1]) > 0.02:        
                            Diffusion.append(float(stripped_list[1]))
                            Alpha.append(float(stripped_list[10]))
                    else:
                        Diffusion.append(float(stripped_list[1]))
                        Alpha.append(float(stripped_list[10]))
                        
                    t = []
                    
                    check = True
                    number+=1
                    
                if len(row)==3 and 'time' not in row:
                    t.append(float(row[0]))
                
        Particles.append(number)
        Trajectory_length.append(t[-1]*1000)
                    
    Particles_boxlot.append(Particles)    
    Diffusion = np.array(Diffusion)
    Alpha = np.array(Alpha)
    Trajectory_length = np.array(Trajectory_length)
        
    Deff_boxplot.append(Diffusion)
    Alpha_boxplot.append(Alpha)
    Trajectory_length_boxplot.append(Trajectory_length)


Plot(exp_cond, Deff_boxplot, 'Experimental Condition', 'Deff', y_limits = [0, 0.08], round_number=2)
Plot(exp_cond, Alpha_boxplot, 'Experimental Condition', 'a', y_limits = [0, 2], round_number=1)
Plot(exp_cond, Trajectory_length_boxplot, 'Experimental Condition', 'trajectory length, ms', y_limits=[50,300],  round_number = 0)
Plot(exp_cond, Particles_boxlot, 'Experimental Condition', 'particles amount', y_limits = [0, 1000],  round_number = 0)


#distribution_comparison(Deff_boxplot, param = 'Deff')
#distribution_comparison(Alpha_boxplot, param = 'a')
#distribution_comparison(Trajectory_length_boxplot, param = 'trajecrotory length, ms')
#distribution_comparison(Particles_boxlot, param = 'particles amount')