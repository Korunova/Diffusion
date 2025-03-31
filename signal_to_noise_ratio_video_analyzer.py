# -*- coding: utf-8 -*-
"""
Script to estimate and visualize S/N ratio of videos
"""

import matplotlib.pyplot as plt
import csv
import numpy as np
import czifile
import os

path_tracks = r'D:\082124_082524\GEM 3d 35oC\60\r6cf0p1_l1d5\converted_data_length11.csv'
path_video = r'D:\082124_082524\GEM 3d 35oC\60.czi'  # Path to your .czi video file

ExperDirectory = 'D:/082124_082524/'
experiments = [f for f in os.listdir(ExperDirectory) if os.path.isdir(os.path.join(ExperDirectory, f))]
experiments = [['0.24ugml doxycycline 4pm 35oC', '0.24 ugml doxycycline 5pm repeat 2'], 
                ['0.24ugml doxycycline 12_30pm  2d 35oC', '0.24 ugml doxycycline 11am 2day repeat 2'], 
                ['0.5ugml doxycycline 6pm 35oC', '0.5ugml doxycycline 12am repeat2 35oC'], 
                ['0.5ugml doxycycline 2pm 2d 35oC','0.5ugml doxycycline 19_30m day2 repeat2 35oC'], 
                ['1ugml doxycycline 7_20pm 35oC', '1ugml doxycycline 10_30am repeat2 35oC'], 
                ['1ugml doxycycline 3pm 2d 35oC', '1ugml doxycycline 8_30am 2day repeat2 35oC'], 
                ['GEM 3d 35oC', 'GEM 35 repeat', 'GEM 35 repeat 3']]


save = False
show_figure = False
pixel_height = 0.0586000  # 1 pixel ~ 0.0586000 um
scale = 1
traj_length_filter = 10


def calculate_SNR(particle_x, particle_y, frame, background_radius=12, particle_radius=6, method = 'ratio'):
    """
    Calculate the Signal-to-Noise Ratio (SNR) for a given particle in a frame.
    Particle signal is measured in a small circle around the particle.
    Background noise is estimated in a larger ring surrounding the particle.

    Args:
    - particle_x: X coordinate of the particle center
    - particle_y: Y coordinate of the particle center
    - frame: The image data (2D numpy array)
    - background_radius: Radius for the background ring around the particle
    - particle_radius: Radius for the particle circle
    
    Returns:
    - SNR: Signal-to-Noise Ratio
    """
    # Define the coordinates for the particle's circle and the background ring
    y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
    mask_particle = (x - particle_x) ** 2 + (y - particle_y) ** 2 <= particle_radius ** 2
    mask_background = (x - particle_x) ** 2 + (y - particle_y) ** 2 <= (background_radius + particle_radius) ** 2
    mask_background &= ~mask_particle  # Exclude the particle region from the background ring

    # Calculate the mean particle signal (signal inside the particle circle)
    particle_signal = np.mean(frame[mask_particle])

    # Calculate the mean background signal (signal inside the background ring)
    background_signal = np.mean(frame[mask_background])

    # Estimate the noise as the square root of the background signal
    noise = np.sqrt(background_signal)
    
    if method == 'poisson':
        snr = (particle_signal - background_signal) / noise
    else:
        snr = particle_signal/ background_signal

    # Calculate the Signal-to-Noise Ratio (Poisson)
    #snr = (particle_signal - background_signal) / noise
    
    # genelar Signal to Noise Ratio
    #snr = particle_signal/background_signal
    return snr


def plot_particle_radius_pixels(particle_radius_pixels = 4, color = "blue"):
    dpi = plt.gcf().dpi
    pixel_to_point = 72 / dpi  # 1 point = 1/72 inch
    radius_in_points = particle_radius_pixels * pixel_to_point
    area_in_points2 = np.pi * (radius_in_points)**2
    #plt.scatter(x, y, edgecolors='b', facecolors='none', s=30)
    plt.scatter(x, y, edgecolors=color, facecolors='none', s=area_in_points2)
    
for experiment in experiments: 
    
    for repeat in experiment: 
        print(repeat)
        parent_dir = f'{ExperDirectory}{repeat}/'
        files = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
            
        # Prepare a list to save the SNR data
        
        for file in files:
            
            if not os.path.exists(f'{parent_dir}{file}/r6cf0p1_l1d5/'):
                continue
            
            particles_frame_library = {}
            snr_data = [file]
            
            print('file', file, 'particle_library_size_check', len(particles_frame_library.keys()), len(snr_data))
            path_video = f'{parent_dir}{file}.czi'
            # Load the .czi video
            with czifile.CziFile(path_video) as czi:
                video_data = czi.asarray()
            # Check the shape of the video
            #print(f"Video shape: {video_data.shape}")  # Useful for debugging
            
            
            # Read the trajectory data
            path_tracks = rf'{parent_dir}\{file}\r6cf0p1_l1d5\converted_data_length10.csv'
            with open(path_tracks) as data:
                data_file = csv.reader(data, delimiter=',')
                trajectories = []  # list with trajectories for plotting
                particle_list = []
                trajectory_length = 0
                for row in data_file:
                    if row[0] == 'frame': 
                        if trajectory_length > traj_length_filter:
                            for item in particle_list:
                                if item[0] not in particles_frame_library:
                                    particles_frame_library[item[0]] = [(item[1], item[2])]
                                else:
                                    particles_frame_library[item[0]].append((item[1], item[2]))
                                
                        trajectory_length = 0
                        particle_list = []
                    if row[0] != 'frame' and row[0] != 'Trajectory':
                        trajectory_length +=1
                        frame = float(row[0])
                        x = float(row[1]) / scale
                        y = float(row[2]) / scale
                        particle_list.append([frame, x, y])
            
            # Display each frame with particles and compute the SNR for each particle
            for frame_num in range(video_data.shape[0]):
                # Extract the frame (assuming video_data is in the shape (frames, height, width))
                frame = video_data[frame_num, 0, :, :]  # Adjust index depending on the video format
            
                # Plot the frame
                if show_figure:
                    plt.imshow(frame, cmap='Greens', vmin=50, vmax=np.percentile(frame, 99))
            
                # Plot the particles for the current frame
                if frame_num in particles_frame_library:
                    for particle in particles_frame_library[frame_num]:
                        x, y = particle
                        # Calculate the Signal-to-Noise Ratio for each particle
                        snr = calculate_SNR(x, y, frame, background_radius=10, particle_radius = 2)
                        snr_data.append(snr)
                        #print(np.mean(snr_data[1::]))
                        
                        if show_figure:
                            plot_particle_radius_pixels(10, color = 'magenta')
                            plot_particle_radius_pixels(2, color = 'blue')
                            # Display the particle and its SNR
                            #plt.scatter(x, y, edgecolors='b', facecolors='none', s=30)
                            plt.text(x + 5, y + 5, f'SNR: {snr:.2f}', color='red', fontsize=20)
            
                # Add scale bar (1 µm line)
                scale_bar_length_um = 10  # 10 µm
                scale_bar_length_pixels = scale_bar_length_um * (1 / pixel_height)  # Convert 10 µm to pixels
                x_start = 50  # Starting x-coordinate of the scale bar
                y_position = 50  # y-coordinate where the scale bar will be placed
                if show_figure:
                    plt.plot([x_start, x_start + scale_bar_length_pixels], [y_position, y_position], color='black', linewidth=3)
                    plt.text(x_start, y_position - 10, '10 µm', color='black', fontsize=20, ha='left')
                    plt.title(frame_num)
                
                    plt.axis('off')  # Hide axes
                
                    if False:
                        plt.savefig(f'D:/article_GEM_U2OS_expression/Main Revision/pictures/Frame_{frame_num + 1}.png', dpi=300)
                    plt.show()
            
            if save:
                # Save the SNR data to a CSV file
                with open(rf'{parent_dir}particle_snr_data_general_noise_r2_bcgrd10.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if file == files[0]:
                        writer.writerow(['cell idx','SNR'])  # Write header
                    writer.writerow(snr_data)  # Write particle data