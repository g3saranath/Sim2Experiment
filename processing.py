import os
from AtomicImageSimulator.main import *
import pandas as pd
import numpy as np
import random
import atomai as aoi
from sklearn.preprocessing import StandardScaler
import gpax
import copy 
import dklgpreg, gptrainer
from atomai.nets import fcFeatureExtractor
from sklearn.model_selection import train_test_split
import dklgpreg, gptrain
import torch
import matplotlib.pyplot as plt
import gc
from statistics import mean 
import copy
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os, re
import sys
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import FormatStrFormatter
import pickle
import atomai as aoi
np.random.seed(0)



global dt, scaler
scaler_fun = StandardScaler()

dt = aoi.transforms.datatransform(1, # number of classes
                                  gauss_noise=[1000, 2000], # scaled values
                                  poisson_noise=[30, 45],
                                  blur=False,
                                  contrast=True,
                                  zoom=False,
                                  #resize=[2, 1], # downsize all the images by a factor of 2
                                  seed=1)
def augmented_simulations(cropped_image_list,cropped_mask_list):
    X_train, y_train = dt.run(np.array(cropped_image_list), np.array(cropped_mask_list)[..., None])
    return X_train, y_train

#@title Utility functions
def extract_coordinates(f_s,label_name):
  cell = np.array(f_s[2:5], dtype=np.float64)

  num_atoms = int(f_s[6][0])+int(f_s[6][1])
  n = num_atoms + 1

  num_Mo = int(f_s[6][0])
  num_S = int(f_s[6][1])

  #extracting coordinates for each trajectory
  #combined in a dictionary
  traj = {}
  for i in range(7, len(f_s), n):
    coord = np.array(f_s[i + 1 : i + n], dtype=np.float64)
    coord = np.dot(coord, cell)
    traj[int(f_s[i][-1])] = coord

  #num of trajectories available
  v = list(traj.values())
  return traj, v, label_name, cell, num_Mo, num_S

def generate_img_mask_pair(local_v,k,cell,num_Mo, num_S,label_name,f_s):
  #read in coordinates from specific trajectory
  xy = local_v[k][:, :2]

  #boundary conditions
  for i in range(xy.shape[0]):
    if xy[i,0] < 0:
      mod_x = cell[0][0] + xy[i,0]
      xy[i,0] = mod_x

  for j in range(xy.shape[0]):
    if xy[j,1] < 0:
      mod_y = cell[1][1] + xy[j,1]
      xy[j,1] = mod_y

  c1 = np.repeat('Mo', int(f_s[6][0]))
  c2 = np.repeat('S', int(f_s[6][1]))

  lattice_coordinates_Mo = np.concatenate((c1[:, None], xy[:num_Mo]), axis=1)
  lattice_coordinates_S = np.concatenate((c2[:, None], xy[num_Mo:]), axis=1)
  lattice_coordinates = np.concatenate((lattice_coordinates_Mo, lattice_coordinates_S), axis=0)

  # angstrom to pixel conversion coefficient
  ang2px = 10.5
  # angle to rotate coordinates (min, max, step; sampled randomly)
  ang = 0
  # list with atom "widths" (sampled randomly)
  sc = [13, 15]
  # blurring due to overlap with a probe
  convprobe = 1.5

  sl = SimulateLattice(lattice_coordinates, ang2px, ang, sc, convprobe=convprobe)
  img, mask = sl.make_image(r_mask=6)
  return img, mask

def dump_with_pickle(pyObj, filename):
    fobj1 = open(filename, 'wb')
    pickle.dump(pyObj, fobj1)
    fobj1.close()

def load_with_pickle(filename):
    fobj1 = open(filename, 'rb')
    pyObj = pickle.load(fobj1)
    fobj1.close()
    return pyObj

def xdatcar_processor(directory,n_sims):

    ind_to_val = {val:None for val in range(n_sims)}
    val = 0
    image_list = []
    mask_list = []
    label_list = []

    for filename in os.listdir(directory):
        if "." in filename:
            continue
        ind_to_val[val] = filename.split('_')[1]
        val += 1
        print(ind_to_val)
        
        f = os.path.join(directory+filename, f"XDATCAR_df_S_{filename.split('_')[1]}")
        
        f_ck = open(f)
        f_ = f_ck.readlines()
        label_name = f.split('/')[-1]
        f_s = []
        for l in f_:
            f_s.append(l.strip("\n").strip().split())
        f_ck.close()
        

        #extracting coordinates from each AIMD trajectory
        traj, v, label_name, cell, num_Mo, num_S = extract_coordinates(f_s,label_name)
        #print(label_name)
        #img-mask pair for each trajectory
        local_img_list=[]
        local_mask_list=[]
        for k in range(len(traj)):
            traj_img, traj_mask = generate_img_mask_pair(v, k, cell, num_Mo, num_S, label_name,f_s)
            local_img_list.append(traj_img)
            local_mask_list.append(traj_mask)
            #print(k)
            #print(traj_img.shape, traj_mask.shape)
        # image_list.append(local_img_list)
        # mask_list.append(local_mask_list)
        # label_list.append(label_name)
        '''
        f = os.path.join(directory+filename, f"XDATCAR_df_S_{filename.split('_')[1]}_1")
        f_ck = open(f)
        f_ = f_ck.readlines()
        label_name = f.split('/')[-1]
        f_s = []
        for l in f_:
            f_s.append(l.strip("\n").strip().split())
        f_ck.close()

        local_img_list_2 = []
        local_mask_list_2 = []
        #extracting coordinates from each AIMD trajectory
        traj, v, label_name2, cell, num_Mo, num_S = extract_coordinates(f_s,label_name)
        #print(label_name)
        #img-mask pair for each trajectory
        for k in range(len(traj)):
            traj_img, traj_mask = generate_img_mask_pair(v, k, cell, num_Mo, num_S, label_name,f_s)
            local_img_list_2.append(traj_img)
            local_mask_list_2.append(traj_mask)
            #print(k)
            #print(traj_img.shape, traj_mask.shape)
        '''
        image_list.append(local_img_list)#+local_img_list_2)
        mask_list.append(local_mask_list)#+local_mask_list_2)
        label_list.append(label_name)#+label_name2)

    return ind_to_val,image_list,mask_list,label_list

def oszicar_generation_additional(file_path,xdatcar_path,oszicar_path,exp_num,last_iter):

    Energy_ref = -765.3104674489797

    last_traj = None
    total_atoms = None

    # File path to the XDATCAR file
    xdatcar_filepath = file_path + xdatcar_path
    oszicar_filepath = file_path + oszicar_path


    # Read the file
    with open(xdatcar_filepath, 'r') as file:
        lines = file.readlines()
    

    atoms_line = lines[6].strip()
    atom_counts = [int(count) for count in atoms_line.split() if count.isdigit()]
    total_atoms = sum(atom_counts)
    print("Total number of atoms:", total_atoms)

    # Iterate through lines in reverse order to find the last occurrence of 'Direct configuration='
    for line in reversed(lines):
        if 'Direct configuration=' in line:
            last_traj = line.split('=')[-1].strip()
            break  # Stop after finding the last occurrence

    # Print the results
    if last_traj is not None:
        direct_configuration_number = float(last_traj)  # Convert to float or int as needed
        print("Last trajectory Number:", direct_configuration_number)
    else:
        print("No line with 'Direct configuration=' found in the file.")

    # Convert last_traj to an integer
    last_traj = int(last_traj)

    # Initialize empty lists to store 'E' and 'T' values
    energy_values = []
    temperature_values = []

    # Read the OSZICAR file
    with open(oszicar_filepath, 'r') as file:
        lines = file.readlines()

    # Loop through each line to find and extract 'E' and 'T' values
    for line in lines:
        if 'E=' in line:
            energy = float(line.split('E=')[1].split()[0])
            energy_values.append(energy)
        if 'T=' in line:
            temperature = float(line.split('T=')[1].split()[0])
            temperature_values.append(temperature)
    
    # with open(oszicar_filepath+"_1", 'r') as file:
    #     lines = file.readlines()

    # # Loop through each line to find and extract 'E' and 'T' values
    # for line in lines:
    #     if 'E=' in line:
    #         energy = float(line.split('E=')[1].split()[0])
    #         energy_values.append(energy)
    #     if 'T=' in line:
    #         temperature = float(line.split('T=')[1].split()[0])
    #         temperature_values.append(temperature)
    

    # Extract every 10th value up to the 1960th iteration
    iteration_numbers = list(range(10+last_iter, min(last_traj + 1, len(energy_values))+last_iter, 10))

    # Truncate 'Energy' and 'Temperature' values to match the length of the iteration numbers
    energy_values = energy_values[:len(iteration_numbers)]
    temperature_values = temperature_values[:len(iteration_numbers)]

    # Create a DataFrame with 'Energy', 'Temperature', and iteration number
    data = {
        'Iteration': iteration_numbers,
        'Energy': energy_values,
        'Temperature': temperature_values
    }
    df = pd.DataFrame(data)

    df['Reference_Energy'] = Energy_ref
    df['Reference_Energy_per_atom'] = df['Reference_Energy'] / 108
    df['Energy_per_atom'] = df['Energy'] / total_atoms
    df['target_total_energy'] = df['Energy'] - df['Reference_Energy']
    df['target_total_energy_per_atom'] = df['Energy_per_atom'] - df['Reference_Energy_per_atom']
    df["target_energy"] = df["Energy"] - df["Reference_Energy"]
    # Save the DataFrame
    
    df.to_csv(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/processed_mds/energy_{exp_num}_data_additional.csv")
    return df

def oszicar_generation(file_path,xdatcar_path,oszicar_path,exp_num):

    Energy_ref = -765.3104674489797

    last_traj = None
    total_atoms = None

    # File path to the XDATCAR file
    xdatcar_filepath = file_path + xdatcar_path
    oszicar_filepath = file_path + oszicar_path


    # Read the file
    with open(xdatcar_filepath, 'r') as file:
        lines = file.readlines()
    

    atoms_line = lines[6].strip()
    atom_counts = [int(count) for count in atoms_line.split() if count.isdigit()]
    total_atoms = sum(atom_counts)
    print("Total number of atoms:", total_atoms)

    # Iterate through lines in reverse order to find the last occurrence of 'Direct configuration='
    for line in reversed(lines):
        if 'Direct configuration=' in line:
            last_traj = line.split('=')[-1].strip()
            break  # Stop after finding the last occurrence

    # Print the results
    if last_traj is not None:
        direct_configuration_number = float(last_traj)  # Convert to float or int as needed
        print("Last trajectory Number:", direct_configuration_number)
    else:
        print("No line with 'Direct configuration=' found in the file.")

    # Convert last_traj to an integer
    last_traj = int(last_traj)

    # Initialize empty lists to store 'E' and 'T' values
    energy_values = []
    temperature_values = []

    # Read the OSZICAR file
    with open(oszicar_filepath, 'r') as file:
        lines = file.readlines()

    # Loop through each line to find and extract 'E' and 'T' values
    for line in lines:
        if 'E=' in line:
            energy = float(line.split('E=')[1].split()[0])
            energy_values.append(energy)
        if 'T=' in line:
            temperature = float(line.split('T=')[1].split()[0])
            temperature_values.append(temperature)
    
    # with open(oszicar_filepath+"_1", 'r') as file:
    #     lines = file.readlines()

    # # Loop through each line to find and extract 'E' and 'T' values
    # for line in lines:
    #     if 'E=' in line:
    #         energy = float(line.split('E=')[1].split()[0])
    #         energy_values.append(energy)
    #     if 'T=' in line:
    #         temperature = float(line.split('T=')[1].split()[0])
    #         temperature_values.append(temperature)
    

    # Extract every 10th value up to the 1960th iteration
    iteration_numbers = list(range(1, min(last_traj + 1, len(energy_values)), 10))

    # Truncate 'Energy' and 'Temperature' values to match the length of the iteration numbers
    energy_values = energy_values[:len(iteration_numbers)]
    temperature_values = temperature_values[:len(iteration_numbers)]

    # Create a DataFrame with 'Energy', 'Temperature', and iteration number
    data = {
        'Iteration': iteration_numbers,
        'Energy': energy_values,
        'Temperature': temperature_values
    }
    df = pd.DataFrame(data)

    df['Reference_Energy'] = Energy_ref
    df['Reference_Energy_per_atom'] = df['Reference_Energy'] / 108
    df['Energy_per_atom'] = df['Energy'] / total_atoms
    df['target_total_energy'] = df['Energy'] - df['Reference_Energy']
    df['target_total_energy_per_atom'] = df['Energy_per_atom'] - df['Reference_Energy_per_atom']
    df["target_energy"] = df["Energy"] - df["Reference_Energy"]
    # Save the DataFrame
    
    df.to_csv(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/processed_mds/energy_{exp_num}_data.csv")
    return df

def randomize_train_and_test_task(n_tasks):
   random.seed(2)
   tasks = list(range(n_tasks))  # Example list of tasks [1, 2, 3, ..., 25]

   # Shuffle the tasks randomly
   random.shuffle(tasks)

   # Select the first 20 tasks for training
   train_tasks = tasks[:20]

   # Select the remaining 5 tasks for testing
   test_tasks = tasks[20:]

   return train_tasks, test_tasks

def sequential_train_and_test_task(n_tasks,ind_to_val):
    train_tasks = []
    test_tasks = []
    for val in range(1,n_tasks+1):
        result = list(ind_to_val.values()).index(f"{str(val)}")
        if val <20:
            train_tasks.append(result)
        else:
            test_tasks.append(result)
    return train_tasks,test_tasks

def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()


def distribution_plotting(energy_ground_truth, predictions, task, feature_extractor, scaler):
    # fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    epsilon = 1e-10
    # combined_min = min(np.min(energy_ground_truth), np.min(predictions))
    # combined_max = max(np.max(energy_ground_truth), np.max(predictions))
    # energy_ground_truth = ((energy_ground_truth - combined_min) / (combined_max - combined_min)) + epsilon
    # predictions = ((predictions - combined_min) / (combined_max - combined_min)) + epsilon

    # Plot ground truth in log scale
    # axes[0].hist(np.log(energy_ground_truth), label='ground truth',histtype='barstacked',alpha=0.75)
    # axes[0].set_title(f"Distribution of full training 'target energy per atom' ground truth for Task {task}")
    # axes[0].legend()

    # # Plot predictions in log scale
    # axes[1].hist(np.log(predictions), color='crimson', label='prediction')
    # axes[1].set_title(f"Distribution of full training 'target energy per atom' prediction for Task {task}")
    # axes[1].legend()

    plt.hist(np.round(energy_ground_truth,4),label='ground truth',histtype='barstacked',alpha=0.3)
    plt.hist(np.round(predictions,4),color='crimson',label='prediction',histtype='barstacked',alpha=0.3)
    plt.title(f"Distribution of full training 'target energy per atom' ground truth vs prediction of Task {task}",fontsize=30)
    plt.legend(fontsize=20)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/reconstruction_graphs/Distribution_Task{task}_graph_{feature_extractor}_{scaler}scaling.png")
    plt.clf()

def reconstruction_graph_plot(iterations_train,train_mean,y_train,iterations_test,mean,y_test,ind_to_val,title,task,scaler_flag,feature_extractor,training_cycles,lr):
    plt.figure(figsize=(20,7))
    plt.scatter(np.array(iterations_train)[-50:],train_mean[-50:],c='r',marker="+",label="Train prediction")
    plt.scatter(np.array(iterations_train)[-50:],y_train.reshape(-1,1)[-50:],c='b',marker="+",label="Train ground_truth")
    plt.scatter(np.array(iterations_test),mean.reshape(1,-1),c='r',label="prediction")
    plt.scatter(np.array(iterations_test),y_test.reshape(1,-1),c='b',label="ground_truth")
    plt.title(f"{title} from Task {task}",fontsize=20)
    plt.xlabel("Iterations")
    plt.ylabel("Target Energy")
    plt.legend(fontsize=10)
    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/reconstruction_graphs/Task{ind_to_val[task]}_Norm{scaler_flag}_{feature_extractor}_training{training_cycles}_lr{lr}_reconstruction.png")
    plt.clf()

def plot_training_loss(training_loss,task,scaler_flag,feature_extractor,training_cycles,lr):
    plt.plot(training_loss)
    plt.title(f"Training loss Task {task}")
    plt.xlabel("Training Cycles")
    plt.ylabel("Loss Optimization - MSE+MLL")
    plt.xticks(range(0,len(training_loss)))
    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/training_loss/Training_loss_Task{task}_Norm{scaler_flag}_{feature_extractor}_training{training_cycles}_lr{lr}_.png")
    plt.clf()


def decimal_formatter_mod(x, pos):
    """ Format ticks to 2 decimal places. """
    return f'{x/1e-3:.4f}'

# @title Modified plotting functions
def mod_reconstruction_graph_plot(
    iterations_train, train_mean, y_train,
    iterations_test, mean, y_test,
    ind_to_val, title, task,
    scaler_flag, feature_extractor,
    training_cycles, lr, augmentation
):
    plt.figure(figsize=(20, 10))

    # Scatter plots
    plt.scatter(
        np.array(iterations_train)[-50:], train_mean[-50:],
        c='darkred', marker='D', s=100, label="Train Prediction", edgecolor='k'
    )
    plt.scatter(
        np.array(iterations_train)[-50:], y_train.reshape(-1, 1)[-50:],
        c='darkblue', marker='^', s=100, label="Train Ground Truth", edgecolor='k'
    )
    plt.scatter(
        np.array(iterations_test), mean.reshape(1, -1),
        c='orange', marker='D', s=100, label="Prediction"
    )
    plt.scatter(
        np.array(iterations_test), y_test.reshape(1, -1),
        c='royalblue', marker='^', s=100, label="Ground Truth"
    )

    # Titles and labels with enhanced styles
    #plt.title(f"{title} from Task {task}", fontsize=22, fontweight='bold', color='darkgreen')
    plt.xlabel("Iterations (fs)", fontsize=16, fontweight='bold')
    plt.ylabel("Target energy difference per atom (meV)", fontsize=16, fontweight='bold')

    # Customize ticks
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Customize legend
    plt.legend(fontsize=14, loc='best', frameon=True, shadow=True, fancybox=True)

    # Add background color
    #plt.gca().set_facecolor('lightgray')
    plt.gca().set_facecolor('white')
    #plt.gca().xaxis.set_major_formatter(FuncFormatter(decimal_formatter_mod))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(decimal_formatter_mod))

    # Save figure (uncomment if needed)
    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/reconstruction_graphs/Task{ind_to_val[task]}_Norm{scaler_flag}_{feature_extractor}_aug{augmentation}_training{training_cycles}_lr{lr}_reconstruction.png")

    plt.clf()

def decimal_formatter(x, pos):
    """ Format ticks to 2 decimal places. """
    return f'{x:.3f}'

def color_gradient_difference_plot(energy_ground_truth, predictions, training_pred, training_ground_truth, task, feature_extractor, scaler, augmentation):
    # Calculate the difference
    pred_data =  np.concatenate((training_pred,predictions),axis=0)
    ground_truth_data = np.concatenate((training_ground_truth,energy_ground_truth),axis=0)

    difference = pred_data - ground_truth_data
    #difference_training = training_pred - training_ground_truth

    # Create a scatter plot
    plt.figure(figsize=(12, 8))

    # Scatter plot
    # scatter = plt.scatter(np.round(energy_ground_truth,6), np.round(predictions,6), c=difference, cmap='coolwarm', edgecolor='k', alpha=0.75)
    scatter = plt.scatter(ground_truth_data, pred_data, c=difference, s=60, cmap='coolwarm', edgecolor='k', alpha=0.75)


    # Add colorbar to show the gradient scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('Error (eV)', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=14, width=2)

    # Titles and labels with enhanced styles
    #plt.title(f"Color Gradient of Differences between Ground Truth and Predictions\nTask {task}", fontsize=24, fontweight='bold', color='darkblue')
    plt.xlabel("Computed energy difference per atom (eV)", fontsize=18, fontweight='bold')
    plt.ylabel("Predicted energy difference per atom (eV)", fontsize=18, fontweight='bold')

    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    # Set axis labels to two decimal places
    plt.gca().xaxis.set_major_formatter(FuncFormatter(decimal_formatter))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(decimal_formatter))

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/gradient_plot/Gradient_plot_Task{task}_{feature_extractor}_{scaler}_aug{augmentation}.png")
    plt.clf()

def color_gradient_difference_plot_modified(energy_ground_truth, predictions, training_pred, training_ground_truth, task, feature_extractor, scaler, augmentation):
    # Calculate the difference
    pred_data = np.concatenate((training_pred, predictions), axis=0).reshape(-1, 1)
    ground_truth_data = np.concatenate((training_ground_truth, energy_ground_truth), axis=0)

    difference = pred_data - ground_truth_data
    difference_training = training_pred - training_ground_truth
    difference_predictions = predictions - energy_ground_truth

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
 
    # Scatter plot for training data
    scatter_train = ax1.scatter(training_ground_truth, training_pred, c=difference[:len(training_pred)]/1e-3, s=60, cmap='viridis', edgecolor='k', alpha=0.75, marker='D')
    cbar_train = plt.colorbar(scatter_train, ax=ax1)
    cbar_train.set_label('Training Error (meV)', fontsize=14, fontweight='bold')
    cbar_train.ax.tick_params(labelsize=14, width=2)

    ax1.set_xlabel("Computed energy difference per atom (meV)", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Predicted energy difference per atom (meV)", fontsize=14, fontweight='bold')
    ax1.set_title("Training Prediction Error", fontsize=18, fontweight='bold')
    ax1.xaxis.set_major_formatter(FuncFormatter(decimal_formatter_mod))
    ax1.yaxis.set_major_formatter(FuncFormatter(decimal_formatter_mod))
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.tick_params(axis='both', which='major', labelsize=14, width=2)

    # Scatter plot for prediction data
    scatter_pred = ax2.scatter(energy_ground_truth, predictions, c=difference[len(training_pred):]/1e-3, s=60, cmap='coolwarm', edgecolor='k', alpha=0.75, marker='^')
    cbar_pred = plt.colorbar(scatter_pred, ax=ax2)
    cbar_pred.set_label('Testing Error (meV)', fontsize=14, fontweight='bold')
    cbar_pred.ax.tick_params(labelsize=14, width=2)

    ax2.set_xlabel("Computed energy difference per atom (meV)", fontsize=14, fontweight='bold')
    ax2.set_ylabel("Predicted energy difference per atom (meV)", fontsize=14, fontweight='bold')
    ax2.set_title("Testing Prediction Error", fontsize=18, fontweight='bold')
    ax2.xaxis.set_major_formatter(FuncFormatter(decimal_formatter_mod))
    ax2.yaxis.set_major_formatter(FuncFormatter(decimal_formatter_mod))
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.tick_params(axis='both', which='major', labelsize=14, width=2)

    plt.suptitle(f"Energy Difference Gradient for Task {task}", fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/gradient_plot/Gradient_plot_Task{task}_{feature_extractor}_{pois}_aug{augmentation}.png")
    plt.clf()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter

def color_gradient_difference_plot_3d(energy_ground_truth, predictions, training_pred, training_ground_truth, task, feature_extractor, scaler, augmentation):
    # Calculate the difference
    pred_data = np.concatenate((training_pred, predictions), axis=0).reshape(-1, 1)
    ground_truth_data = np.concatenate((training_ground_truth, energy_ground_truth), axis=0)

    difference = pred_data - ground_truth_data

    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for training data
    scatter_train = ax.scatter(training_ground_truth, training_pred, difference[:len(training_pred)]/1e-3, 
                               c=difference[:len(training_pred)]/1e-3, cmap='viridis', edgecolor='k', alpha=0.75, marker='D')
    
    # Scatter plot for prediction data
    scatter_pred = ax.scatter(energy_ground_truth, predictions, difference[len(training_pred):]/1e-3, 
                              c=difference[len(training_pred):]/1e-3, cmap='coolwarm', edgecolor='k', alpha=0.75, marker='^')

    # Adding color bar
    cbar = plt.colorbar(scatter_train, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Error (meV)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=14, width=2)

    # Set labels
    ax.set_xlabel("Computed energy difference per atom (meV)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Predicted energy difference per atom (meV)", fontsize=14, fontweight='bold')
    ax.set_zlabel("Error (meV)", fontsize=14, fontweight='bold')

    ax.set_title(f"3D Error Gradient for Task {task}", fontsize=18, fontweight='bold')

    ax.xaxis.set_major_formatter(FuncFormatter(decimal_formatter_mod))
    ax.yaxis.set_major_formatter(FuncFormatter(decimal_formatter_mod))
    ax.zaxis.set_major_formatter(FuncFormatter(decimal_formatter_mod))
    
    ax.tick_params(axis='both', which='major', labelsize=12, width=2)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/gradient_plot/Gradient_plot_3D_Task{task}_{feature_extractor}_{scaler}_aug{augmentation}.png")
    plt.show()

# Example call (assuming you have the required data and function dependencies):
# color_gradient_difference_plot_3d(energy_ground_truth, predictions, training_pred, training_ground_truth, task, feature_extractor, scaler, augmentation)

def mod_plot_training_loss(training_loss, task, scaler_flag, feature_extractor, training_cycles, lr, augmentation):
    plt.figure(figsize=(12, 8))

    # Plot the training loss with enhancements
    plt.plot(training_loss, color='darkblue', linestyle='-', marker='o', markersize=5, linewidth=2, label='Training Loss')

    # Add a title with larger, bold font
    plt.title(f"Training Loss - Task {task}", fontsize=24, fontweight='bold', color='darkblue')

    # Label x and y axes with larger, bold font
    plt.xlabel("Training Cycles", fontsize=18, fontweight='bold')
    plt.ylabel("Loss Optimization - MSE+MLL", fontsize=18, fontweight='bold')

    # Set x-axis ticks to show every training cycle, if appropriate
    plt.xticks(fontsize=14, fontweight='bold')

    # Set y-axis ticks with consistent styling
    plt.yticks(fontsize=14, fontweight='bold')

    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a legend
    plt.legend(fontsize=16, loc='best', frameon=True, shadow=True, fancybox=True)

    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, label='Reference Line')

    # Optionally save the plot
    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/training_loss/Training_loss_Task{task}_Norm{scaler_flag}_{feature_extractor}_aug{augmentation}_training{training_cycles}_lr{lr}_.png")

    plt.clf()
def plot_images_by_indices(images, indices, true_labels, iteration, task_id, exp_step, acquisition, cols=5):
    """Plot images corresponding to the specified indices."""
    num_images = len(images)
    rows = num_images // cols + (1 if num_images % cols else 0)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, 5))

    for idx in range(num_images):
      img_ax = plt.subplot(rows, cols, idx+1)
      img_ax.imshow(images[idx][0].cpu().numpy())
      img_ax.set_title(f"Iteration: {iteration[indices[idx]]}",fontsize=30)
      img_ax.axis('off')

    '''
    scatter_ax = axs[cols-1]
    scatter_ax.set_title('Energy at Optimal Iteration Plot - Active Learning Exploration')
    scatter_ax.set_xlabel('Iteration')
    scatter_ax.set_ylabel('Energy')
    for index in indices:
      scatter_ax.scatter(iteration[index], true_labels[index], color='red', s=100, marker=f"${int(iteration[index])}$")  # Increase marker size with `s`
    '''
    plt.tight_layout()
    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/data_for_plot_generations/AL_selected_Simulated_images/task{task_id+1}/selected_exp{exp_step}_with_{acquisition}acq_trajectory.png")
    plt.clf()


def single_model_training_and_validation(train_tasks,ind_to_val,image_mask,energy_to_ind,title,**kwargs):
    datadim = kwargs.get("datadim", 43264)
    embedim = 2
    training_cycles = kwargs.get("training_cycles", 150)
    lr = kwargs.get("lr", 1e-2)
    scaler_flag = kwargs.get("scaler",False)
    results = {int(ind_to_val[task]):[] for task in train_tasks}
    feature_extractor = kwargs.get("reptile","default_FE")
    augmentation = kwargs.get("augmentation",True)
    print(feature_extractor)
    model_sample = torch.nn.Sequential(
                                torch.nn.Linear(datadim,2048),
                                torch.nn.ReLU(),
                                # torch.nn.Linear(10240,2048),
                                # torch.nn.ReLU(),
                                torch.nn.Linear(2048,1024),
                                torch.nn.ReLU(),
                                torch.nn.Linear(1024, 256),
                                torch.nn.ReLU(),
                                *(list(fcFeatureExtractor(256,embedim).children())),)

    model_mod = kwargs.get("model_mod",model_sample)
    

    for task in train_tasks:
        print(f"Training Task {ind_to_val[task]}")
        (X_train, X_test, y_train, y_test, iterations_train, iterations_test) = train_test_split(
            image_mask[task][0], energy_to_ind[task]["target_total_energy_per_atom"], energy_to_ind[task]['Iteration'], test_size=0.15, shuffle=False, random_state=2)
        # X_train, y_train = image_mask[0][0], energy_to_ind[0]["target_energy"]
        X_train = np.squeeze(X_train,axis=1)
        X_test = np.squeeze(X_test,axis=1)

        s1,s2,s3 = X_train.shape
        X_train = X_train.reshape(-1,s2*s3)
        X_test = X_test.reshape(-1,s2*s3)


        y_train = np.array(y_train).reshape(1,-1)
        y_test = np.array(y_test).reshape(1,-1) 

        if scaler_flag:
            X_train = scaler_fun.fit_transform(X_train)
            X_test = scaler_fun.fit_transform(X_test)
        
        X , y = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
        y = y.reshape(1,-1)    

        dklgp = dklgpreg.dklGPR(datadim,embedim,hidden_dim=[1000,500,50],feature_extract=copy.deepcopy(model_mod),precision="single")#aoi.models.dklGPR(data_dim, embedim=2, precision="double")
        dklgp.fit(X, y, training_cycles=training_cycles, lr=lr)

        training_loss = list(dklgp.train_loss)
        
        with torch.no_grad():
            train_mean, train_var = dklgp.predict(X)
            mean, var = dklgp.predict(X_test)
        
        torch.save(dklgp,f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/model/Task{ind_to_val[task]}_Norm{scaler_flag}_{feature_extractor}_training{training_cycles}_lr{lr}_model.pt")
        clear_gpu_cache()

        # Reconstruction Graph
        mod_reconstruction_graph_plot(iterations_train,train_mean,y_train,iterations_test,mean,y_test,ind_to_val,title,task,scaler_flag,feature_extractor,training_cycles,lr,augmentation)

        # Gradient Plotting
        #color_gradient_difference_plot(y_test.reshape(-1,1),mean.reshape(-1,1),ind_to_val[task],feature_extractor,scaler_flag,augmentation)
        #distribution_plotting(y_test.reshape(-1,1),mean.reshape(-1,1),ind_to_val[task],feature_extractor,scaler_flag)
        color_gradient_difference_plot_modified(y_test.reshape(-1,1), mean, train_mean, y_train.reshape(-1,1), task, feature_extractor, scaler_flag,augmentation)
        #Training Loss plotting
        mod_plot_training_loss(training_loss,ind_to_val[task],scaler_flag,feature_extractor,training_cycles,lr,augmentation)

        results[int(ind_to_val[task])] = [iterations_train,iterations_test,train_mean,mean,y_train,y_test,feature_extractor,training_loss,scaler_flag,ind_to_val,title,training_cycles,lr]
    return results


def select_most_uncertain_samples(model, unlabeled_features, reference_energy, num_samples=1, error_threshold=0.05):
    # Switch the model to evaluation mode
    # Use the model to predict the unlabeled data
    with torch.no_grad():
      mean, variance = model.predict(unlabeled_features)

    # Compute uncertainties (variances) for each prediction
    uncertainties = torch.tensor(variance)

    # Select the indices of the `num_samples` most uncertain predictions
    _, most_uncertain_indices = torch.topk(uncertainties, num_samples)

    return most_uncertain_indices

def select_most_stable_energy_samples(model, unlabeled_features, reference_energy, num_samples=1, error_threshold=0.05):
   
    # Use the model to predict the unlabeled data
    with torch.no_grad():
        predicted_energies, variance = model.predict(unlabeled_features)
    
    # Compute the absolute errors from the reference energy
    errors = torch.abs(torch.tensor(predicted_energies, dtype=torch.float32).cpu() - reference_energy.cpu())
    
    # # Compute the percentage errors
    percentage_errors = errors / torch.abs(reference_energy).cpu()
    
    # # Filter out samples with errors greater than the threshold
    
    valid_indices = torch.where(percentage_errors < percentage_errors.mean())[1]  
    
    # if len(valid_indices) < num_samples:
    #     raise ValueError(f"Not enough samples with error within {error_threshold * 100}% of the reference energy.")
    # Select the indices of the `num_samples` most stable energy predictions
    #_, most_stable_indices = torch.topk(-percentage_errors[:,valid_indices], num_samples)
    
    sorted_tensor =  torch.argsort(-percentage_errors[:,valid_indices],dim=1)[0]

    most_stable_indices = sorted_tensor[-num_samples:]
    # Map back to original indices
    # most_stable_indices = valid_indices[most_stable_indices]
    
    #result = torch.tensor(most_stable_indices.item(),dtype=torch.int8)
    
    return most_stable_indices

# Simulate querying labels for the most uncertain samples
def query_labels(indices):
    # In practice, replace this with actual querying of labels
    # Here we simulate by selecting from the true labels
    new_labels = true_labels[list(indices.cpu())]  # Assuming `true_labels` is available
    return new_labels

global fig
fig = None
def plot_images_by_indices(images, indices, true_labels, iteration, task_id, exp_step, acquisition, cols=5):
    """Plot images corresponding to the specified indices."""
    num_images = len(images)
    rows = num_images // cols + (1 if num_images % cols else 0)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, 5))

    for idx in range(num_images):
      img_ax = plt.subplot(rows, cols, idx+1)
      img_ax.imshow(images[idx][0].cpu().numpy())
      img_ax.set_title(f"Iteration: {iteration[indices[idx]]}",fontsize=30)
      img_ax.axis('off')

    '''
    scatter_ax = axs[cols-1]
    scatter_ax.set_title('Energy at Optimal Iteration Plot - Active Learning Exploration')
    scatter_ax.set_xlabel('Iteration')
    scatter_ax.set_ylabel('Energy')
    for index in indices:
      scatter_ax.scatter(iteration[index], true_labels[index], color='red', s=100, marker=f"${int(iteration[index])}$")  # Increase marker size with `s`
    '''
    plt.tight_layout()
    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/active_learning/task{task_id+1}/selected_exp{exp_step}_with_{acquisition}acq_trajectory.png")
    plt.clf()

def plot_selected_points(iterations_test,test_y,selected_iter,selected_energy,acquisition,task,num_iterations):
    plt.figure(figsize=(10,7))
    plt.scatter(iterations_test.cpu(),test_y.cpu(),marker="+",label='Ground truth')

    plt.scatter(selected_iter,selected_energy,s=30,c='crimson',marker=f'*',label='Prediction')

    plt.title(f"Selected Points through active learning, with {acquisition} acquisition function",fontsize=20)
    plt.xlabel("Iteration")
    plt.ylabel("Energy (eV)")
    plt.tight_layout()
    plt.legend(fontsize=10)
    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/active_learning/task{task+1}/final_actively_selected_{num_iterations}_points_with_{acquisition}_acq.png")
    plt.clf()

# @title modified graph utility functions for AL plots
def mod_plot_selected_points(iterations_test, test_y, selected_iter, selected_energy, acquisition, task, num_iterations):
    plt.figure(figsize=(12, 8))

    # Scatter plot for ground truth
    plt.scatter(
        iterations_test.cpu(),
        test_y.cpu(),
        marker="o",
        color='royalblue',
        s=100,  # Marker size
        edgecolor='k',
        label='Ground Truth'
    )

    # Scatter plot for predictions
    plt.scatter(
        selected_iter,
        selected_energy,
        s=150,  # Larger marker size
        color='crimson',
        marker='D',
        edgecolor='k',
        label='Prediction'
    )

    # Add titles and labels with enhanced styles
    # plt.title(
    #     f"Selected Points through Active Learning with {acquisition} Acquisition Function",
    #     fontsize=24,
    #     fontweight='bold',
    #     color='darkblue'
    # )
    plt.xlabel(
        "Iterations (fs)",
        fontsize=18,
        fontweight='bold'
    )
    plt.ylabel(
        "Target Energy Difference per Atom (eV)",
        fontsize=18,
        fontweight='bold'
    )

    # Adjust x and y axis ticks
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    # Add a grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a legend with enhanced styles
    plt.legend(fontsize=16, loc='best', frameon=True, shadow=True, fancybox=True)

    # Adjust layout to ensure everything fits well
    plt.tight_layout()

    # Optionally save the plot
    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/active_learning/task{task+1}/final_actively_selected_{num_iterations}_points_with_{acquisition}_acq.png")
    plt.clf()

def mod_plot_al_training_loss(training_loss, task, num_iterations, acquisition):
    plt.figure(figsize=(12, 8))

    # Plot the training loss with enhancements
    plt.plot(
        training_loss,
        color='darkblue',
        linestyle='-',
        marker='o',
        markersize=8,
        linewidth=2,
        label='Training Loss'
    )

    # Add titles and labels with enhanced styles
    plt.title(
        f"Active Learning Training Loss for Task {task} - {num_iterations} Exploration Steps",
        fontsize=24,
        fontweight='bold',
        color='darkblue'
    )
    plt.xlabel(
        "Exploration Steps",
        fontsize=18,
        fontweight='bold'
    )
    plt.ylabel(
        "Active Learning Loss",
        fontsize=18,
        fontweight='bold'
    )

    # Adjust x-axis ticks
    plt.xticks(
        fontsize=14,
        fontweight='bold'
    )

    # Adjust y-axis ticks
    plt.yticks(
        fontsize=14,
        fontweight='bold'
    )

    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add a legend
    plt.legend(fontsize=16, loc='best', frameon=True, shadow=True, fancybox=True)

    # Optionally save the plot
    # plt.savefig(f"result/active_learning/{acquisition}/Training_loss_task{task+1}.png")
    plt.savefig(f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/active_learning/task{task+1}/Training_loss_{acquisition}_acq.png")
    plt.tight_layout()
    
    plt.clf()


def active_learning_single_tasks(task,image_mask,energy_to_ind,exploration_steps,acquisition):

    (X_train, X_test, y_train, y_test, iterations_train, iterations_test) = train_test_split(
                image_mask[task][0], energy_to_ind[task]["target_total_energy_per_atom"], energy_to_ind[task]['Iteration'], test_size=0.9, shuffle=False, random_state=2)

    datadim = 43264
    embedim = 2 
    num_iterations = exploration_steps
    num_samples_per_iteration = 1

    training_cycles = 20
    lr = 1e-1

    unlabeled_img = torch.tensor(copy.deepcopy(X_test),dtype=torch.float32)

    X_train = np.squeeze(X_train,axis=1)
    X_test = np.squeeze(X_test,axis=1)

    s1,s2,s3 = X_train.shape
    X_train = X_train.reshape(-1,s2*s3)
    X_test = X_test.reshape(-1,s2*s3)

    y_train = np.array(y_train).reshape(1,-1)
    y_test = np.array(y_test).reshape(1,-1)

    true_labels = energy_to_ind[task]['target_total_energy_per_atom'] 

    train_x = torch.tensor(X_train, dtype = torch.float32)
    train_y = torch.tensor(np.array(y_train), dtype = torch.float32)
    test_x = torch.tensor(X_test, dtype = torch.float32)
    test_y = torch.tensor(np.array(y_test), dtype = torch.float32)

    target_labels = torch.tensor(np.array(energy_to_ind[task]['target_total_energy_per_atom'][iterations_test.index]),dtype=torch.float32)

    iterations_train = torch.tensor(np.array(iterations_train),dtype=torch.float32)
    iterations_test = torch.tensor(np.array(iterations_test),dtype=torch.float32)

    print(f"True labels shape: {true_labels.shape}")
    print(f"True labels indices: {true_labels.index}")

    acquisiton_function = {
        "uncertainty": select_most_uncertain_samples,
        "stability" : select_most_stable_energy_samples,
    }

    selected_iter = []
    selected_energy = []
    training_loss = []
    model_mod = torch.nn.Sequential(
                                    torch.nn.Linear(datadim,2048),
                                    torch.nn.ReLU(),
                                    # torch.nn.Linear(10240,2048),
                                    # torch.nn.ReLU(),
                                    torch.nn.Linear(2048,1024),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(1024, 256),
                                    torch.nn.ReLU(),
                                    *(list(fcFeatureExtractor(256,embedim).children())),)

    dklgp = dklgpreg.dklGPR(datadim,embedim,hidden_dim=[1000,500,50],feature_extract=copy.deepcopy(model_mod),precision="single")#aoi.models.dklGPR(data_dim, embedim=2, precision="double")

    for iteration in range(num_iterations):

        print(f"Iteration {iteration+1}/{num_iterations}")
        
        dklgp.fit(train_x, train_y, training_cycles=training_cycles, lr=lr)

        training_loss.append(mean(list(dklgp.train_loss)))

        #most_uncertain_index = select_most_uncertain_samples(dklgp, test_x, num_samples=num_samples_per_iteration)
        most_uncertain_index = acquisiton_function[acquisition](dklgp, test_x, test_y, num_samples=num_samples_per_iteration, error_threshold=0.05)
        print(f"Most uncertain indices: {most_uncertain_index}")

        unlabelled_images = [unlabeled_img[i] for i in most_uncertain_index]

        selected_iter.append(iterations_test[most_uncertain_index.cpu().numpy()].cpu().numpy())
        selected_energy.append(target_labels[most_uncertain_index.cpu().numpy()].cpu().numpy())

        plot_images_by_indices(unlabelled_images, most_uncertain_index.cpu().numpy(), target_labels.cpu().numpy(), iterations_test.cpu().numpy(), task, iteration, acquisition, cols=num_samples_per_iteration)

        train_x = torch.cat((train_x, test_x[most_uncertain_index.item()][None]), 0)
        train_y = torch.cat((train_y, test_y[:,most_uncertain_index.item()].reshape(1,-1)), 1)
        
        test_x = torch.cat((test_x[:most_uncertain_index.item()], test_x[most_uncertain_index.item()+1:]), 0)
        test_y = torch.cat((test_y[:,:most_uncertain_index.item()].reshape(1,-1), test_y[:,most_uncertain_index.item()+1:].reshape(1,-1)), 1)
        
        unlabeled_img = torch.cat((unlabeled_img[:most_uncertain_index.item()], unlabeled_img[most_uncertain_index.item()+1:]), 0)
        target_labels = torch.cat((target_labels[:most_uncertain_index.item()], target_labels[most_uncertain_index.item()+1:]), 0)
        
        iterations_test = torch.cat((iterations_test[:most_uncertain_index.item()], iterations_test[most_uncertain_index.item()+1:]), 0)


        print(f"New data shape: {train_x.shape}")
        print(f"New data label: {train_y.shape}")
        torch.save(dklgp,f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/results/single_tasks/active_learning/task{task+1}/AL_exp{iteration}_with_{acquisition}_acq_model.pt")
    
    #plot_selected_points(iterations_test,test_y,selected_iter,selected_energy,acquisition,task,num_iterations)
    mod_plot_selected_points(iterations_test, test_y, selected_iter, selected_energy, acquisition, task, num_iterations)
    
    #Plot Training Loss
    mod_plot_al_training_loss(training_loss,task,num_iterations,acquisition)


    return selected_energy, selected_iter, training_loss, iterations_test,test_y, acquisition,task, num_iterations