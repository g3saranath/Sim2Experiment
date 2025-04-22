import os, re
global cwd_path
cwd_path = os.getcwd()
from processing import *
import sys
import time
from AtomicImageSimulator.main import *
# Add the parent directory to sys.path
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import joblib
import glob
import matplotlib.pyplot as plt
import pickle
import atomai as aoi
import argparse
np.random.seed(0)
from processing import *
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import torch
import warnings
# Ignore DeprecationWarning
warnings.filterwarnings("ignore")
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import imageio


def data_load():
    
    directory = f"{cwd_path}/processed_mds/image_energy_data/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory {directory} created ")
        ind_to_val, image_list, mask_list, label_list, coordinates =  xdatcar_processor_concat(f"{cwd_path}/to_share_HP_defects/", n_sims=25)
        with open(f"{cwd_path}/processed_mds/image_energy_data/ind_to_val.pkl", "wb") as f:
            pickle.dump(ind_to_val, f)
        with open(f"{cwd_path}/processed_mds/image_energy_data/image_list.pkl", "wb") as f:
            pickle.dump(image_list, f)
        with open(f"{cwd_path}/processed_mds/image_energy_data/mask_list.pkl", "wb") as f:
            pickle.dump(mask_list, f)
        with open(f"{cwd_path}/processed_mds/image_energy_data/label_list.pkl", "wb") as f:
            pickle.dump(label_list, f)
        with open(f"{cwd_path}/processed_mds/image_energy_data/coordinates.pkl", "wb") as f:
            pickle.dump(coordinates, f)
    else:
        with open(f"{cwd_path}/processed_mds/image_energy_data/ind_to_val.pkl", "rb") as f:
            ind_to_val = pickle.load(f)
        with open(f"{cwd_path}/processed_mds/image_energy_data/image_list.pkl", "rb") as f:
            image_list = pickle.load(f)
        with open(f"{cwd_path}/processed_mds/image_energy_data/mask_list.pkl", "rb") as f:
            mask_list = pickle.load(f)
        with open(f"{cwd_path}/processed_mds/image_energy_data/label_list.pkl", "rb") as f:
            label_list = pickle.load(f)
        with open(f"{cwd_path}/processed_mds/image_energy_data/coordinates.pkl", "rb") as f:
            coordinates = pickle.load(f)
        print(f"Data loaded from {directory}")

    return ind_to_val, image_list, mask_list, label_list, coordinates

def apply_pca(features, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features

def energy_data(ind_to_val):
    val = 0
    energy_to_ind = {val:None for val in range(25)}
    new_energy_ind = {val:None for val in range(25)}
    directory =  f"{cwd_path}/to_share_HP_defects/"
    for filename in os.listdir(directory):
        if "." in filename:
            continue
        task = ind_to_val[val]
        file_name = directory + filename + "/"
        print(val)
        # filepath = f"/lustre/saranath/Techcon24/AIMD/Sim2Experiment/df_{task}/"
        xdatcar_filepath = f"XDATCAR_df_S_{filename.split('_')[1]}"
        oszicar_filepath = f"OSZICAR_df_S_{filename.split('_')[1]}"
        energy_to_ind[val] = oszicar_generation(file_name,xdatcar_filepath,oszicar_filepath,filename.split("_")[1])
        last_iter = energy_to_ind[val]["Iteration"].values[-1]
        last_idx = len(energy_to_ind[val]["Iteration"])

        xdatcar_filepath = f"XDATCAR_df_S_{filename.split('_')[1]}_1"
        oszicar_filepath = f"OSZICAR_df_S_{filename.split('_')[1]}_1"
        new_energy_ind[val] = oszicar_generation(file_name,xdatcar_filepath,oszicar_filepath,filename.split("_")[1])
        new_energy_ind[val]["Iteration"] = new_energy_ind[val]["Iteration"]+last_iter #oszicar_generation(file_name,xdatcar_filepath,oszicar_filepath,filename.split("_")[1])
        energy_to_ind[val] = pd.concat([energy_to_ind[val],new_energy_ind[val]])
        
        # energy_to_ind[val] = pd.concat([energy_to_ind[val],oszicar_generation_additional(file_name,xdatcar_filepath,oszicar_filepath,filename.split("_")[1],last_iter)])
        energy_to_ind[val]["target_total_energy_per_atom"] = energy_to_ind[val]["target_total_energy_per_atom"] - energy_to_ind[val]["target_total_energy_per_atom"].mean()
        # energy_to_ind[val]["Iteration"][last_idx:] = energy_to_ind[val]["Iteration"][last_idx:] + last_iter
        val += 1
    return energy_to_ind


def fit_image_dimensions(image_list, task):
    # Find the maximum dimensions
    max_height = max(row.shape[0] for row in image_list[task])
    max_width = max(row.shape[1] for row in image_list[task])

    # Pad each tensor to the maximum dimensions
    padded_tensors = [torch.nn.functional.pad(torch.tensor(row), (0, max_width - row.shape[1], 0, max_height - row.shape[0])) for row in image_list[task]]

    # Stack the padded tensors
    image_tensor = torch.stack(padded_tensors)
    return image_tensor

def extract_features_vgg16(images):
    # Load pre-trained VGG16 model without the top layer (for feature extraction)
    device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = models.vgg16(pretrained=True).to(device)
    base_model.classifier = nn.Identity()  # Remove the fully connected layers
    base_model.eval()
    
    # Preprocess and extract features for each image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 273)),  # Adjust resize to match image size while keeping aspect ratio close
        transforms.CenterCrop((224, 224)),  # Crop to required size for VGG16
        transforms.Lambda(lambda x: x.convert("RGB")),  # Convert single-channel to three-channel
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for three channels
    ])
    
    processed_images = torch.stack([transform(image) for image in images]).to(device)
    with torch.no_grad():
        features = base_model(processed_images)
    
    # Flatten the features
    #features_flat = features.view(features.size(0), -1)
    return features

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def extract_latent_features(image_list, task, y, epochs=50, batch_size=32, lr=0.001, plot_latentspace=False, plot_regression=False):
    # Extract features using VGG16
    image_tensor = fit_image_dimensions(image_list,task)
    features = extract_features_vgg16(image_tensor)
    input_dim = features.shape[1]
    autoencoder = Autoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    features_tensor = torch.tensor(features, dtype=torch.float32)
    dataset = TensorDataset(features_tensor, features_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train autoencoder
    for epoch in range(epochs):
        for data in dataloader:
            inputs, _ = data
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

    # Get latent representation
    with torch.no_grad():
        latent_features = autoencoder.encoder(features_tensor).numpy()
    if plot_latentspace == True:
        plt.figure(figsize=(10, 6))
        sc = plt.scatter(latent_features[:, 0], latent_features[:, 1], c=y, cmap='viridis', alpha=0.5)
        plt.colorbar(sc, label='Energy')
        plt.xlabel('Latent Feature 1')
        plt.ylabel('Latent Feature 2')
        plt.title('Latent Features vs Energy')
        plt.show(block=True)
    return latent_features

def training_tree_regressor(image_list, task, energy_to_ind, plot_regression=True):
    latent_features = extract_latent_features(image_list, task, energy_to_ind[task]["target_total_energy_per_atom"])
    energy_data = energy_to_ind[task]["target_total_energy_per_atom"]
    iterations = energy_to_ind[task]["Iteration"]
    # Split dataset into training and testing
    X_train, X_test, y_train, y_test, iterations_train, iterations_test = train_test_split(latent_features, energy_data, iterations, test_size=0.2, shuffle=True, random_state=42)

    # Build a Regression Pipeline with Scaling and Model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    # Hyperparameter Tuning
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=2)
    start = time.time()
    grid_search.fit(X_train, y_train)
    end = time.time()
    print(f"Training Time: {end - start} seconds")
    prediction = grid_search.predict(X_test)

    # Evaluate Model
    best_model = grid_search.best_estimator_

    # Variance Calculation with Random Forest

    random_forest = RandomForestRegressor(
        n_estimators=grid_search.best_params_['regressor__n_estimators'],
        max_depth=grid_search.best_params_['regressor__max_depth'],
        min_samples_split=grid_search.best_params_['regressor__min_samples_split'],
        random_state=42,
        bootstrap=True
    )

    random_forest.fit(X_train, y_train)
    
    # Obtain predictions from each tree in the forest
    all_train_predictions = np.array([tree.predict(X_train) for tree in random_forest.estimators_])
    all_tree_predictions = np.array([tree.predict(X_test) for tree in random_forest.estimators_])

    # Mean predictions across trees
    train_predictions = np.mean(all_train_predictions, axis=0)
    test_predictions = np.mean(all_tree_predictions, axis=0)

    # Calculate variance (or standard deviation) across predictions
    train_prediction_variance = np.var(all_train_predictions, axis=0)
    prediction_variance = np.var(all_tree_predictions, axis=0)
    prediction_std_dev = np.sqrt(prediction_variance)

    print("Best Model:", best_model)
    y_mean = train_predictions #best_model.predict(X_train)
    y_pred = test_predictions #best_model.predict(X_test)

    # Calculate Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R2 Score: {r2}')
    if plot_regression == True:
        # Optional: Plotting Predicted vs True Values
        plt.scatter(iterations_train,y_train, c='b', marker='*', alpha=0.5, label="Train - True")
        plt.scatter(iterations_train, y_mean, c="r", marker='*', alpha=0.5, label="Train - Predicted")
        plt.scatter(iterations_test,y_test, c='b', marker='D', alpha=0.5, label="Test - True")
        plt.scatter(iterations_test, y_pred, c="r", marker='D', alpha=0.5, label="Test - Predicted")
        plt.xlabel('Values')
        plt.ylabel('True vs Predicted Values')
        plt.title('True vs Predicted Energy Values for Task ' + str(task))
        plt.legend()
        if not os.path.exists(f"{cwd_path}/results/tree_regressor/"):
            os.makedirs(f"{cwd_path}/results/tree_regressor/")
        plt.savefig(f"{cwd_path}/results/tree_regressor/true_vs_predicted_task{task}.png")
        plt.clf()
    return grid_search, train_predictions, test_predictions, train_prediction_variance, prediction_variance

def select_most_uncertain_samples(predicted_variance, num_samples=1):
    # Switch the model to evaluation mode
    # Use the model to predict the unlabeled data
    # with torch.no_grad():
    #   mean, variance = model.predict(unlabeled_features)

    # Compute uncertainties (variances) for each prediction
    uncertainties = torch.tensor(predicted_variance)

    # Select the indices of the `num_samples` most uncertain predictions
    _, most_uncertain_indices = torch.topk(uncertainties, num_samples)

    return most_uncertain_indices

def plot_images_by_indices(images, indices, true_labels, iteration, task_id, exp_step, acquisition, cols=5):
    """Plot images corresponding to the specified indices."""
    num_images = len(images)
    rows = num_images // cols + (1 if num_images % cols else 0)

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, 5))

    for idx in range(num_images):
      img_ax = plt.subplot(rows, cols, idx+1)
      img_ax.imshow(images[idx])
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
    if not os.path.exists(f"{cwd_path}/results/tree_regressor/active_learning/AL_selected_Simulated_images/task{task_id+1}/"):
        os.makedirs(f"{cwd_path}/results/tree_regressor/active_learning/AL_selected_Simulated_images/task{task_id+1}/")
    plt.savefig(f"{cwd_path}/results/tree_regressor/active_learning/AL_selected_Simulated_images/task{task_id+1}/selected_exp{exp_step}_with_{acquisition}acq_trajectory.png")
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
    if not os.path.exists(f"{cwd_path}/results/tree_regressor/single_tasks/active_learning/task{task+1}/"):
        os.makedirs(f"{cwd_path}/results/tree_regressor/single_tasks/active_learning/task{task+1}/")
    plt.savefig(f"{cwd_path}/results/tree_regressor/single_tasks/active_learning/task{task+1}/final_actively_selected_{num_iterations}_points_with_{acquisition}_acq.png")
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
        f"Active Learning Loss for Task {task} - {num_iterations} Exploration Steps",
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
    plt.savefig(f"{cwd_path}/results/tree_regressor/single_tasks/active_learning/task{task+1}/Training_loss_{acquisition}_acq.png")
    plt.tight_layout()
    
    plt.clf()

def mod_plot_selected_points_gif(iterations_test, test_y, selected_iter, selected_energy, acquisition, task, num_iterations):
    images = []
    for i in range(len(selected_iter)):
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

        # Scatter plot for predictions up to the current point
        plt.scatter(
            selected_iter[:i+1],
            selected_energy[:i+1],
            s=200,  # Larger marker size
            color='crimson',
            marker='D',
            edgecolor='k',
            label='Prediction'
        )

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

        plt.xticks(fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=16, loc='best', frameon=True, shadow=True, fancybox=True)
        plt.tight_layout()
        if not os.path.exists(f"{cwd_path}/results/tree_regressor/single_tasks/active_learning/task{task+1}/stepwise/"):
            os.makedirs(f"{cwd_path}/results/tree_regressor/single_tasks/active_learning/task{task+1}/stepwise/")

        # Save the plot as an image and add to the list
        img_path = f"{cwd_path}/results/tree_regressor/single_tasks/active_learning/task{task+1}/stepwise/temp_plot_{i}.png"
        plt.savefig(img_path)
        images.append(imageio.imread(img_path))
        plt.clf()

    # Save images as a GIF
    gif_path = f"{cwd_path}/results/tree_regressor/single_tasks/active_learning/task{task+1}/stepwise/final_actively_selected_{num_iterations}_points_with_{acquisition}_acq.gif"
    imageio.mimsave(gif_path, images, duration=0.75)

    # # Clean up temporary images
    # for img_path in images:
    #     os.remove(img_path)

def tree_based_active_learning(grid_search, task,image_mask,energy_to_ind,exploration_steps,acquisition):
    latent_features = extract_latent_features(image_mask, task, energy_to_ind[task]["target_total_energy_per_atom"])
    (X_train, X_test, y_train, y_test, iterations_train, iterations_test) = train_test_split(
                latent_features, energy_to_ind[task]["target_total_energy_per_atom"], energy_to_ind[task]['Iteration'], test_size=0.9, shuffle=True, random_state=2)

    num_iterations = exploration_steps
    num_samples_per_iteration = 1

    training_cycles = 20
    lr = 1e-1
    energy_index = energy_to_ind[task]['target_total_energy_per_atom'][iterations_test.index].index
    
    images = np.array(image_mask[task], dtype=np.float32)
    
    unlabeled_img = images[energy_index]
    # X_train = np.squeeze(X_train,axis=1)
    # X_test = np.squeeze(X_test,axis=1)

    # s1,s2,s3 = X_train.shape
    # X_train = X_train.reshape(-1,s2*s3)
    # X_test = X_test.reshape(-1,s2*s3)

    # y_train = np.array(y_train).reshape(1,-1)
    # y_test = np.array(y_test).reshape(1,-1)

    # true_labels = energy_to_ind[task]['target_total_energy_per_atom'] 
    # print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    train_x = torch.tensor(X_train, dtype = torch.float32)
    train_y = torch.tensor(np.array(y_train), dtype = torch.float32)
    test_x = torch.tensor(X_test, dtype = torch.float32)
    test_y = torch.tensor(np.array(y_test), dtype = torch.float32)

    target_labels = torch.tensor(np.array(energy_to_ind[task]['target_total_energy_per_atom'][iterations_test.index]),dtype=torch.float32)

    iterations_train = torch.tensor(np.array(iterations_train),dtype=torch.float32)
    iterations_test = torch.tensor(np.array(iterations_test),dtype=torch.float32)


    acquisiton_function = {
        "uncertainty": select_most_uncertain_samples,
        "stability" : select_most_stable_energy_samples,
    }

    selected_iter = []
    selected_energy = []
    training_loss = []
    # model_mod = torch.nn.Sequential(
    #                                 torch.nn.Linear(datadim,2048),
    #                                 torch.nn.ReLU(),
    #                                 # torch.nn.Linear(10240,2048),
    #                                 # torch.nn.ReLU(),
    #                                 torch.nn.Linear(2048,1024),
    #                                 torch.nn.ReLU(),
    #                                 torch.nn.Linear(1024, 256),
    #                                 torch.nn.ReLU(),
    #                                 *(list(fcFeatureExtractor(256,embedim).children())),)

    # dklgp = dklgpreg.dklGPR(datadim,embedim,hidden_dim=[1000,500,50],feature_extract=copy.deepcopy(model_mod),precision="single")#aoi.models.dklGPR(data_dim, embedim=2, precision="double")
    model = RandomForestRegressor(
        n_estimators=grid_search.best_params_['regressor__n_estimators'],
        max_depth=grid_search.best_params_['regressor__max_depth'],
        min_samples_split=grid_search.best_params_['regressor__min_samples_split'],
        random_state=42,
        bootstrap=True
    )
    for iteration in range(num_iterations):
        
        print(f"Iteration {iteration+1}/{num_iterations}")
        
        model.fit(train_x, train_y)
        # Obtain predictions from each tree in the forest
        all_train_predictions = np.array([tree.predict(train_x) for tree in model.estimators_])
        all_tree_predictions = np.array([tree.predict(test_x) for tree in model.estimators_])

        # Mean predictions across trees
        train_predictions = np.mean(all_train_predictions, axis=0)
        test_predictions = np.mean(all_tree_predictions, axis=0)

        # Calculate variance (or standard deviation) across predictions
        train_prediction_variance = np.var(all_train_predictions, axis=0)
        prediction_variance = np.var(all_tree_predictions, axis=0)
        
        y_mean = train_predictions #best_model.predict(X_train)
        y_pred = test_predictions #best_model.predict(X_test)
        
        mse = mean_squared_error(test_y, y_pred)
        print(f'Mean Squared Error of Iteration {iteration+1}: {mse}')
        training_loss.append(mse)

        # most_uncertain_index = select_most_uncertain_samples(dklgp, test_x, num_samples=num_samples_per_iteration)
        most_uncertain_index = acquisiton_function[acquisition](prediction_variance, num_samples=num_samples_per_iteration)
        # print(f"Most uncertain indices: {most_uncertain_index}")

        unlabelled_images = [unlabeled_img[i] for i in most_uncertain_index]

        selected_iter.append(iterations_test[most_uncertain_index.cpu().numpy()].cpu().numpy())
        selected_energy.append(test_y[most_uncertain_index.cpu().numpy()].cpu().numpy())

        plot_images_by_indices(unlabelled_images, most_uncertain_index.cpu().numpy(), target_labels.cpu().numpy(), iterations_test.cpu().numpy(), task, iteration, acquisition, cols=num_samples_per_iteration)
        if iteration == num_iterations-1:
            break
        train_x = torch.cat((train_x, test_x[most_uncertain_index.item()][None]), 0)
        train_y = torch.cat((train_y, test_y[most_uncertain_index.item()].unsqueeze(0)), 0)

        
        test_x = torch.cat((test_x[:most_uncertain_index.item()], test_x[most_uncertain_index.item()+1:]), 0)
        test_y = torch.cat((test_y[:most_uncertain_index.item()], test_y[most_uncertain_index.item()+1:]), 0)
        
        unlabeled_img = torch.tensor(unlabeled_img)

        unlabeled_img = torch.cat((unlabeled_img[:most_uncertain_index.item()], unlabeled_img[most_uncertain_index.item()+1:]), 0)
        target_labels = torch.cat((target_labels[:most_uncertain_index.item()], target_labels[most_uncertain_index.item()+1:]), 0)
        
        iterations_test = torch.cat((iterations_test[:most_uncertain_index.item()], iterations_test[most_uncertain_index.item()+1:]), 0)


        # print(f"New data shape: {train_x.shape}")
        # print(f"New data label: {train_y.shape}")
        # torch.save(dklgp,f"results/single_tasks/active_learning/task{task+1}/AL_exp{iteration}_with_{acquisition}_acq_model.pt")
    
    #plot_selected_points(iterations_test,test_y,selected_iter,selected_energy,acquisition,task,num_iterations)
    mod_plot_selected_points(iterations_test, test_y, selected_iter, selected_energy, acquisition, task, num_iterations)
    mod_plot_selected_points_gif(iterations_test, test_y, selected_iter, selected_energy, acquisition, task, num_iterations)
    
    #Plot Training Loss
    mod_plot_al_training_loss(training_loss,task,num_iterations,acquisition)

    plt.figure(figsize=(10, 6))
    # plt.scatter(iterations_train, y_train, c='r', marker='*', alpha=0.5, label="Train - True")
    # plt.scatter(iterations_train, y_mean, c="b", marker='*', alpha=0.5, label="Train - Predicted")
    # plt.scatter(iterations_train, train_prediction_variance, color='green',  marker='*', alpha=0.5, label="Train Variance")
    plt.scatter(iterations_test, test_y, c='r', marker='D', alpha=0.5, label="Test - True")
    plt.scatter(iterations_test, y_pred, c="b", marker='D', alpha=0.5, label="Test - Predicted")
    # plt.scatter(iterations_test, prediction_variance, color='green', marker="D",alpha=0.5, label="Test Variance")
    plt.xlabel('Iterations')
    plt.ylabel('True vs Predicted Values')
    plt.title('True vs Predicted Energy Values for Task ' + str(task+1) + " after {} iterations".format(num_iterations))
    plt.legend()
    plt.savefig(f"{cwd_path}/results/tree_regressor/active_learning/regression_true_vs_predicted_task{task+1}.png")
    plt.clf()


    return selected_energy, selected_iter, training_loss, iterations_test,test_y, acquisition,task, num_iterations

def cropping_image(image_list, mask_list):
    #crop out the empty spaces
    cropped_image_list = []
    cropped_mask_list = []

    for i in range(0,len(image_list)):
        local_cropped_imglist = []
        local_cropped_masklist = []
        for j in range(0,len(image_list[i])):
            local_cropped_imglist.append(image_list[i][j][:208,0:208])
            local_cropped_masklist.append(mask_list[i][j][:208,0:208])

        cropped_image_list.append(np.array(local_cropped_imglist))
        cropped_mask_list.append(np.array(local_cropped_masklist))

    print("Number of tasks : ",len(cropped_image_list))
    cropped_image_list = np.array(cropped_image_list, dtype=object)
    cropped_mask_list = np.array(cropped_mask_list, dtype=object)
    
    return np.array(cropped_image_list), np.array(cropped_mask_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="tree_regressor", help="Method to use for training")
    args = parser.parse_args()
    print(f"Method: {args.method}")
    ind_to_val, image_list, mask_list, label_list, coordinates = data_load()

    cropped_image_list, cropped_mask_list = cropping_image(image_list, mask_list)
    energy_to_ind = energy_data(ind_to_val)

    for task in range(25):
        print(cropped_image_list[task].shape)
        if args.method == "tree_regressor":
            if not os.path.exists(f"{cwd_path}/results/tree_regressor/single_tasks/model/task{task+1}/"):
                grid_search, train_predictions, test_predictions, train_prediction_variance, prediction_variance = training_tree_regressor(cropped_image_list, task, energy_to_ind, plot_regression=True)
                os.makedirs(f"{cwd_path}/results/tree_regressor/single_tasks/model/task{task+1}/")
                joblib_file = f"{cwd_path}/results/tree_regressor/single_tasks/model/task{task+1}/grid_search_model_{task+1}.pkl"
                joblib.dump(grid_search, joblib_file)
            else:
                joblib_file = f"{cwd_path}/results/tree_regressor/single_tasks/model/task{task+1}/grid_search_model_{task+1}.pkl"
                grid_search = joblib.load(joblib_file)
            selected_energy, selected_iter, training_loss, iterations_test,test_y, acquisition,task, num_iterations = tree_based_active_learning(grid_search, task, cropped_image_list, energy_to_ind, 30, "uncertainty")
        elif args.method == "GMM":
            pass
if __name__ == "__main__":
    main()
