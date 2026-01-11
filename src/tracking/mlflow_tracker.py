from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import interpolate
import mlflow

from src.visualization import (
    confusion_matrix_plot,
)

class BaseTracker(ABC):
    @abstractmethod
    def log_experiment(self, experiment_data: dict[str, any]):
        pass

@dataclass
class MLflowTracker(BaseTracker):
    experiment_name : str = field(init=True)
    uri             : str = field(init=True, default="http://127.0.0.1:8080")

    def __post_init__(self):
        """Sets up the MLflow connection and experiment."""
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self.experiment_name)

    def log_experiment(self, experiment_data: dict[str, any], y_train, y_test):
        # Log metadata
        mlflow.log_params(experiment_data['params'])
        # Log metrics
        self._log_metrics(metrics=experiment_data['metrics_train'], prefix='train_')
        self._log_metrics(metrics=experiment_data['metrics_test'], prefix='test_')
        # Log tags
        mlflow.set_tags({
            'dataset':          experiment_data['dataset_name'],
            'processing_lib' :  experiment_data['processing_lib'],
            'model_type':       experiment_data['model_type'],
            'PA'      :         experiment_data['use_pa'],
            'run_type':         'hyperparameter_tuning',
        })
        # Log artefacts
        images = {'x_train_dim' : experiment_data['x_train_dim'],
                  'x_test_dim'  : experiment_data['x_test_dim']}
        
        # Plot 2D
        self._log_artefacts_2d(images, y_train, y_test)
        # Plot 3D
        self._log_artefacts_3d(images, y_train, y_test)
        self._log_artefacts_3d_proj(images, y_train, y_test)

    def _log_metrics(self, metrics: dict, prefix: str = ''):
        """Methods to log metrics to mlflow"""
        # Precision, Recall, and F1-Score
        metrics_avg = {f'{prefix}{key}': value for key, value in metrics.items()}
        mlflow.log_metrics(metrics_avg)

    def _log_artefacts_2d(self, images, y_train, y_test):
        """Methods to log artefacts to mlflow with separate plots for left_hand and right_hand"""
        
        # Criar figura com 2 subplots (uma para cada classe)
        figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot para LEFT_HAND
        # Dados de treino - left_hand
        mask_train_left = y_train == 'left_hand'
        ax1.scatter(images['x_train_dim'][mask_train_left, 0], 
                    images['x_train_dim'][mask_train_left, 1], 
                    label='Train - Left Hand', 
                    alpha=0.7, 
                    s=60,
                    color='blue',
                    marker='o')
        
        # Dados de teste - left_hand
        mask_test_left = y_test == 'left_hand'
        ax1.scatter(images['x_test_dim'][mask_test_left, 0], 
                    images['x_test_dim'][mask_test_left, 1], 
                    label='Test - Left Hand', 
                    alpha=0.7, 
                    s=60,
                    color='red',
                    marker='s')
        
        ax1.set_title('Left Hand - Train vs Test')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot para RIGHT_HAND
        # Dados de treino - right_hand
        mask_train_right = y_train == 'right_hand'
        ax2.scatter(images['x_train_dim'][mask_train_right, 0], 
                    images['x_train_dim'][mask_train_right, 1], 
                    label='Train - Right Hand', 
                    alpha=0.7, 
                    s=60,
                    color='green',
                    marker='o')
        
        # Dados de teste - right_hand
        mask_test_right = y_test == 'right_hand'
        ax2.scatter(images['x_test_dim'][mask_test_right, 0], 
                    images['x_test_dim'][mask_test_right, 1], 
                    label='Test - Right Hand', 
                    alpha=0.7, 
                    s=60,
                    color='orange',
                    marker='s')
        
        ax2.set_title('Right Hand - Train vs Test')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar imagem
        mlflow.log_figure(figure=figure, artifact_file='Dim_Reduction_by_class_2d.png')
        plt.close(figure)
        
        # Plot adicional: todos juntos para comparação
        figure2 = plt.figure(figsize=(12, 8))
        
        # Left Hand - Train
        mask_train_left = y_train == 'left_hand'
        plt.scatter(images['x_train_dim'][mask_train_left, 0], 
                    images['x_train_dim'][mask_train_left, 1], 
                    label='Train - Left Hand', 
                    alpha=0.7, 
                    s=50,
                    color='blue',
                    marker='o')
        
        # Left Hand - Test
        mask_test_left = y_test == 'left_hand'
        plt.scatter(images['x_test_dim'][mask_test_left, 0], 
                    images['x_test_dim'][mask_test_left, 1], 
                    label='Test - Left Hand', 
                    alpha=0.7, 
                    s=50,
                    color='red',
                    marker='s')
        
        # Right Hand - Train
        mask_train_right = y_train == 'right_hand'
        plt.scatter(images['x_train_dim'][mask_train_right, 0], 
                    images['x_train_dim'][mask_train_right, 1], 
                    label='Train - Right Hand', 
                    alpha=0.7, 
                    s=50,
                    color='green',
                    marker='o')
        
        # Right Hand - Test
        mask_test_right = y_test == 'right_hand'
        plt.scatter(images['x_test_dim'][mask_test_right, 0], 
                    images['x_test_dim'][mask_test_right, 1], 
                    label='Test - Right Hand', 
                    alpha=0.7, 
                    s=50,
                    color='orange',
                    marker='s')
        
        plt.title('All Data - Train vs Test by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        mlflow.log_figure(figure=figure2, artifact_file='Dim_Reduction_Combined_2d.png')
        plt.close(figure2)

    def _log_artefacts_3d_proj(self, images, y_train, y_test):
        """Methods to log artefacts with 4 separate 3D plots"""
        
        if images['x_train_dim'].shape[1] < 3:
            print("Warning: Data with less than 3 dims.")
            return
        
        figure = plt.figure(figsize=(20, 16))
        
        # Plot 1: train + Left
        ax1 = figure.add_subplot(2, 2, 1, projection='3d')
        mask_train_left = y_train == 'left_hand'
        
        x_train_left = images['x_train_dim'][mask_train_left, 0]
        y_train_left = images['x_train_dim'][mask_train_left, 1]
        z_train_left = images['x_train_dim'][mask_train_left, 2]
        
        ax1 = self.plot_3d_with_kde_projections(ax1, 
                                            x_train_left, y_train_left, z_train_left,
                                            'blue', 'Train - Left Hand')
        
        ax1.set_title('Train - Left Hand', fontsize=14, fontweight='bold')
        
        # Plot 2: train + Right
        ax2 = figure.add_subplot(2, 2, 2, projection='3d')
        mask_train_right = y_train == 'right_hand'

        x_train_right = images['x_train_dim'][mask_train_right, 0]
        y_train_right = images['x_train_dim'][mask_train_right, 1]
        z_train_right = images['x_train_dim'][mask_train_right, 2]
        
        ax2 = self.plot_3d_with_kde_projections(ax2, 
                                            x_train_right, y_train_right, z_train_right,
                                            'green', 'Train - Right Hand')
        
        ax2.set_title('Train - Right Hand', fontsize=14, fontweight='bold')
        
        # Plot 3: Test + Left
        ax3 = figure.add_subplot(2, 2, 3, projection='3d')
        mask_test_left = y_test == 'left_hand'
        
        x_test_left = images['x_test_dim'][mask_test_left, 0]
        y_test_left = images['x_test_dim'][mask_test_left, 1]
        z_test_left = images['x_test_dim'][mask_test_left, 2]
        
        ax3 = self.plot_3d_with_kde_projections(ax3, 
                                            x_test_left, y_test_left, z_test_left,
                                            'red', 'Test - Left Hand')
        
        ax3.set_title('Test - Left Hand', fontsize=14, fontweight='bold')
        
        # Plot 4: Test + Right 
        ax4 = figure.add_subplot(2, 2, 4, projection='3d')
        mask_test_right = y_test == 'right_hand'

        x_test_right = images['x_test_dim'][mask_test_right, 0]
        y_test_right = images['x_test_dim'][mask_test_right, 1]
        z_test_right = images['x_test_dim'][mask_test_right, 2]
        
        ax4 = self.plot_3d_with_kde_projections(ax4, 
                                            x_test_right, y_test_right, z_test_right,
                                            'orange', 'Test - Right Hand')
        
        ax4.set_title('Test - Right Hand', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        mlflow.log_figure(figure=figure, artifact_file='Dim_Reduction_Separated_3d_proj.png')
        plt.close(figure)

    def _log_artefacts_3d(self, images, y_train, y_test):
        """Methods to log artefacts with simple 3D plots"""
        
        if images['x_train_dim'].shape[1] < 3:
            print("Warning: data with less than 3 dims.")
            return
        
        figure = plt.figure(figsize=(18, 6))
        
        # Plot 1: Left Hand 3D
        ax1 = figure.add_subplot(1, 3, 1, projection='3d')
        mask_train_left = y_train == 'left_hand'
        mask_test_left = y_test == 'left_hand'
        
        ax1.scatter(images['x_train_dim'][mask_train_left, 0], 
                    images['x_train_dim'][mask_train_left, 1],
                    images['x_train_dim'][mask_train_left, 2],
                    label='Train', alpha=0.7, s=50, color='blue', marker='o')
        ax1.scatter(images['x_test_dim'][mask_test_left, 0], 
                    images['x_test_dim'][mask_test_left, 1],
                    images['x_test_dim'][mask_test_left, 2],
                    label='Test', alpha=0.7, s=50, color='red', marker='s')
        ax1.set_title('Left Hand - 3D')
        ax1.legend()
        
        # Plot 2: Right Hand 3D
        ax2 = figure.add_subplot(1, 3, 2, projection='3d')
        mask_train_right = y_train == 'right_hand'
        mask_test_right = y_test == 'right_hand'
        
        ax2.scatter(images['x_train_dim'][mask_train_right, 0], 
                    images['x_train_dim'][mask_train_right, 1],
                    images['x_train_dim'][mask_train_right, 2],
                    label='Train', alpha=0.7, s=50, color='green', marker='o')
        ax2.scatter(images['x_test_dim'][mask_test_right, 0], 
                    images['x_test_dim'][mask_test_right, 1],
                    images['x_test_dim'][mask_test_right, 2],
                    label='Test', alpha=0.7, s=50, color='orange', marker='s')
        ax2.set_title('Right Hand - 3D')
        ax2.legend()
        
        # Plot 3: Overview 3D
        ax3 = figure.add_subplot(1, 3, 3, projection='3d')
        for class_label, color in [('left_hand', 'blue'), ('right_hand', 'green')]:
            mask_train = y_train == class_label
            mask_test = y_test == class_label
            ax3.scatter(images['x_train_dim'][mask_train, 0], 
                        images['x_train_dim'][mask_train, 1],
                        images['x_train_dim'][mask_train, 2],
                        label=f'{class_label} - Train', alpha=0.6, s=40, color=color, marker='o')
            ax3.scatter(images['x_test_dim'][mask_test, 0], 
                        images['x_test_dim'][mask_test, 1],
                        images['x_test_dim'][mask_test, 2],
                        label=f'{class_label} - Test', alpha=0.6, s=40, color=color, marker='s')
        ax3.set_title('Overview - 3D')
        ax3.legend()
        
        # plt.tight_layout()
        mlflow.log_figure(figure=figure, artifact_file='Dim_Reduction_Combined_3d.png')
        plt.close(figure)

    def _grid(self, x, y, z, resX=1000, resY=1000):
        "Convert 3 column data to matplotlib grid"
        xi = np.linspace(min(x), max(x), resX)
        yi = np.linspace(min(y), max(y), resY)
        Z = interpolate.griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')
        X, Y = np.meshgrid(xi, yi)
        return X, Y, Z

    def plot_3d_projections_only(self, ax, x, y, z):
        """
        Plots only the 3D projections on the coordinate planes at the edges.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The 3D axis to plot on
        x, y, z : array-like
            Arrays of x, y, z coordinates to plot
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axis with the projection plots
        """
        # Get the limits for positioning projections at edges
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        z_min, z_max = np.min(z), np.max(z)
        
        # Projection on XZ plane (y = y_min - at the bottom)
        ax.plot(x, z, 'y+', zdir='y', zs=y_max+1)
        
        # Projection on YZ plane (x = x_min - at the left)
        ax.plot(y, z, 'g+', zdir='x', zs=x_min-1)
        
        # Projection on XY plane (z = z_min - at the back)
        ax.plot(x, y, 'k+', zdir='z', zs=z_min-1)

        #ax.set_xlim([x_min-0.5, x_max+0.5])
        #ax.set_ylim([y_min-0.5, y_max+0.5])
        #ax.set_zlim([z_min-0.5, z_max+0.5])
        
        return ax
    
    def plot_3d_separate_projections(self, ax, x_train, y_train, z_train,
                                x_test, y_test, z_test,
                                train_color, test_color):
        """
        Projections separated by set.
        """
        # Scatters
        ax.scatter(x_train, y_train, z_train, alpha=0.8, s=50, 
                color=train_color, marker='o', label='Train')
        ax.scatter(x_test, y_test, z_test, alpha=0.8, s=50, 
                color=test_color, marker='s', label='Test')
        
        # Limits
        z_min = np.min(np.concatenate([z_train, z_test]))
        y_min = np.min(np.concatenate([y_train, y_test]))
        x_min = np.min(np.concatenate([x_train, x_test]))
        
        # Train projections
        if len(x_train) > 0:
            ax.plot(x_train, y_train, [z_min]*len(x_train), ',', 
                color=train_color, alpha=0.4, zdir='z', linewidth=1)
            ax.plot(x_train, [y_min]*len(x_train), z_train, ',', 
                color=train_color, alpha=0.4, zdir='y', linewidth=1)
            ax.plot([x_min]*len(x_train), y_train, z_train, ',', 
                color=train_color, alpha=0.4, zdir='x', linewidth=1)
        
        # Test projections
        if len(x_test) > 0:
            ax.plot(x_test, y_test, [z_min]*len(x_test), ',', 
                color=test_color, alpha=0.4, zdir='z', linewidth=1)
            ax.plot(x_test, [y_min]*len(x_test), z_test, ',', 
                color=test_color, alpha=0.4, zdir='y', linewidth=1)
            ax.plot([x_min]*len(x_test), y_test, z_test, ',', 
                color=test_color, alpha=0.4, zdir='x', linewidth=1)
        
        return ax
    
    def plot_3d_with_heatmap_projections(self, ax, x, y, z, color, label):
        """
        3D Scatter with 2D heatmaps in the plane projections
        """
        # Scatter plot
        scatter = ax.scatter(x, y, z, alpha=0.7, s=30, 
                            color=color, marker='o', label=label)
        
        # Data Limits
        if len(x) > 0:
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            z_min, z_max = np.min(z), np.max(z)
            
            # Add margin
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            z_margin = (z_max - z_min) * 0.1

            # Define limits manually
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.set_zlim(z_min - z_margin, z_max + z_margin)
            
            # Create a grid fot the heatmaps
            grid_size = 30
            
            # Plane XY - z = z_min
            if len(x) > 0 and len(y) > 0:
                heatmap_xy, x_edges, y_edges = np.histogram2d(x, y, 
                                                            bins=grid_size, 
                                                            range=[[x_min, x_max], [y_min, y_max]])
                if np.max(heatmap_xy) > 0:
                    heatmap_xy = heatmap_xy / np.max(heatmap_xy)
                
                X_xy, Y_xy = np.meshgrid(x_edges[:-1], y_edges[:-1])
                ax.contourf(X_xy, Y_xy, heatmap_xy.T, zdir='z', offset=z_min, 
                        alpha=0.4, cmap=self._get_cmap_for_color(color), levels=5)
            
            # Plane XZ - y = y_min
            if len(x) > 0 and len(z) > 0:
                heatmap_xz, x_edges, z_edges = np.histogram2d(x, z, 
                                                            bins=grid_size, 
                                                            range=[[x_min, x_max], [z_min, z_max]])
                if np.max(heatmap_xz) > 0:
                    heatmap_xz = heatmap_xz / np.max(heatmap_xz)
                
                X_xz, Z_xz = np.meshgrid(x_edges[:-1], z_edges[:-1])
                ax.contourf(X_xz, heatmap_xz.T, Z_xz, zdir='y', offset=y_max, 
                        alpha=0.3, cmap=self._get_cmap_for_color(color), levels=5)
            
            # Plane YZ - x = x_max
            if len(y) > 0 and len(z) > 0:
                heatmap_yz, y_edges, z_edges = np.histogram2d(y, z, 
                                                            bins=grid_size, 
                                                            range=[[y_min, y_max], [z_min, z_max]])
                if np.max(heatmap_yz) > 0:
                    heatmap_yz = heatmap_yz / np.max(heatmap_yz)
                
                Y_yz, Z_yz = np.meshgrid(y_edges[:-1], z_edges[:-1])
                ax.contourf(heatmap_yz.T, Y_yz, Z_yz, zdir='x', offset=x_min, 
                        alpha=0.3, cmap=self._get_cmap_for_color(color), levels=5)
        
        return ax

    def plot_3d_with_kde_projections(self, ax, x, y, z, color, label):
        """
        3D Scatter with KDE projections in the planes.
        """
        from scipy.stats import gaussian_kde
        
        # Scatter plot
        scatter = ax.scatter(x, y, z, alpha=0.7, s=30, 
                            color=color, marker='o', label=label)
        
        # Data limits
        if len(x) > 0:
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            z_min, z_max = np.min(z), np.max(z)
            
            # Add margin
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            z_margin = (z_max - z_min) * 0.1
            
            # Define the limits manually
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.set_zlim(z_min - z_margin, z_max + z_margin)
            
            # Create KDE grids
            grid_size = 30
            x_grid = np.linspace(x_min, x_max, grid_size)
            y_grid = np.linspace(y_min, y_max, grid_size)
            z_grid = np.linspace(z_min, z_max, grid_size)
            
            # Plano XY - z = z_min
            if len(x) > 1 and len(y) > 1:  # KDE precisa de pelo menos 2 pontos
                try:
                    # Calculates KDE
                    xy_data = np.vstack([x, y])
                    kde_xy = gaussian_kde(xy_data)
                    
                    # Create meshgrid
                    X_xy, Y_xy = np.meshgrid(x_grid, y_grid)
                    positions_xy = np.vstack([X_xy.ravel(), Y_xy.ravel()])
                    
                    # Calculate KDE density
                    Z_kde_xy = kde_xy(positions_xy).reshape(X_xy.shape)
                    
                    # Normalization
                    Z_kde_xy = Z_kde_xy / np.max(Z_kde_xy)
                    
                    # Plot contours
                    ax.contourf(X_xy, Y_xy, Z_kde_xy, zdir='z', offset=z_min, 
                            alpha=0.4, cmap=self._get_cmap_for_color(color), levels=8)
                except (ValueError, np.linalg.LinAlgError):
                    # Fallback
                    self._plot_histogram_fallback(ax, x, y, z_min, 'z', color, 'xy')
            
            # Plano XZ - y = y_min
            if len(x) > 1 and len(z) > 1:
                try:
                    # Calculates KDE
                    xz_data = np.vstack([x, z])
                    kde_xz = gaussian_kde(xz_data)
                    
                    # Create meshgrid
                    X_xz, Z_xz = np.meshgrid(x_grid, z_grid)
                    positions_xz = np.vstack([X_xz.ravel(), Z_xz.ravel()])
                    
                    # Calculate KDE density
                    Z_kde_xz = kde_xz(positions_xz).reshape(X_xz.shape)
                    
                    # Normalization
                    Z_kde_xz = Z_kde_xz / np.max(Z_kde_xz)
                    
                    # Plot contours
                    ax.contourf(X_xz, Z_kde_xz, Z_xz, zdir='y', offset=y_max, 
                            alpha=0.3, cmap=self._get_cmap_for_color(color), levels=8)
                except (ValueError, np.linalg.LinAlgError):
                    self._plot_histogram_fallback(ax, x, z, y_min, 'y', color, 'xz')
            
            # Plano YZ - x = x_max
            if len(y) > 1 and len(z) > 1:
                try:
                    # Calculates KDE
                    yz_data = np.vstack([y, z])
                    kde_yz = gaussian_kde(yz_data)
                    
                    # Create meshgrid
                    Y_yz, Z_yz = np.meshgrid(y_grid, z_grid)
                    positions_yz = np.vstack([Y_yz.ravel(), Z_yz.ravel()])
                    
                    # Calculate KDE density
                    Z_kde_yz = kde_yz(positions_yz).reshape(Y_yz.shape)
                    
                    # Normalization
                    Z_kde_yz = Z_kde_yz / np.max(Z_kde_yz)
                    
                    # Plot contours
                    ax.contourf(Z_kde_yz, Y_yz, Z_yz, zdir='x', offset=x_min, 
                            alpha=0.3, cmap=self._get_cmap_for_color(color), levels=8)
                except (ValueError, np.linalg.LinAlgError):
                    self._plot_histogram_fallback(ax, y, z, x_max, 'x', color, 'yz')
        
        return ax

    def _plot_histogram_fallback(self, ax, data1, data2, offset, zdir, color, plane):
        """
        Fallbacj if the KDE fails
        """
        grid_size = 15
        heatmap, edges1, edges2 = np.histogram2d(data1, data2, bins=grid_size)
        
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
            
            X, Y = np.meshgrid(edges1[:-1], edges2[:-1])
            
            if plane == 'xy':
                ax.contourf(X, Y, heatmap.T, zdir=zdir, offset=offset, 
                        alpha=0.3, cmap=self._get_cmap_for_color(color), levels=5)
            elif plane == 'xz':
                ax.contourf(X, heatmap.T, Y, zdir=zdir, offset=offset, 
                        alpha=0.3, cmap=self._get_cmap_for_color(color), levels=5)
            elif plane == 'yz':
                ax.contourf(heatmap.T, X, Y, zdir=zdir, offset=offset, 
                        alpha=0.3, cmap=self._get_cmap_for_color(color), levels=5)

    def _get_cmap_for_color(self, color):
        """
        Return the apropriated colormap based on the color
        """
        color_map = {
            'blue': 'Blues',
            'green': 'Greens', 
            'red': 'Reds',
            'orange': 'Oranges',
            'purple': 'Purples',
            'brown': 'YlOrBr'
        }
        return color_map.get(color, 'viridis')