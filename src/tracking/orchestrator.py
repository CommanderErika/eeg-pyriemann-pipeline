from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Any, Union
from collections import defaultdict
from dataclasses import dataclass, field

import os
import gc
import sys
import traceback
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import mlflow
import optuna
import numpy as np
import yaml
import logging.config

# Machine Learning Imports
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from umap import UMAP

# Custom Imports
from procrustes.generic import generic
from src.data import HDF5Manager
from src.models import BaseModel
from .mlflow_tracker import BaseTracker


@dataclass
class ExperimentOrchestrator:
    # Public
    tracker: BaseTracker
    process_lib: str = field(init=True)
    data_dir: str = field(init=True)
    dataset: str = field(init=True)
    tracking_uri: str = field(default="http://127.0.0.1:8888")

    # Internal
    files: List[str] = field(init=False, default_factory=list)
    _data_manager: HDF5Manager = field(init=False)
    # Stores registered models
    _models: Dict[str, Any] = field(init=False, default_factory=dict)
    _params_list: Dict[str, Dict[str, Any]] = field(init=False, default_factory=dict)
    _model: Optional[BaseModel] = field(init=False, default=None)
    
    # Tracks if a specific model is forced for the current run
    _target_model_name: Optional[str] = field(init=False, default=None)

    def __post_init__(self):

        # Setup Logging using YAML
        self._setup_logging()
        
        # Logger
        self.logger = logging.getLogger(__name__)

        # Data Manager
        self.files = self._get_file_list()
        self._data_manager = HDF5Manager()

        self.logger.info(f"Orchestrator initialized. Dataset: {self.dataset} | Files: {len(self.files)}")

    def _setup_logging(self, config_path="configs.yaml"):
        """Loads logging configuration from YAML file."""
        # Ensure logs directory exists (otherwise FileHandler fails)
        os.makedirs("logs", exist_ok=True)
        
        if os.path.exists(config_path):
            with open(config_path, 'rt') as f:
                config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
        else:
            # Fallback if yaml is missing
            logging.basicConfig(level=logging.INFO)
            print(f"Warning: {config_path} not found. Using default logging.")

    def registry_model(self, model_class: Any, params: Dict[str, Any], name: str = None):
        """
        Registers a model. 
        Args:
            name: Optional alias. If None, uses the class name (e.g., 'SVMModel').
                  Use this to register the same model class multiple times.
        """
        # 1. Determine the unique key for this model
        model_key = name if name else model_class.__name__
        
        # 2. Safety check: Warn if we are overwriting
        if model_key in self._models:
            print(f"WARNING: Overwriting existing registration for '{model_key}'")

        # 3. Store using the key
        self._models[model_key] = model_class
        self._params_list[model_key] = params

    def run_benchmark(self, n_trials: int = 20) -> Dict[str, Dict]:
        """
        NEW: Systematic Benchmark Mode.
        Iterates through ALL registered models and runs a dedicated optimization
        experiment for each one sequentially.
        """
        results = {}
        self.logger.info(f"{'='*60}")
        self.logger.info(f"STARTING BENCHMARK: {len(self._models)} models")
        self.logger.info(f"{'='*60}")

        for model_name in self._models.keys():
            self.logger.info(f">>> BENCHMARKING MODEL: {model_name} <<<")
            try:
                # Force the experiment to run ONLY this model
                best_params = self.run_experiments(
                    n_trials=n_trials, 
                    n_jobs=1, 
                    model_name=model_name
                )
                results[model_name] = best_params
            except Exception as e:
                self.logger.error(f"!!! FAILED BENCHMARK FOR {model_name}: {e}", exc_info=True)
                results[model_name] = "FAILED"
        
        return results

    def run_experiments(self, n_trials: int = 2, n_jobs: int = 1, model_name: str = None):
        """
        Run Optuna optimization.
        
        Args:
            n_trials: Number of trials.
            n_jobs: Parallel jobs (keep to 1 if using SQLite).
            model_name: If provided, forces Optuna to ONLY optimize this specific model.
                        If None, Optuna chooses the model (AutoML style).
        """
        # Store target model for _objective to see
        self._target_model_name = model_name

        # Create distinct study names for benchmarks vs automl
        if model_name:
            study_name = f"{self.dataset}_{model_name}"
        else:
            study_name = f"{self.dataset}_automl"
        
        # Create the full path string first
        project_root = os.getcwd()
        data_path = "./data/optuna_db"

        # Create the filename
        db_filename = f"{self.dataset}_{model_name}.db"

        # Create the absolute path
        db_path = os.path.join(project_root, data_path, db_filename)

        # Create the URL
        storage_url = f"sqlite:///{db_path}"

        print(f"DEBUG: Storing database at: {db_path}")

        try:
            
            study = optuna.create_study(
                direction="maximize",
                study_name=study_name,
                storage=storage_url,
                load_if_exists=True
            )
            
            study.optimize(
                self._objective,
                n_trials=n_trials,
                gc_after_trial=True,
                n_jobs=n_jobs,
                show_progress_bar=True,
            )
            return study.best_params
        finally:
            self._cleanup_resources()
            self._kill_orphaned_processes()

    def _objective(self, trial: optuna.trial.Trial) -> float:
        """
        Core optimization loop. Handles both AutoML mode and Benchmark mode.
        """
        try:
            with mlflow.start_run(nested=True):
                self.logger.info(f"--- Starting Trial {trial.number} ---")
                
                # Model Selection Logic
                if self._target_model_name:
                    model_name = self._target_model_name
                    trial.set_user_attr("model_type", model_name)
                    mlflow.log_param("model_type", model_name)
                else:
                    available_models = list(self._models.keys())
                    if not available_models:
                        raise ValueError("No models registered.")
                    model_name = trial.suggest_categorical("model_type", available_models)

                # Get Model & Hyperparameter Suggestion
                model_class = self._models[model_name]
                params = self._verify_space_params(model_name, trial)

                self.logger.debug(f"Model: {model_name} | Params: {params}")

                # Load
                x_data, y_data, subjects = self._load_data()

                # ----- Leave-One-Subject-Out (LOSO) -----
                logo = LeaveOneGroupOut()
                # n_groups = logo.get_n_splits(groups=subjects)

                # Dictionary to store list of scores for every metric
                cv_results_train = defaultdict(list)
                cv_results_test = defaultdict(list)

                # Placeholders for visualization data (we will grab them from the last fold)
                last_x_train = None
                last_x_test_RPA = None
                last_y_train = None
                last_y_test = None

                # Initialize counter
                fold_idx = 1

                for train_idx, test_idx in logo.split(x_data, y_data, groups=subjects):

                    # Slice Data using indices
                    # uv run ./experiments/pyriemann_benchmark.py
                    x_train, y_train = x_data[train_idx], y_data[train_idx]
                    x_test, y_test = x_data[test_idx], y_data[test_idx]

                    # --- SAFETY CHECKS (Add this block) ---
                    # Check for NaNs
                    if np.isnan(x_train).any() or np.isinf(x_train).any():
                        self.logger.error(f"CRITICAL: NaNs found in Training Data (Fold {fold_idx})")
                        raise ValueError("Data contains NaN or Infinity.")

                    # Check Class Balance
                    unique_classes = np.unique(y_train)
                    if len(unique_classes) < 2:
                        self.logger.warning(f"Fold {fold_idx}: Training set has only 1 class ({unique_classes}). Skipping this fold.")
                        fold_idx += 1
                        continue

                    try:
                        # Init & Train
                        model = model_class(params=params)
                        model.train(x_train, y_train)
                    except Exception as e:
                        self.logger.error(f"TRAINING CRASH (Fold {fold_idx}): {e}")
                        self.logger.error(f"X shape: {x_train.shape}, Y unique: {unique_classes}")
                        raise e

                    # Procrustes Alignment
                    x_test_RPA = self._procrustes_by_class(
                        x_train, y_train, x_test, y_test, translate=False, scale=False
                    )

                    # Evaluate
                    _, _, metrics_train = model.evaluate(x_train, y_train)
                    _, _, metrics_test = model.evaluate(x_test_RPA, y_test)

                    # Store metrics
                    for metric_name, score_value in metrics_train.items():
                        cv_results_train[metric_name].append(score_value)

                    for metric_name, score_value in metrics_test.items():
                        cv_results_test[metric_name].append(score_value)

                    # Save data from this fold for logging/viz later
                    last_x_train = x_train
                    last_x_test_RPA = x_test_RPA
                    last_y_train = y_train
                    last_y_test = y_test

                    self.logger.debug(f"Fold {fold_idx} processing...")
                    # Increment counter for next loop
                    fold_idx += 1

                # Aggregating Scores
                final_metrics_train = {}
                for metric_name, scores in cv_results_train.items():
                    final_metrics_train[f"cv_mean_{metric_name}"] = np.mean(scores)
                    final_metrics_train[f"cv_std_{metric_name}"]  = np.std(scores)

                final_metrics_test = {}
                for metric_name, scores in cv_results_test.items():
                    final_metrics_test[f"cv_mean_{metric_name}"] = np.mean(scores)
                    final_metrics_test[f"cv_std_{metric_name}"]  = np.std(scores)

                # Dimensionality Reduction (Visualization only)
                x_train_reduced, x_RPA_reduced = self.apply_pca_lda_umap(
                    last_x_train, last_y_train, last_x_test_RPA, last_y_test, 
                    method='umap', n_components=3
                )

                # Logging
                experiment_data = {
                    'params': params,
                    'metrics_train': final_metrics_train,
                    'metrics_test': final_metrics_test,
                    'dataset_name': self.dataset,
                    'processing_lib': self.process_lib,
                    'model_type': model_name,
                    'x_train_dim': x_train_reduced,
                    'x_test_dim': x_RPA_reduced
                }

                self.tracker.log_experiment(
                    experiment_data=experiment_data, 
                    y_train=y_train, 
                    y_test=y_test
                )

                target_metric = final_metrics_test.get('cv_mean_f1-score_macro')
                if target_metric is None:
                     target_metric = final_metrics_test.get('cv_mean_macro_f1-score', 0.0)

                return target_metric

        except Exception as e:
            self.logger.critical(f"Trial {trial.number} failed: {e}")
            traceback.print_exc()
            mlflow.log_param("ERROR", str(e))
            raise optuna.TrialPruned()

    def _procrustes_by_class(self, x_train, y_train, x_test, y_test, translate=False, scale=False):
        """
        Applies Procrustes Alignment using Class Centroids (Landmarks).
        
        Process:
        1. Compute Centroids (Mean of Left, Mean of Right) for Train and Test.
        2. Center these Centroids to the origin.
        3. Use 'generic' to find the rotation Matrix T.
        4. Apply: (Test_Data - Test_Center) @ T + Train_Center
        """
        # --- 1. Calculate Centroids (Landmarks) ---
        classes = np.unique(y_train)
        
        train_landmarks = []
        test_landmarks = []
        
        for cls in classes:
            # Calculate mean vector for each class (resulting in 1 point per class)
            # Check if class exists in test set (safety for small datasets)
            if np.sum(y_test == cls) > 0:
                train_mean = np.mean(x_train[y_train == cls], axis=0)
                test_mean = np.mean(x_test[y_test == cls], axis=0)
                
                train_landmarks.append(train_mean)
                test_landmarks.append(test_mean)
        
        # Convert to numpy arrays (Shape: [N_classes, N_features])
        L_train = np.array(train_landmarks)
        L_test = np.array(test_landmarks)
        
        # --- 2. Manual Centering (Crucial Step) ---
        # We calculate the "Global Center" of the landmarks
        center_train = np.mean(L_train, axis=0)
        center_test = np.mean(L_test, axis=0)
        
        # Move landmarks to origin
        L_train_centered = L_train - center_train
        L_test_centered = L_test - center_test

        try:
            # --- 3. Compute Transformation Matrix T ---
            # We use translate=False because we already manually centered them.
            # We map Test -> Train
            RPA_result = generic(L_test_centered, L_train_centered, translate=translate, scale=scale)
            
            # RPA_result.t is the optimal transformation matrix
            Transformation_Matrix = RPA_result.t
            
            # --- 4. Apply to Full Test Dataset ---
            # Formula: X_new = (X_old - Source_Center) * T + Target_Center
            
            # A. Shift to Origin
            x_test_centered = x_test - center_test
            
            # B. Apply Rotation/Scaling
            x_test_rotated = np.dot(x_test_centered, Transformation_Matrix)
            
            # C. Shift to Target (Train) Space
            if scale and hasattr(RPA_result, 'scale'):
                 # If generic calculated a scale factor, we might need to handle it, 
                 # but usually 't' absorbs the scaling in general linear maps.
                 pass

            x_test_aligned = x_test_rotated + center_train

            return x_test_aligned

        except Exception as e:
            print(f"RPA Warning: Alignment failed ({e}), using raw test data.")
            return x_test

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load data from HDF5 files and consolidate into arrays."""
        x_list = []
        y_list = []
        subjects_list = []

        for file in self.files:
            filename = f"{self.data_dir}/{file}"
            data = self._data_manager.load(filename)
            
            print(f"\nLoaded data dims: {data.x.shape}")
            
            dim = len(data.x.shape)
            if dim == 3:
                n_trials, n_channels, _ = data.x.shape
                x_list.append(data.x.reshape((n_trials, n_channels * n_channels)))
            elif dim == 2:
                x_list.append(data.x)
            else:
                continue
            
            y_list.append(data.y)
            subjects_list.append(data.subjects)

        if not x_list:
            raise ValueError(f"No valid data found in directory {self.data_dir}")

        x_data = np.concatenate(x_list, axis=0)
        y_data = np.concatenate(y_list, axis=0)
        subjects = np.concatenate(subjects_list, axis=0)

        mlflow.log_metric("total_samples", x_data.shape[0])
        return x_data, y_data, subjects

    def _sep_by_subjects(self, data: np.ndarray, label: np.ndarray, subjects: np.ndarray, train_size: float = 0.5):
        """Splits data so that subjects in test set do not appear in training set."""
        unique_subjects = np.unique(subjects)
        n_subjects = len(unique_subjects)
        n_train = int(n_subjects * train_size)
        
        train_subjects = unique_subjects[:n_train]
        test_subjects = unique_subjects[n_train:]
        
        train_mask = np.isin(subjects, train_subjects)
        test_mask = np.isin(subjects, test_subjects)
        
        x_train, x_test = data[train_mask], data[test_mask]
        y_train, y_test = label[train_mask], label[test_mask]
        
        print(f"Train samples: {len(x_train)} | Test samples: {len(x_test)}")
        return x_train, x_test, y_train, y_test

    def apply_pca_lda_umap(self, x_train, y_train, x_test, y_test, method='pca', n_components=2, **kwargs):
        """Applies dimensionality reduction on the whole dataset."""
        if isinstance(y_train[0], str):
            label_map = {label: i for i, label in enumerate(np.unique(y_train))}
            y_train_num = np.array([label_map[label] for label in y_train])
        else:
            y_train_num = y_train

        try:
            if method.lower() == 'pca':
                model = PCA(n_components=n_components, random_state=42)
                x_train_trans = model.fit_transform(x_train)
                x_test_trans = model.transform(x_test)

            elif method.lower() == 'lda':
                n_classes = len(np.unique(y_train_num))
                actual_components = min(n_components, n_classes - 1)
                actual_components = max(1, actual_components)
                
                model = LDA(n_components=actual_components)
                x_train_trans = model.fit_transform(x_train, y_train_num)
                x_test_trans = model.transform(x_test)

            elif method.lower() == 'umap':
                n_neighbors = min(kwargs.get('n_neighbors', 15), len(x_train) - 1)
                umap_params = {'n_components': n_components, 'n_neighbors': n_neighbors, 
                               'min_dist': 0.1, 'metric': 'euclidean', 'random_state': 42}
                umap_params.update(kwargs)
                
                model = UMAP(**umap_params)
                if 'y' in kwargs or kwargs.get('target_metric'):
                     x_train_trans = model.fit_transform(x_train, y=y_train_num)
                else:
                     x_train_trans = model.fit_transform(x_train)
                x_test_trans = model.transform(x_test)
            else:
                raise ValueError("Method must be 'pca', 'lda', or 'umap'")

            return x_train_trans, x_test_trans

        except Exception as e:
            print(f"Error applying {method.upper()}: {e}")
            limit = min(n_components, x_train.shape[1])
            return x_train[:, :limit], x_test[:, :limit]

    def _verify_space_params(self, model_name: str, trial: optuna.trial.Trial) -> Dict:
        """
        Parses the registered dictionary into Optuna suggestions.
        
        Conventions:
        - List [a, b, c]    -> Categorical (Choice)
        - List [a]          -> Fixed Value
        - Tuple (min, max)  -> Range (Int or Float)
        """
        param_space = self._params_list[model_name]
        params = {}
        
        for name, config in param_space.items():
            
            # 1. Handle Single Fixed Values (List or Tuple of length 1)
            # e.g., [1000] or (1000,)
            if isinstance(config, (list, tuple)) and len(config) == 1:
                # Treat as a categorical with one option (Fixed)
                params[name] = trial.suggest_categorical(name, config)
                continue

            # 2. Handle Integer definitions
            if isinstance(config, (list, tuple)) and all(isinstance(x, int) for x in config):
                # Legacy Support: [min, max] list treated as range
                if len(config) == 2:
                    params[name] = trial.suggest_int(name, config[0], config[1])
                elif len(config) == 3:
                    params[name] = trial.suggest_int(name, config[0], config[1], step=config[2])
                else:
                    # If it's a list of ints with length != 2 or 3 (e.g. [1, 5, 10, 20])
                    # Treat as Categorical choices
                    params[name] = trial.suggest_categorical(name, config)

            # 3. Handle Float Ranges (Must be Tuples)
            # e.g., (0.1, 1.0)
            elif isinstance(config, tuple):
                if len(config) == 2:
                    params[name] = trial.suggest_float(name, config[0], config[1])
                elif len(config) == 3:
                    params[name] = trial.suggest_float(name, config[0], config[1], step=config[2])
            
            # 4. Handle Standard Categorical (Strings, mixed types)
            # e.g., ['linear', 'rbf']
            elif isinstance(config, list):
                params[name] = trial.suggest_categorical(name, config)
                
        return params

    def _get_file_list(self) -> List[str]:
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]
        if self.dataset == 'all':
            return sorted(all_files)
        return sorted(f for f in all_files if self.dataset in f)

    def _cleanup_resources(self):
        plt.close('all')
        if 'tkinter' in sys.modules:
            import tkinter
            tkinter._default_root = None
        gc.collect()

    def _kill_orphaned_processes(self):
        import psutil
        current_process = psutil.Process(os.getpid())
        for child in current_process.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass