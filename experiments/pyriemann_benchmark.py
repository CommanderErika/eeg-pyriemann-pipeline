import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracking import ExperimentOrchestrator, MLflowTracker
from src.models import SVMModel, LogisticModel, LDAModel

# First setup the server
# mlflow server --host 127.0.0.1 --port 8080

if __name__ == "__main__":

    LIB: str        = "RiemannDSP"
    DATA_DIR: str   = "./data/processed/ts/" if LIB == "PyRiemann" else "./data/riemanndsp/ts/"
    DATASETS: str   = [ "Cho2017", "BNCI2014_004", "BNCI2014_001",
                        "Liu2024", "Lee2019_MI", "PhysionetMI"]
    URI: str        = "http://127.0.0.1:8080"
    N_TRIALS        = 1
    USE_PROCRUSTES  = False

    for dataset in DATASETS:

        # Experiment Name
        EXP_NAME: str   = f'[{LIB}][{dataset}]: Classify MI with Tangent Space'

        if USE_PROCRUSTES:
            EXP_NAME = EXP_NAME + " with Procruste Alignment"

        # Tracker and Orchestrator
        mlflow_tracker  = MLflowTracker(experiment_name         =EXP_NAME)
        orchestrator    = ExperimentOrchestrator(process_lib    =LIB,
                                                 data_dir       =DATA_DIR,
                                                 use_pa         =USE_PROCRUSTES,
                                                 tracker        =mlflow_tracker,
                                                 dataset        =dataset
                                                 )
        # Registry Models

        # SVM Model
        orchestrator.registry_model(model_class     = SVMModel,
                                    name            = "SVM",
                                    params          = { 'kernel' : ['linear', 'rbf'],
                                                        'C'      : (0.5, 50.0),   
                                                        # Increase cache to use more RAM (default is 200MB)
                                                        'cache_size': [2000]
                                                        })
        
        # Logistic Regression
        orchestrator.registry_model(model_class     = LogisticModel,
                                    name            = "Logistic_Regression",
                                    params          = { 
                                                        'penalty'     : ['l2'], # Keep simple for stability
                                                        'C'           : (0.1, 10.0),
                                                        'solver'      : ['lbfgs'],
                                                        'max_iter'    : [2000], # Fixed value is fine
                                                    }                
                                    )
        
        # LDA
        orchestrator.registry_model(model_class = LDAModel,
                                    name        = "LDA",
                                    params      = { 
                                        'solver'    : ['lsqr', 'eigen'],
                                        'shrinkage' : ['auto', 0.1, 0.5, 0.9] 
                                    })

        # Running Experiments
        _ = orchestrator.run_benchmark(n_trials=N_TRIALS)

        del mlflow_tracker
        del orchestrator
    