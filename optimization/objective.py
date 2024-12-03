import optuna
from optimization.objective import Objective
from config import load_config

def main():
    config = load_config("config/config.yaml")
    study = optuna.create_study(direction="maximize", study_name="object_detection_optimization")
    study.optimize(Objective(config), n_trials=config["optuna"]["n_trials"])
    print(f"Best trial: {study.best_trial.params}")
    study.trials_dataframe().to_csv("logs/optuna_results.csv")

if __name__ == "__main__":
    main()
