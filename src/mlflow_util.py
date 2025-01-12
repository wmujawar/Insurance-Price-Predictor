import mlflow


def setup_mlflow_experiment(exp_name: str, exp_description: str) -> mlflow.entities.Experiment:
    """
    Set up an MLflow experiment.

    Parameters:
    exp_name (str): The name of the experiment.
    exp_description (str): The description of the experiment.

    Returns:
    mlflow.entities.Experiment: The MLflow experiment object.
    """
    
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    mlflow.set_tracking_uri(f"file://{parent_dir}/mlruns")
    experiment = mlflow.get_experiment_by_name(exp_name)

    if not experiment:
        mlflow.create_experiment(exp_name, tags={'mlflow.note.content': exp_description})
        experiment = mlflow.get_experiment_by_name(exp_name)
        
    return experiment
