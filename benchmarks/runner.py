# benchmark/runner.py
import argparse
import yaml
import importlib
import mlflow
import os
from datetime import datetime
from benchmarks.utils.helpers import fix_seeds, setup_mlflow, get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_task(task_name: str, task_config: dict, global_config: dict) -> dict:
    class_name = task_config.get("evaluator_class")
    module_name = f"benchmark.tasks.{task_config.get('module')}"

    logger.info(f"Initializing {class_name} from {module_name}")
    module = importlib.import_module(module_name)
    evaluator_class = getattr(module, class_name)

    evaluator_config = {**global_config, **task_config}
    evaluator = evaluator_class(evaluator_config)

    evaluator.load_model(task_config.get("model_path", "default"))
    evaluator.load_dataset(task_config.get("dataset_path"))

    metrics = evaluator.compute_metrics()

    output_dir = global_config.get("output_dir", "./benchmark/results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{task_name}_{timestamp}.json")
    evaluator.save_results(output_path, metrics)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Enterprise Benchmark Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--task", type=str, help="Run specific task by name")
    args = parser.parse_args()

    config = load_config(args.config)
    global_config = config.get("global", {})

    fix_seeds(global_config.get("seed", 1235))
    setup_mlflow(global_config.get("mlflow_tracking_uri", ""), global_config.get("experiment_name", "benchmarks"))

    tasks = config.get("tasks", {})
    if args.task:
        if args.task not in tasks:
            raise ValueError(f"Task {args.task} not found in config.")
        tasks = {args.task: tasks[args.task]}

    for task_name, task_config in tasks.items():
        logger.info(f"--- Starting task: {task_name} ---")
        with mlflow.start_run(run_name=task_name):
            mlflow.log_params(global_config)
            mlflow.log_params(task_config)
            mlflow.set_tag("run_type", "smoke" if global_config.get("smoke_test") else "full")

            try:
                metrics = run_task(task_name, task_config, global_config)
                # Filter out lists/matrices before logging to MLflow
                loggable_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                mlflow.log_metrics(loggable_metrics)
            except Exception as e:
                logger.error(f"Task {task_name} failed: {str(e)}")
                raise


if __name__ == "__main__":
    main()