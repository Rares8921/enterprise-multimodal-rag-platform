from kfp import dsl
from kfp.dsl import InputPath, OutputPath

@dsl.component(
    base_image='python:3.11',
    packages_to_install=['transformers', 'torch', 'datasets']
)
def load_dataset(
        dataset_path: str,
        output_dataset_path: OutputPath(str),
        smoke_test: bool = False
):
    import json
    import logging
    from datasets import Dataset, DatasetDict

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("Dataset must be a dictionary containing splits (train, validation, test).")

        if 'train' in data:
            dataset = DatasetDict({
                split: Dataset.from_dict(split_data)
                for split, split_data in data.items()
            })
        else:
            dataset = Dataset.from_dict(data)

        data_to_validate = dataset['train'] if 'train' in dataset else dataset
        sample_size = min(100, len(data_to_validate))
        required_keys = ['image', 'words', 'boxes', 'labels']

        for i in range(sample_size):
            for key in required_keys:
                if key not in data_to_validate[i]:
                    raise KeyError(f"Missing required key '{key}' in dataset schema at index {i}.")

        if smoke_test:
            logger.info("Smoke test enabled. Subsetting dataset.")
            if 'train' in dataset:
                dataset = DatasetDict({
                    split: split_data.select(range(min(10, len(split_data))))
                    for split, split_data in dataset.items()
                })
            else:
                dataset = dataset.select(range(min(10, len(dataset))))

        dataset.save_to_disk(output_dataset_path)
        logger.info(f"Dataset saved to {output_dataset_path}")

    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise


@dsl.component(
    base_image='pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime',
    packages_to_install=['transformers==4.37.0', 'mlflow==2.10.0', 'accelerate', 'datasets']
)
def finetune_layoutlm(
        dataset_path: InputPath(str),
        output_model_path: OutputPath(str),
        model_name: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        experiment_name: str,
        mlflow_tracking_uri: str,
        num_labels: int,
        smoke_test: bool = False
) -> NamedTuple('Outputs', [('model_version', str), ('model_path', str)]):
    import logging
    import mlflow
    import os
    import torch
    import warnings
    from transformers import (
        LayoutLMv3ForTokenClassification,
        LayoutLMv3Processor,
        TrainingArguments,
        Trainer
    )
    from datasets import load_from_disk

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run() as run:
            mlflow.log_params({
                'model_name': model_name,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_labels': num_labels,
                'smoke_test': smoke_test
            })

            dataset = load_from_disk(dataset_path)
            train_dataset = dataset['train'] if 'train' in dataset else dataset
            eval_dataset = dataset['validation'] if 'validation' in dataset else None

            model = LayoutLMv3ForTokenClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
            processor = LayoutLMv3Processor.from_pretrained(model_name)

            checkpoint_dir = os.path.join(output_model_path, "checkpoints")

            training_args = TrainingArguments(
                output_dir=checkpoint_dir,
                num_train_epochs=1 if smoke_test else epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=1 if smoke_test else 2,
                learning_rate=learning_rate,
                logging_strategy="epoch",
                save_strategy="epoch",
                evaluation_strategy="epoch" if eval_dataset else "no",
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="f1" if eval_dataset else None,
                save_total_limit=3,
                report_to=["mlflow"],
                fp16=torch.cuda.is_available(),
                max_steps=5 if smoke_test else -1
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=processor.tokenizer
            )

            resume = os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0
            trainer.train(resume_from_checkpoint=resume)

            model.save_pretrained(output_model_path)
            processor.save_pretrained(output_model_path)

            mlflow.pytorch.log_model(model, "model")
            model_version = run.info.run_id
            mlflow.set_tag("model_version", model_version)

            return (model_version, output_model_path)

    except Exception as e:
        logger.error(f"Fine-tuning failed: {str(e)}")
        raise


@dsl.component(
    base_image='pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime',
    packages_to_install=['mlflow==2.10.0', 'transformers==4.37.0', 'datasets', 'scikit-learn']
)
def evaluate_model(
        model_path: InputPath(str),
        dataset_path: InputPath(str),
        metrics_path: OutputPath(str),
        num_labels: int,
        mlflow_tracking_uri: str
):
    import json
    import logging
    import torch
    import mlflow
    from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
    from datasets import load_from_disk
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
    import numpy as np

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

        dataset = load_from_disk(dataset_path)
        if 'test' not in dataset:
            raise ValueError("Dataset does not contain a 'test' split for evaluation.")
        test_dataset = dataset['test']

        model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        processor = LayoutLMv3Processor.from_pretrained(model_path)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for sample in test_dataset:
                encoding = processor(
                    sample['image'],
                    sample['words'],
                    boxes=sample['boxes'],
                    return_tensors='pt',
                    truncation=True,
                    padding="max_length"
                )

                encoding = {k: v.to(device) for k, v in encoding.items()}

                outputs = model(**encoding)
                predictions = outputs.logits.argmax(-1).reshape(-1).tolist()
                labels = sample['labels']

                for pred, label in zip(predictions, labels):
                    if label != -100:
                        all_predictions.append(pred)
                        all_labels.append(label)

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)

        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='weighted',
            zero_division=0
        )

        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            labels=list(range(num_labels)),
            average=None,
            zero_division=0
        )

        conf_matrix = confusion_matrix(all_labels, all_predictions, labels=list(range(num_labels)))

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'confusion_matrix': conf_matrix.tolist(),
            'per_class_metrics': {
                f'class_{i}': {
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i]),
                    'f1': float(per_class_f1[i]),
                    'support': int(support[i])
                }
                for i in range(num_labels)
            }
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_metrics({
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1
        })

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


@dsl.pipeline(
    name='LayoutLM Fine-tuning Pipeline',
    description='Robust Fine-tune LayoutLM for document structure extraction'
)
def layoutlm_training_pipeline(
    dataset_path: str = '/data/training/legal_financial_docs.json',
    model_name: str = 'microsoft/layoutlmv3-base',
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    experiment_name: str = 'layoutlm-finetuning',
    mlflow_tracking_uri: str = 'http://mlflow:5000',
    num_labels: int = 15,
    min_accuracy: float = 0.90,
    smoke_test: bool = False
):
    load_task = load_dataset(
        dataset_path=dataset_path,
        smoke_test=smoke_test
    ).set_cpu_request('2').set_memory_request('8G').set_caching_options(True)

    finetune_task = finetune_layoutlm(
        dataset_path=load_task.outputs['output_dataset_path'],
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        experiment_name=experiment_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        num_labels=num_labels,
        smoke_test=smoke_test
    ).set_cpu_request('4').set_memory_request('16G').set_accelerator_type('NVIDIA_TESLA_T4').set_gpu_limit(1)

    eval_task = evaluate_model(
        model_path=finetune_task.outputs['model_path'],
        dataset_path=load_task.outputs['output_dataset_path'],
        num_labels=num_labels,
        mlflow_tracking_uri=mlflow_tracking_uri
    ).set_cpu_request('2').set_memory_request('8G').set_accelerator_type('NVIDIA_TESLA_T4').set_gpu_limit(
        1).set_caching_options(True)

    register_model(
        model_version=finetune_task.outputs['model_version'],
        metrics_path=eval_task.outputs['metrics_path'],
        experiment_name=experiment_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        min_accuracy=min_accuracy
    ).set_cpu_request('1').set_memory_request('2G')

if __name__ == '__main__':
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=layoutlm_training_pipeline,
        package_path='layoutlm_pipeline.yaml'
    )