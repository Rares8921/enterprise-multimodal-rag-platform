from kfp import dsl
from kfp.dsl import InputPath, OutputPath
from typing import NamedTuple


KFP_BASE_IMAGE_CPU = "multimodal-doc-mlops/kfp-layoutlm-cpu:latest"
KFP_BASE_IMAGE_GPU = "multimodal-doc-mlops/kfp-layoutlm-gpu:latest"


@dsl.component(
    base_image=KFP_BASE_IMAGE_CPU
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
        
        from kubeflow.layoutlm_logic import validate_dataset_json_schema, subset_dataset_json_for_smoke_test

        if not isinstance(data, dict):
            raise ValueError("Dataset must be a dict")

        # Shared business-logic validation
        schema_kind = validate_dataset_json_schema(data)

        if smoke_test:
            logger.info("Smoke test enabled. Subsetting dataset.")
            data = subset_dataset_json_for_smoke_test(data, max_samples=10)

        # Convert to HF datasets after validation/subsetting
        if schema_kind == 'split':
            dataset = DatasetDict({
                split: Dataset.from_dict(split_data)
                for split, split_data in data.items()
            })
        else:
            dataset = Dataset.from_dict(data)

        dataset.save_to_disk(output_dataset_path)
        logger.info(f"Dataset saved to {output_dataset_path}")

    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise


@dsl.component(
    base_image=KFP_BASE_IMAGE_GPU
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
    from pathlib import Path
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
            
            from kubeflow.layoutlm_logic import build_training_args, should_resume_from_checkpoint, validate_hyperparameters

            params_ok = validate_hyperparameters({
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'num_labels': num_labels,
            })
            if not params_ok:
                raise ValueError("Invalid hyperparameters")

            training_args_kwargs = build_training_args(
                smoke_test=smoke_test,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                has_eval_dataset=bool(eval_dataset),
                cuda_available=torch.cuda.is_available(),
            )

            training_args = TrainingArguments(
                output_dir=checkpoint_dir,
                **training_args_kwargs,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=processor.tokenizer
            )
            
            resume = should_resume_from_checkpoint(Path(checkpoint_dir))
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
    base_image=KFP_BASE_IMAGE_GPU
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


@dsl.component(
    base_image=KFP_BASE_IMAGE_CPU
)
def register_model(
    model_version: str,
    metrics_path: InputPath(str),
    experiment_name: str,
    mlflow_tracking_uri: str,
    min_accuracy: float
) -> str:
    import json
    import logging
    from pathlib import Path
    import mlflow
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        from kubeflow.layoutlm_logic import (
            build_model_tags,
            build_model_uri,
            build_registered_model_name,
            determine_registration_status,
            get_accuracy_from_metrics,
            load_metrics_from_file,
        )

        model_metrics = load_metrics_from_file(Path(metrics_path))
        accuracy = get_accuracy_from_metrics(model_metrics)

        status = determine_registration_status(accuracy, min_accuracy)
        if status == "registered":
            model_uri = build_model_uri(model_version)
            mlflow.register_model(
                model_uri,
                name=build_registered_model_name(experiment_name),
                tags=build_model_tags(accuracy, "approved"),
            )
        
        return status

    except Exception as e:
        logger.error(f"Model registration failed: {str(e)}")
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
    ).set_cpu_request('2').set_memory_request('8G').set_accelerator_type('NVIDIA_TESLA_T4').set_gpu_limit(1).set_caching_options(True)
    
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