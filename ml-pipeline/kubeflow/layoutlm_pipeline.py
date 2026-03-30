from kfp import dsl

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