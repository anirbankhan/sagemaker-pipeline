import boto3
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.functions import Join




# -------------------------------------------------------------
# LOCAL MACHINE CONNECTIVITY CONFIGURATION
# -------------------------------------------------------------
boto_session = boto3.Session(profile_name="default")
aws_region = boto_session.region_name

pipeline_session = PipelineSession(boto_session=boto_session)

role = "arn:aws:iam::937387180258:role/service-role/SageMaker-ExecutionRole-20260824T234194"

instance_type = "ml.c4.xlarge"
framework_version = "1.2-1"

# -------------------------------------------------------------
# Step 1. Processing Step CONFIGURATION
# -------------------------------------------------------------

sklearn_processor = SKLearnProcessor(
    framework_version=framework_version,
    instance_type=instance_type,
    instance_count=1,
    role=role,
    sagemaker_session=pipeline_session

)

step_process = ProcessingStep(
    name="PreprocessSKLearnData",
    processor=sklearn_processor,
    code="preprocess.py",
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train/"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
    ]
)

# -------------------------------------------------------------
# Step 2. Training Step CONFIGURATION
# -------------------------------------------------------------

sklearn_estimator = SKLearn(
    entry_point="train.py",
    framework_version=framework_version,
    instance_type=instance_type,
    instance_count=1,
    role=role,
    sagemaker_session=pipeline_session,
    hyperparameters={"n_estimators": 150}
)



step_train = TrainingStep(
    name="TrainSklearnModel",
    estimator=sklearn_estimator,
    inputs={
        "train": sagemaker.inputs.TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        ),
        "test": sagemaker.inputs.TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            content_type="text/csv"
        ),
    }
)

# -------------------------------------------------------------
# Step 3. Evaluation metrics extraction and parsing
# -------------------------------------------------------------

eval_extractor = SKLearnProcessor(
    framework_version = framework_version,
    instance_type = instance_type,
    instance_count = 1,
    role = role,
    sagemaker_session = pipeline_session
)

evaluation_report = PropertyFile(
    name="TrainingEvaluationReport",
    output_name="evaluation",
    path="evaluation.json"
)

step_eval_parse = ProcessingStep(
    name="ExtractAndParseMetrics",
    processor=eval_extractor,
    code="evaluate.py",
    inputs=[
        ProcessingInput(
            # source=step_train.properties.ModelArtifacts.S3ModelArtifacts.replace("model.tar.gz", "output.tar.gz"),
                source=Join(
                on="/",
                values=[
                    f"s3://{pipeline_session.default_bucket()}",
                    step_train.properties.TrainingJobName,
                    "output",
                    "output.tar.gz"
                ]
            ),
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
    ],
    property_files=[evaluation_report]
)




# -------------------------------------------------------------
# STEP 3C: Model Creation Step Configuration - Optional
# -------------------------------------------------------------

# model = Model (
#     image_uri = sagemaker.image_uris.retrieve(
#         framework = "sklearn", region=aws_region, version=framework_version, image_scope="inference"
#     ),
#     model_data = step_train.properties.ModelArtifacts.S3ModelArtifacts,
#     role=role,
#     sagemaker_session=pipeline_session
# )

# step_model_create = ModelStep(
#     name="CreateSklearnModel",
#     step_args=model.create(instance_type=instance_type)
# )





# -------------------------------------------------------------
# Model Register Step Configuration based on quality gate steps condition
# -------------------------------------------------------------
pipeline_model = SKLearnModel(
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    entry_point="inference.py",
    framework_version="1.2-1",
    sagemaker_session=pipeline_session
)

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=Join(
            on="/",
            values=[
                step_eval_parse.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                "evaluation.json"
            ]
        ),
        content_type="application/json"
    )
)

step_register_model = ModelStep(
    name="RegisterNewSklearnVersion",
    step_args=pipeline_model.register(
        model_package_group_name="IrisClassifierGroup", # Group container family name
        approval_status="PendingManualApproval",        # Default protective safety gate status
        model_metrics=model_metrics,
        content_types=["application/json"],
        response_types=["application/json"]
    )
)

# -------------------------------------------------------------
# Step 4. Conditional Model registry based on metrics
# -------------------------------------------------------------
condition_accuracy = ConditionGreaterThanOrEqualTo(
    left=JsonGet(
        step_name=step_eval_parse.name,
        property_file=evaluation_report,
        json_path="classification_metrics.accuracy.value"
    ),
    right=0.85  # Target Quality Gate: 85% Minimum Accuracy
)
step_conditional_gate = ConditionStep(
    name="AccuracyThresholdGate",
    conditions=[condition_accuracy],
    if_steps=[step_register_model],
    else_steps=[]
)





# -------------------------------------------------------------
# PIPELINE CONFIGURATION
# -------------------------------------------------------------

pipeline = Pipeline(
    name="FirstPipeline",
    # steps=[step_process, step_train, step_register_model],
    steps=[step_process, step_train, step_eval_parse, step_conditional_gate],
    sagemaker_session=pipeline_session
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
print(f"Pipeline execution started : {execution.arn}")
execution.wait()

# Extract the auto-generated model asset URI from completed execution
# steps = execution.list_steps()
# training_job_name = [s for s in steps if s['StepName'] == 'TrainSklearnModel'][0]['Metadata']['TrainingJob']['Arn'].split('/')[-1]
# bucket = pipeline_session.default_bucket()
# model_s3_url = f"s3://{bucket}/{training_job_name}/output/model.tar.gz"

# # Deploy Endpoint
# final_model = Model(
#     image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=aws_region, version=framework_version, image_scope="inference"),
#     model_data=model_s3_url,
#     role=role,
#     sagemaker_session=pipeline_session
# )

# predictor = final_model.deploy(initial_instance_count=1, instance_type=instance_type)
# print(f"Active dynamic endpoint exposed at: {predictor.endpoint_name}")

# # Predict on test features (sepal/petal rows)
# predictions = predictor.predict([[5.1, 3.5, 1.4, 0.2]])
# print(f"Model Prediction: {predictions}")

# # Cleanup infrastructure to stop charges
# predictor.delete_endpoint()




