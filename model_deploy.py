import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# 1. Initialize sessions
boto_session = boto3.Session()
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::937387180258:role/service-role/SageMaker-ExecutionRole-20260818T000579"

# approval_status - PendingManualApproval
# 2. Define the Model Package ARN from your Registry
MODEL_PACKAGE_ARN = "arn:aws:sagemaker:ap-south-1:937387180258:model-package/IrisClassifierGroup/3"

# 3. Query the Model Registry via boto3 to extract the hidden container and artifact parameters
sm_client = boto_session.client("sagemaker")
model_package_details = sm_client.describe_model_package(ModelPackageName=MODEL_PACKAGE_ARN)

approval_status = model_package_details['ModelApprovalStatus']
print (f'approval status :: {approval_status}')
# Extract the S3 location of the packaged model.tar.gz file
model_data_url = model_package_details["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]

# Extract the Docker Inference Image URI used during training/registration
inference_image_uri = model_package_details["InferenceSpecification"]["Containers"][0]["Image"]

print(f" Extracted S3 URL: {model_data_url}")
print(f" Extracted Container Image: {inference_image_uri}")

# 4. Pass the extracted data directly into SKLearnModel
registered_sklearn_model = SKLearnModel(
    model_data=model_data_url,          # Points to the registry's S3 asset path
    image_uri=inference_image_uri,      # Uses the exact container image registered with the model
    role=role,
    entry_point="inference.py",          # Allows you to provide a fresh or modified local inference script
    framework_version="1.2-1",
    sagemaker_session=sagemaker_session
)

# 5. Deploy the model to a live real-time endpoint
if approval_status == 'Approved':
    predictor = registered_sklearn_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large"
    )
    print(f"\n Endpoint deployed successfully: {predictor.endpoint_name}")
else:
    print("We can't deploy as the model is not approved yet!")
