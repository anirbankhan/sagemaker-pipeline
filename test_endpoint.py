import json
import boto3
import sagemaker

# 1. Initialize the AWS runtime client
# It automatically fetches your local CLI credentials
# 1. Initialize sessions
boto_session = boto3.Session()
sagemaker_session = sagemaker.Session(boto_session=boto_session)
role = "arn:aws:iam::937387180258:role/service-role/SageMaker-ExecutionRole-20260818T000579"

# approval_status - PendingManualApproval
# 2. Define the Model Package ARN from your Registry
# MODEL_PACKAGE_ARN = "arn:aws:sagemaker:ap-south-1:937387180258:model-package/IrisClassifierGroup/3"

# 3. Query the Model Registry via boto3 to extract the hidden container and artifact parameters
sm_client = boto_session.client("sagemaker-runtime")

# 2. Configure endpoint variables
ENDPOINT_NAME = "sagemaker-scikit-learn-2026-09-12-18-23-35-921"

# 3. Formulate raw structural prediction rows (Iris feature matrix)
payload = {
    "instances": [
        [5.1, 3.5, 1.4, 0.2],  # Expected Class: 0 (Setosa)
        [6.7, 3.0, 5.2, 2.3]   # Expected Class: 2 (Virginica)
    ]
}

# 4. Invoke the live HTTPS endpoint infrastructure
print(f"Sending payload invocation request to: {ENDPOINT_NAME}")
response = sm_client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(payload)
)

# 5. Parse response stream output payload
response_body = response["Body"].read().decode("utf-8")
result = json.loads(response_body)

print(f"\nModel Prediction Response: {result}")
