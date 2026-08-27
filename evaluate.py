import os
import tarfile
import shutil


if __name__ == "__main__":
    input_tar_path = "/opt/ml/processing/input/output.tar.gz"
    extract_path = "./extracted_path"
    output_dir = "/opt/ml/processing/evaluation"
    with tarfile.open(input_tar_path) as tar:
        tar.extractall(path=extract_path)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(
        os.path.join(extract_path, "evaluation.json"), 
        os.path.join(output_dir, "evaluation.json")
    )
    print("Metrics extracted successfully for step evaluation parsing")