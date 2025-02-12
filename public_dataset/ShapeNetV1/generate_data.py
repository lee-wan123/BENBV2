import os
import zipfile
from pathlib import Path
from tqdm import tqdm


def extract_files(set_file, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the file paths from the set file
    with open(set_file, "r") as f:
        file_paths = f.read().splitlines()

    total_files = len(file_paths)

    # Process each file path with tqdm progress bar
    for index, file_path in enumerate(tqdm(file_paths, desc=f"Extracting {output_folder} set", unit="file")):
        # Split the path into components
        zip_file, _, internal_path = file_path.partition("/")

        # Extract the folder name (which is the unique identifier)
        folder_name = internal_path.split("/")[1]

        # Create the new filename
        new_filename = f"{folder_name}_model.obj"

        # Full path for the output file
        output_path = os.path.join(output_folder, new_filename)

        # Check if the file already exists
        if os.path.exists(output_path):
            tqdm.write(f"Skipping existing file ({index + 1}/{total_files}): {output_path}")
            continue

        # Extract the file from the zip archive
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extract(internal_path, path=output_folder)

        # Rename and move the extracted file
        extracted_file = os.path.join(output_folder, internal_path)
        os.rename(extracted_file, output_path)

        # Remove the empty directories created during extraction
        temp_dir = os.path.dirname(extracted_file)
        while temp_dir != output_folder:
            try:
                os.rmdir(temp_dir)
                temp_dir = os.path.dirname(temp_dir)
            except OSError:
                break  # Stop if the directory is not empty

        tqdm.write(f"Extracted ({index + 1}/{total_files}): {output_path}")


def main():
    # Extract training set
    extract_files("train_set.txt", "train")

    # Extract evaluation set
    extract_files("eval_set.txt", "eval")

    # Extract evaluation set
    extract_files("test_set.txt", "test")


if __name__ == "__main__":
    main()
