import os
import zipfile
from pathlib import Path
from tqdm import tqdm


def extract_files(set_file, output_folder, zip_path):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the file paths from the set file
    with open(set_file, "r") as f:
        file_paths = f.read().splitlines()

    total_files = len(file_paths)

    # Process each file path with tqdm progress bar
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for index, file_path in enumerate(tqdm(file_paths, desc=f"Extracting {output_folder} set", unit="file")):
            # Ensure the internal path includes the "ModelNet40" folder
            internal_path = file_path

            # Extract the category and filename
            _, category, _, filename = internal_path.split("/")

            # Create the new filename
            new_filename = f"{category}_{filename}"

            # Full path for the output file
            output_path = os.path.join(output_folder, new_filename)

            # Check if the file already exists
            if os.path.exists(output_path):
                tqdm.write(f"Skipping existing file ({index + 1}/{total_files}): {output_path}")
                continue

            try:
                # Extract the file from the zip archive
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
            except KeyError as e:
                tqdm.write(f"Error extracting file ({index + 1}/{total_files}): {e}")


def main():
    zip_path = "ModelNet40.zip"  # Specify the correct ZIP file name

    # Extract training set
    extract_files("train_set.txt", "train", zip_path)

    # Extract evaluation set
    extract_files("eval_set.txt", "eval", zip_path)

    # Extract test set
    extract_files("test_set.txt", "test", zip_path)


if __name__ == "__main__":
    main()
