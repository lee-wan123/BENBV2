import os
import zipfile
from pathlib import Path
import json
import random

json_file = "./taxonomy.json"
with open(json_file, "r") as f:
    taxonomy = json.load(f)


def syssetID_to_name(syssetID, json=taxonomy):
    for item in json:
        if item["synsetId"] == syssetID:
            return item["name"].split(",")[0]
    return None


def find_model_obj_files():
    current_dir = Path.cwd()
    model_obj_paths = {}

    for zip_file in current_dir.glob("*.zip"):
        model_file = 0
        obj_name = syssetID_to_name(zip_file.name[:-4])

        if obj_name not in model_obj_paths:
            model_obj_paths[obj_name] = []

        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                if file_info.filename.lower().endswith("model.obj"):
                    model_obj_paths[obj_name].append(f"{zip_file.name}/{file_info.filename}")
                    model_file += 1

            print(f"{zip_file.name}({obj_name}) -> {model_file} / {len(zip_ref.infolist())}")

    return model_obj_paths


def split_train_eval_test(model_obj_paths, train_num=60, eval_num=20, test_num=20):
    train_set = []
    eval_set = []
    test_set = []

    for obj_class, paths in model_obj_paths.items():
        random.shuffle(paths)
        total_samples = len(paths)

        # Calculate the actual numbers for each set
        actual_train = min(train_num, total_samples)
        actual_eval = min(eval_num, total_samples - actual_train)
        actual_test = min(test_num, total_samples - actual_train - actual_eval)

        train_set.extend(paths[:actual_train])
        eval_set.extend(paths[actual_train : actual_train + actual_eval])
        test_set.extend(paths[actual_train + actual_eval : actual_train + actual_eval + actual_test])

        print(f"{obj_class}: Total {total_samples}, Train {actual_train}, Eval {actual_eval}, Test {actual_test}")

    return train_set, eval_set, test_set


if __name__ == "__main__":
    result = find_model_obj_files()
    train_set, eval_set, test_set = split_train_eval_test(result)

    total_files = sum(len(paths) for paths in result.values())
    print(f"\nTotal model.obj files: {total_files}")
    print(f"Training set size: {len(train_set)} ({len(train_set)/total_files:.2%})")
    print(f"Evaluation set size: {len(eval_set)} ({len(eval_set)/total_files:.2%})")
    print(f"Test set size: {len(test_set)} ({len(test_set)/total_files:.2%})")

    # Save the sets to files
    with open("train_set.txt", "w") as f:
        f.write("\n".join(train_set))

    with open("eval_set.txt", "w") as f:
        f.write("\n".join(eval_set))

    with open("test_set.txt", "w") as f:
        f.write("\n".join(test_set))
