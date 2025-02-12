import os
import random
from zipfile import ZipFile
from collections import defaultdict


def generate_data_lists(zip_path, output_dir, train_size, eval_size, test_size):
    train_list = []
    eval_list = []
    test_list = []

    category_files = defaultdict(lambda: {"train": [], "test": []})
    category_stats = defaultdict(lambda: {"train": 0, "test": 0})

    with ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.endswith("/"):
                continue

            parts = file_info.filename.split("/")
            if len(parts) != 4:
                continue

            _, category, split, filename = parts
            category_files[category][split].append(file_info.filename)
            category_stats[category][split] += 1

    for category, splits in category_files.items():
        # Select files for training
        if len(splits["train"]) >= train_size:
            train_files = random.sample(splits["train"], train_size)
            train_list.extend(train_files)
            remaining_train = [f for f in splits["train"] if f not in train_files]
        else:
            train_list.extend(splits["train"])
            remaining_train = []
            print(f"Warning: Category {category} has less than {train_size} training samples.")

        # Select files for evaluation
        eval_from_train = min(eval_size, len(remaining_train))
        eval_from_test = eval_size - eval_from_train

        eval_list.extend(random.sample(remaining_train, eval_from_train))

        if len(splits["test"]) >= eval_from_test + test_size:
            eval_test_files = random.sample(splits["test"], eval_from_test)
            eval_list.extend(eval_test_files)
            remaining_test = [f for f in splits["test"] if f not in eval_test_files]
            test_list.extend(random.sample(remaining_test, test_size))
        else:
            print(f"Warning: Category {category} doesn't have enough samples for evaluation and testing.")
            if len(splits["test"]) > eval_from_test:
                eval_list.extend(random.sample(splits["test"], eval_from_test))
                test_list.extend([f for f in splits["test"] if f not in eval_list])
            else:
                eval_list.extend(splits["test"])

    # Write lists to files
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "train_set.txt"), "w") as f:
        f.write("\n".join(train_list))

    with open(os.path.join(output_dir, "eval_set.txt"), "w") as f:
        f.write("\n".join(eval_list))

    with open(os.path.join(output_dir, "test_set.txt"), "w") as f:
        f.write("\n".join(test_list))

    print(f"Total files: {len(train_list) + len(eval_list) + len(test_list)}")
    print(f"Train: {len(train_list)}")
    print(f"Eval: {len(eval_list)}")
    print(f"Test: {len(test_list)}")

    print("\nCategory Statistics:")
    for category, stats in category_stats.items():
        print(f"{category}:")
        print(f"  Train folder: {stats['train']} files")
        print(f"  Test folder: {stats['test']} files")


def main():
    zip_path = "./ModelNet40.zip"
    output_dir = "./"
    train_size = 20
    eval_size = 15
    test_size = 15

    generate_data_lists(zip_path, output_dir, train_size, eval_size, test_size)


if __name__ == "__main__":
    main()
