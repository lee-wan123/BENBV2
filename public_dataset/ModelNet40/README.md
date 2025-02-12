# Usage

0. Download the dataset from [ModelNet40](https://modelnet.cs.princeton.edu/), and just put the dataset under current directory.
   The structure of the dataset should be like this:

   ``` bash
    │   eval_set.txt
    │   generate_data.py
    │   generate_split_list.py
    │   ModelNet40.zip
    │   README.md
    ```

1. Run the following command to generate data list for training, evaluation and testing:

    ```bash
    python generate_split_list.py
    ```

2. Run the following command to genereate data files in to specific folders:

    ```bash
    python generate_data.py
    ```

    Now, the structure of the dataset should be like this:

    ``` bash
    │   eval_set.txt
    │   generate_data.py
    │   generate_split_list.py
    │   ModelNet40.zip
    │   README.md
    │   train_set.txt
    │   test_set.txt
    │   eval_set.txt
    │   train # (folder)
    │   eval # (folder)
    │   test # (folder)
    ```