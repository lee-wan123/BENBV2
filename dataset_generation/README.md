# Dataset Generation

## Public Dataset

There are some public datasets under ``./public_dataset``, which is necessary for the project. 

## Usage

We used the Pybullet to simulate the robotic scanning process where also includes the traditional NBV-selection methods such as Random boundary, random sphere, random uniform sphere, PC-NBV and SEE.

To run the code, you can use the following command:

```shell
python main.py --mn 0 --sc 3 -n 0
```

where the arguments are:

``` shell
--method_name {0,1,2,3,4,5,6} 
    Specify the method name index (default: 0). Choices are: 
    0: Ours, 1: Random_boundary, 
    2: Random_sphere, 3: Random_uniform_sphere, 
    4: PC-NBV, 5: SEE, 6: Ours_DL
--simulation_count SIMULATION_COUNT
    Simulation count for evaluating the whole given dataset (default: 10)
--noise ignored argument for now, take it as 0 as default
```

In addition, you have to specify the dataset in main.py as:

```python
dataset_name = "Stanford3D"
# dataset_name = "ModelNet40"
# dataset_name = "ShapeNetV1"
```

The results are stored in the `./output/{dataset_name}` folder, where `frame_history` folder contains the NBV information, and you can visualize the NBV selection process by running:

```shell
python show_views.py
```

which will show the all NBV results.

## Training Dataset Generation

Do not forget the dataset generated from training later on. The data in `frame_history` is used for training the model.

## Link

The trainable dataset is available at [huggingface - datasets/Leihui/NBV](https://huggingface.co/datasets/Leihui/NBV/tree/main). If you do not want to generate the dataset by yourself, you can download it from the link.