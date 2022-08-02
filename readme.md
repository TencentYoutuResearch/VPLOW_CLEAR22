![CLEAR](https://clear-benchmark.github.io/img/examples_new.png)

# [CLEAR](https://www.aicrowd.com/challenges/cvpr-2022-clear-challenge/) | [Starter Kit](https://gitlab.aicrowd.com/Geniussh/clear-starter-kit/) 

This repository is the CLEAR Challenge **1st place methods** for [CVPR 2022 Workshop on Visual Perception and Learning in an Open World](https://www.cs.cmu.edu/~shuk/vplow.html)! Clone the repository to compete now!


# Installation

To set up the environment, you will need [Ananconda](https://www.anaconda.com) and  [timm](https://github.com/rwightman/pytorch-image-models) installed on your system. Then you may follow the following steps:

1.  **Clone the repository**
    ```
    git clone https://github.com/TencentYoutuResearch/cvpr22_vplow_clear.git
    cd cvpr22_vplow_clear
    ```

2. **Install** the most recent PyTorch version from their [official website](https://pytorch.org/get-started/). Make sure to specify a CUDA version for GPU usage. For example:
    ```
    pip install -r requirements.txt
    ```

3. **Clone** the master branch of [Avalanche](https://avalanche.continualai.org) and update conda env via:
    ```
    git clone https://github.com/ContinualAI/avalanche.git
    cd avalanche & pip3 install -e .
    ```

4. **Modify** [config.py](./config.py) with respect to the task you are submitting to. Change ```AVALANCHE_PATH``` to your local path to avalanche library. And change ```MODEL_ROOT```, ```LOG_ROOT``` and ```TENSORBOARD_ROOT``` to local path to save your models, logs and tensorboards. ```DATASET_NAME``` is to specify which dataset you will use to train and test your models, either ```clear10``` or ```clear100_cvpr2022```. ```ROOT``` will be the local path to save the corresponding dataset.  

5. **Training** Start training the baseline models via running:
    ```
    python starter_code.py
    ```

# Downloads
The train and test datasets can be downloaded from the following links:
* [clear10 train dataset](https://clear-challenge.s3.us-east-2.amazonaws.com/clear10-train-image-only.zip)
* [clear10 test dataset](https://clear-challenge.s3.us-east-2.amazonaws.com/clear10-test.zip)
* [clear100 train dataset](https://clear-challenge.s3.us-east-2.amazonaws.com/clear100-train-image-only.zip)
* [clear100 test dataset](https://clear-challenge.s3.us-east-2.amazonaws.com/clear100-test.zip)

# Evaluation 

We require that you place your 10 trained models in `models` directory and use the interface defined in `evaluation_setup.py`. In `evaluation_setup.py`, you need to explicitly provide the two following functions so that we can evaluate your models and auto-generated your scores on our end. 
- `load_models(models_path)` takes in the path to the 10 trained models, i.e. `models/`, and it should return a list of loaded models. 
- `data_transform()` describes the data transform you used to test your 10 models. 

To validate your `evaluation_setup.py` as well as your models, run `python local_evaluation.py` by passing in the path to the dataset, which should direct to your downloaded dataset, i.e. `<config.ROOT>/<config.DATASET_NAME>/labeled_images`. 
```
python local_evaluation.py --dataset-path <config.ROOT>/<config.DATASET_NAME>/labeled_images --resume ./models --num_classes <config.NUM_CLASSES[config.DATASET_NAME]>
```
It would print a weighted average score, which will be used for your ranking on the leaderboard, four scores corresponding to the four metrics as described above, and a visualization of the accuracy matrix in `accuracy_matrix.png`. 

# Result

clear10 and clear100 test dataset

| Dataset | Weighted Average Score | Next-Domain | In-Domain | Backward Transfer | Forward Transfer | 
| :----: | :----: | :----: | :----: | :----: | :----: |
| clear10 | 0.927 | 0.925 | 0.934 | 0.942 | 0.909 |
| clear100 | 0.915 | 0.913 | 0.920 | 0.934 | 0.892 |


# Structure

Please follow the example structure as it is in the repository for the code structure.
The different files and directories have following meaning:

```
.
├── aicrowd.json           # Submission meta information - like your username
├── evaluation_utils/      # Directory containing helper scripts for evaluation (DO NOT EDIT)
├── requirements.txt       # Python packages to be installed
├── local_evaluation.py    # Helper script for local evaluations
└── evaluation_setup.py    # IMPORTANT: Add your data transform and model loading functions that are consistent with your trained models
└── starter_code.py        # Example model training script using Avalanche on CLEAR benchmark
└── config.py              # Configuration file for Avalanche library
└── submit.sh              # Helper script for submission
```

# Acknowledgements

Our code is heavily built upon [clear-starter-kit](https://gitlab.aicrowd.com/Geniussh/clear-starter-kit/)

# Licence

This project is licensed under the Apache License - see the [LICENSE](License_CLEAR.txt) file for details.

# 📎 Important links

💪 &nbsp;Challenge Page: https://www.aicrowd.com/challenges/cvpr-2022-clear-challenge

🗣️ &nbsp;Discussion Forum: https://www.aicrowd.com/challenges/cvpr-2022-clear-challenge/discussion

🏆 &nbsp;Leaderboard: https://www.aicrowd.com/challenges/cvpr-2022-clear-challenge/leaderboards
