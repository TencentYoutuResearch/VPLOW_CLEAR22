'''
    Absolute path where you clone the avalanche library
'''
AVALANCHE_PATH = "./avalanche"


'''
    Where to store the datasetsx
'''
ROOT = "/XXX/CLEAR" # Where to store datasets

'''
    Which dataset to train on
'''
# DATASET_NAME对应clear10, clear100
# DATASET_NAME = "clear100"
DATASET_NAME = "clear10"

NUM_CLASSES = {"clear10": 11, "clear100": 100}

# xxx数据存储路径
'''
    where to save the models
'''
MODEL_ROOT = "/XXX/models"

'''
    where to save the logs
'''
LOG_ROOT = "/XXX/logs"

'''
    where to save the tensorboard
'''
TENSORBOARD_ROOT = "/XXX/tensorboard"


## parameter setting
model = "resnet50d"
batch_size = 64
num_workers = 8

# decay
# num_epochs = [200, 200, 200, 150, 150, 100, 100, 100, 100, 100] # decay1  clear10
num_epochs = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100] # decay2  clear100
# clear10 对应num_epochs是decay1，clear100对应num_epochs是decay2, 其他配置默认即可

# buffer
buffer_size = 4
repeat_sample = True 

# scheduler
scheduler_step_decay = 30
scheduler_gamma = 0.1

start_lr = 0.1 
min_lr = 1e-5
warmup_lr = 0.0001
warmup_epochs = 5
cooldown_epochs = 0
decay_ratio = 1

# optimizer
opt = "sgd"          # ["sgd", "adamw"]
weight_decay = 1e-4
momentum = 0.9

# data
input_size = 224
ratio = [3./4., 4./3.]
hflip = 0.5
vflip = 0.
color_jitter = 0.4
auto_augment = 'rand-m15-mstd0.5-n2'
interpolation = 'bicubic'
re_prob = 0
re_mode = 'const'
re_count = 1
mixup = 0.
mixup_off_epoch = 0
cutmix = 0.
mixup_prob = 1.0
mixup_switch_prob = 0.5
mixup_mode = 'batch'

# loss
loss = "smooth"      # ["ce", "smooth"]
smoothing = 0.1
epsilon = 2
