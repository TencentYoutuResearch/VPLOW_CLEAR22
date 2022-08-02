'''Example: Training and evaluating on CLEAR benchmark (RGB images)
   The basic parameters are in the config.py.
   FILE_NAME can be modified according to the setting parameters.
'''
import json
from pathlib import Path
import copy
import sys
import config
import time
# IMPORTANT! Need to add avalanche to sys path
if config.AVALANCHE_PATH:
    print(f"Importing avalanche library path {config.AVALANCHE_PATH} to sys.path")
    sys.path.append(config.AVALANCHE_PATH)
else:
    print("Please specify avalanche library path in config.py")
    exit(0)

import numpy as np
import torch
import torchvision
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    timing_metrics,
    cpu_usage_metrics,
    confusion_matrix_metrics,
    disk_usage_metrics,
)
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins.lr_scheduling import LRSchedulerPlugin
from avalanche.training.supervised import Naive
from avalanche.benchmarks.classic.clear import CLEAR, CLEARMetric
from timm.data.transforms_factory import create_transform
from timm.models import create_model
from timm.data import Mixup
from timm.loss.cross_entropy import SoftTargetCrossEntropy
from timm.loss import LabelSmoothingCrossEntropy
from train_utils.dataset import get_cumulative_dataset, get_dataset, get_bucketsample_dataset
from train_utils.optimizer import get_optimizer
from train_utils.scheduler import make_cosine_scheduler, make_scheduler, adjust_learning_rate
from pdb import set_trace as stop

def build_model(num_classes):

    if config.model == "resnet18":
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    elif config.model == "resnet50":
        model = torchvision.models.resnet50(pretrained=False, num_classes=num_classes)
    elif config.model == "resnet50d":
        # use timm framework to create model
        model = create_model(
            'resnet50d',
            pretrained=False,
            num_classes=num_classes,
        )
    return model


def build_transform():
    # data argument
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    normalize = torchvision.transforms.Normalize(
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
    )

    train_transform = create_transform(
        input_size=config.input_size,
        is_training=True,
        ratio=config.ratio,
        hflip=config.hflip,
        vflip=config.vflip,
        color_jitter=config.color_jitter,
        auto_augment=config.auto_augment,
        interpolation=config.interpolation,
        re_prob=config.re_prob,
        re_mode=config.re_mode,
        re_count=config.re_count,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    )

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(config.input_size),
            torchvision.transforms.CenterCrop(config.input_size),
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, test_transform


def train(index, loader, model, optimizer, criterion, mixup_fn=None):
    '''Navie train strategy
    Args:
        index: index value of bucket
        loader: dataloader for train
        model: model containing parameters to train
        optimizer: optimizer
        criterion: criterion to compute loss
        mixup_fn: name of mixup function(default None)
    '''
    # print("Epochs: ", nepoch)
    for epoch in range(config.num_epochs[index]):
        acc_ = 0
        if config.mixup_off_epoch and epoch >= config.mixup_off_epoch and mixup_fn is not None:
            mixup_fn.mixup_enabled = False
        for iter, data in enumerate(loader):

            adjust_learning_rate(optimizer=optimizer, epoch=iter / len(loader) + epoch, index=index)

            input, target, _ = data
            optimizer.zero_grad()
            input = input.cuda()
            target = target.cuda()
            if mixup_fn is not None:
                # print(f'Using mixup or cutmix for training')
                input, target = mixup_fn(input, target)
            pred = model(input)
            loss = criterion(pred, target)
            loss.backward()
            acc_ += (torch.sum(torch.eq(torch.max(pred, 1)[1], 
                               target)) / len(pred)).item()
            optimizer.step()
            
        acc_ = acc_/len(loader)
        print(f'training accuracy for epoch {epoch} is {acc_}, data nums are {len(loader)}')
    
    return model

def get_accuracy_matrix(results):
    # generate accuracy matrix
    num_timestamp = len(results)
    accuracy_matrix = np.zeros((num_timestamp, num_timestamp))
    for train_idx in range(num_timestamp):
        for test_idx in range(num_timestamp):
            accuracy_matrix[train_idx][test_idx] = results[train_idx][
                f"Top1_Acc_Stream/eval_phase/test_stream/Task{str(test_idx).zfill(3)}"]
    print('Accuracy_matrix : ')
    print(accuracy_matrix)
    metric = CLEARMetric().get_metrics(accuracy_matrix)
    print(metric)

    return accuracy_matrix, metric


def main():

    # For CLEAR dataset setup
    print(
        f"This script will train on {config.DATASET_NAME}. "
        "You may change the dataset in config.py."
    )
    DATASET_NAME = config.DATASET_NAME
    # NUM_CLASSES = {"clear10": 11}
    NUM_CLASSES = config.NUM_CLASSES

    EVALUATION_PROTOCOL = "streaming"  # trainset = testset per timestamp

    # Paths for saving datasets/models/results/log files
    print(
        f"The dataset/models will be saved at {Path(config.ROOT).resolve()}. "
        f"You may change this path in config.py."
    )

    # local path to load dataset and save model, log, tensorboard
    ROOT = Path(config.ROOT)
    MODEL_ROOT = Path(config.MODEL_ROOT)
    LOG_ROOT = Path(config.LOG_ROOT)
    TENSORBOARD_ROOT = Path(config.TENSORBOARD_ROOT)

    FILE_NAME = "r50d_n11_b64_decay_smooth_uniformrepeat_b4_argu_wp5" # create your files name
    DATA_ROOT = ROOT / DATASET_NAME
    MODEL_PATH = MODEL_ROOT / FILE_NAME  
    LOG_PATH = LOG_ROOT / FILE_NAME
    TENSORBOARD_PATH = TENSORBOARD_ROOT / FILE_NAME

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_PATH.mkdir(parents=True, exist_ok=True)

    # model
    model = build_model(num_classes=NUM_CLASSES[DATASET_NAME])

    train_transform, test_transform = build_transform()

    mixup_fn = None
    mixup_active = config.mixup > 0 or config.cutmix > 0
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=config.mixup, cutmix_alpha=config.cutmix, prob=config.mixup_prob, 
            switch_prob=config.mixup_switch_prob, mode=config.mixup_mode,
            label_smoothing=config.smoothing, num_classes=NUM_CLASSES[DATASET_NAME]
        )
        mixup_fn = Mixup(**mixup_args)

    # log to Tensorboard
    tb_logger = TensorboardLogger(TENSORBOARD_PATH)

    # log to text file
    text_logger = TextLogger(open(LOG_PATH / "log.txt", "w+"))

    # print to stdout
    interactive_logger = InteractiveLogger()

    # use avalance framewor to create plugin and eval function
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(
            num_classes=NUM_CLASSES[DATASET_NAME], save_image=False, stream=True
        ),
        disk_usage_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loggers=[interactive_logger, text_logger, tb_logger],
    )

    if EVALUATION_PROTOCOL == "streaming":
        seed = None
    else:
        seed = 0

    scenario = CLEAR(
        data_name=DATASET_NAME,
        evaluation_protocol=EVALUATION_PROTOCOL,
        feature_type=None,
        seed=seed,
        train_transform=train_transform,
        eval_transform=test_transform,
        dataset_root=DATA_ROOT,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # optimizer function
    optimizer = get_optimizer(model, opt=config.opt)

    # continual learning train strategy of avlance framowork
    cl_strategy = Naive(
        model,
        optimizer,
        torch.nn.CrossEntropyLoss(),
        train_mb_size=config.batch_size,
        eval_mb_size=config.batch_size,
        evaluator=eval_plugin,
        device=device,
    )

    # loss function
    if mixup_active and config.loss == "ce":
        criterion = SoftTargetCrossEntropy().cuda()
    elif config.smoothing > 0 and config.loss == "smooth":
        criterion = LabelSmoothingCrossEntropy().to(device)
    elif config.loss == "ce":
        criterion = torch.nn.CrossEntropyLoss().cuda()

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    print("Current protocol : ", EVALUATION_PROTOCOL)
    for index, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        train_loader = get_bucketsample_dataset(scenario.train_stream, index, 
                            buffer_size=config.buffer_size, repeat_sample=config.repeat_sample)
        model = train(index=index, loader=train_loader, model=model, 
                      optimizer=optimizer, criterion=criterion, mixup_fn=mixup_fn)
        cl_strategy.model = copy.deepcopy(model)
        torch.save(
            model.state_dict(),
            str(MODEL_PATH / f"model{str(index).zfill(2)}.pth")
        )
        print("Training completed")
        print(
            "Computing accuracy on the whole test set with"
            f" {EVALUATION_PROTOCOL} evaluation protocol"
        )
        results.append(cl_strategy.eval(scenario.test_stream, num_workers=config.num_workers, pin_memory=False))

    accuracy_matrix, metric = get_accuracy_matrix(results)

    # save metric log
    metric_log = open(LOG_PATH / "metric_log.txt", "w+")
    metric_log.write(
        f"Protocol: {EVALUATION_PROTOCOL} "
        f"Seed: {seed} "
    )
    json.dump(accuracy_matrix.tolist(), metric_log, indent=6)
    json.dump(metric, metric_log, indent=6)
    metric_log.close()


if __name__ == '__main__':

    main()
