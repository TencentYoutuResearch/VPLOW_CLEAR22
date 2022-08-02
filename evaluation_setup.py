import torch
import torchvision.models as models
import torchvision.transforms as transforms
from evaluation_utils.models import BaseModel
import os
import math
from timm.models import create_model
from pdb import set_trace as stop


def load_models(models_path, num_classes=100):
    model_files = os.listdir(models_path)
    model_files = [os.path.join(models_path, ff) for ff in model_files if (ff.endswith('.pth') or ff.endswith('.pt'))]
    model_files.sort()
    if len(model_files) == 11:
        model_files = model_files[1:]
    assert len(model_files) == 10

    loaded_models = [None] * 10    
    for i in range(10):
        loaded_models[i] = create_model('resnet50d', pretrained=False, num_classes=num_classes)
        loaded_models[i].load_state_dict(torch.load(model_files[i]))
        print(model_files[i])

    new_models = [None] * 10
    for i in range(10):
        new_models[i] = BaseModel(loaded_models[i])
    
    return new_models
    # return loaded_models
    
# 不添加任何后处理
# def data_transform():
#    # Data Loader
#    IMAGENET_MEAN = [0.485, 0.456, 0.406]
#    IMAGENET_STD = [0.229, 0.224, 0.225]
#    transform = transforms.Compose([
#        transforms.Resize(224),
#        transforms.CenterCrop(224), 
#        transforms.ToTensor(),            
#        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
#    ])

#    return transform


# 添加fivecrop及模型尺寸后处理
def data_transform():
    # Data Loader
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    img_size = 256                          # 修改image size 尺寸
    crop_pct = 0.875
    scale_size = int(math.floor(img_size / crop_pct))
    tfl = [
        transforms.Resize(scale_size),
        transforms.FiveCrop(img_size),
    ]
    tfl += [
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ]

    transform = transforms.Compose(tfl)

    return transform
