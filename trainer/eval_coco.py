import sys
sys.path.append('../')
import os

import torch
import torch.utils.data
from torch import nn

from toolkits.coco_utils import get_coco, get_coco_kp, get_fewshot_coco
from toolkits.voc_utils import get_voc

from toolkits.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from toolkits.engine import train_one_epoch, voc_evaluate, coco_evaluate

from toolkits import utils
import toolkits.transforms as T

from model_zoo.baseline_model import FewshotBaseline

def get_dataset(name, image_set, transform, data_path):
    p, ds_fn, num_classes = data_path, get_coco, 91
    ds = ds_fn(p, image_set=image_set, transforms=transform)
    return ds, num_classes

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # Use  CUDA_AVAILABLE_DEVICES=0 to control which device to use
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'using device {device}')
    # Data loading code
    print("Loading data")
    dataset_test, num_classes = get_dataset("coco", "minival_novel", get_transform(train=False), os.path.join('..','coco'))

    print("Creating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=8,
        collate_fn=utils.collate_fn)

    print(next(iter(data_loader_test)))
    print("Creating model")
    model = FewshotBaseline()

    pretrain = os.path.join('..','checkpoints_coco','model_finetune2_99.pth')

    if pretrain != '':
        print(f'loading model from {pretrain}')
        checkpoint = torch.load(pretrain, map_location='cpu')
        #model.load(checkpoint['model'])
        model.load_state_dict(checkpoint['model'])

    #model.init_class(novel_cls)

    model.to(device)
    coco_evaluate(model, data_loader_test, device=device)


if __name__ == "__main__":
    main()
