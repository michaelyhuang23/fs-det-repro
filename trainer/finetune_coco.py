import sys
sys.path.append('../')
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

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
    novel_cls = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
    # Use  CUDA_AVAILABLE_DEVICES=0 to control which device to use
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f'using device {device}')
    # Data loading code
    print("Loading data")
    dataset = get_fewshot_coco(os.path.join('..','coco'), "trainval2014", get_transform(train=True), seed=0, shot=30, mode='train')
    dataset_test, num_classes = get_dataset("coco", "minival_novel", get_transform(train=False), os.path.join('..','coco'))

    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    
    batch_size = 2

    train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=8,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=8,
        collate_fn=utils.collate_fn)

    print(next(iter(data_loader_test)))
    print("Creating model")
    model = FewshotBaseline()

    pretrain = os.path.join('..','checkpoints_coco','model_baseclass_22.pth')

    if pretrain != '':
        print(f'loading model from {pretrain}')
        checkpoint = torch.load(pretrain, map_location='cpu')
        model.load(checkpoint['model'])

    model.init_class(novel_cls)

    model.to(device)

    model_without_ddp = model
    coco_evaluate(model, data_loader_test, device=device)


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.02/8, momentum=0.9, weight_decay=1e-4)
    # 0.02 / 20 / 8
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16, 22], gamma=0.1)

    print("Start training")
    start_time = time.time()

    #coco_evaluate(model, data_loader_test, device=device)
    utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            },
            os.path.join('..','checkpoints', 'model_finetune_{}.pth'.format(-1)))
    epochs = 26 
    train_print_freq = 100

    for epoch in range(epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, train_print_freq)
        lr_scheduler.step()
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            },
            os.path.join('..','checkpoints', 'model_finetune_{}.pth'.format(epoch)))

        # evaluate after every epoch
        coco_evaluate(model, data_loader_test, device=device)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
