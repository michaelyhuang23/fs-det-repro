r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        train.py ... --world-size $NGPU

"""
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco, get_coco_kp
from voc_utils import get_voc

from group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, voc_evaluate, coco_evaluate

import utils
import transforms as T


def get_dataset(name, image_set, transform, data_path):
    p, ds_fn, num_classes = data_path, get_voc, 21
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
    torch.cuda.set_device(1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'using device {device}')
    device_id = torch.cuda.current_device()
    print(f'using gpu {torch.cuda.get_device_name(device_id)}')
    # Data loading code
    print("Loading data")

    dataset, num_classes = get_dataset("voc", 'train', get_transform(train=True), 'voc')
    dataset_test, _ = get_dataset("voc", 'val', get_transform(train=False), 'voc')

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    batch_size = 4 
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=8,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=8,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes,
                                                              pretrained=False)
    checkpoint = torch.load('checkpoints_voc/model_14.pth', map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    model_without_ddp = model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.04/8/10,
                                momentum=0.9, weight_decay=0.0001)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.1)

    resume = '' # else give the file loc

    if resume:
        checkpoint = torch.load(resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    print("Start training")
    start_time = time.time()
    epochs = 10 
    #voc_evaluate(model, data_loader_test, device=device)
    print_freq = 1000
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)
        lr_scheduler.step()
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()},
            os.path.join('checkpoints_voc', 'model_retrain_Nov1_{}.pth'.format(epoch)))

        voc_evaluate(model, data_loader_test, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    main()
