import sys
sys.path.append('../')
import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn

from toolkits.voc_utils import get_voc

from toolkits.engine import voc_evaluate

from toolkits import utils
import toolkits.transforms as T


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

    dataset_test, num_classes = get_dataset("voc", 'val', get_transform(train=False), 'voc')

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

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

    voc_evaluate(model, data_loader_test, device=device)

    print('Finish testing')


if __name__ == "__main__":
    main()
