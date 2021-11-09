from torchvision.datasets import VOCDetection
voc07val = VOCDetection('voc', year = '2007', image_set = 'trainval', download=True)

