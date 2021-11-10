import sys
sys.path.append('../')
import torch
import math
import torchvision
from torch import nn
from torch.nn.init import uniform_
from collections import OrderedDict
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class Predictor(nn.Module):
	# this is a modification of pytorch's implementation
    def __init__(self, faster_predictor : FastRCNNPredictor):
    	# in_channels, num_classes must be the same as faster_predictor
        super(Predictor, self).__init__()
        self.cls_score = faster_predictor.cls_score
        self.inchannel = self.cls_score.in_features
        self.num_classes = self.cls_score.out_features
        self.bbox_pred = faster_predictor.bbox_pred

    def init_func(data, bias = None):
        stdv = 1. / math.sqrt(data.size(1)) # first dim is output dim
        data.uniform_(-stdv, stdv)
        if bias is not None:
            bias.uniform_(-stdv, stdv)

    def init_class(self, classes):
        for c in classes:
            if self.cls_score.bias is not None:
                Predictor.init_func(self.cls_score.weight.data[c], self.cls_score.bias.data[c])
            else:
                Predictor.init_func(self.cls_score.weight.data[c])
            if self.bbox_pred.bias is not None:
                Predictor.init_func(self.bbox_pred.weight.data[c*4 : (c+1)*4], self.bbox_pred.bias.data[c*4 : (c+1)*4])
            else:
                Predictor.init_func(self.bbox_pred.weight.data[c*4 : (c+1)*4])

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class FewshotBaseline(nn.Module):
    def __init__(self):
        super(FewshotBaseline, self).__init__()
        self.fasterRCNN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)
        self.fasterRCNN.roi_heads.box_predictor = Predictor(self.fasterRCNN.roi_heads.box_predictor) 
        for param in self.fasterRCNN.parameters():
            param.requires_grad = False
        for param in self.fasterRCNN.roi_heads.box_predictor.parameters():
            param.requires_grad = True

    def init_class(self, classes):
        with torch.no_grad():
            self.fasterRCNN.roi_heads.box_predictor.init_class(classes)

    def save(self, file_name="FewshotBaseline.pkl"):
        torch.save(self.fasterRCNN.state_dict(), file_name)

    def load(self, load_obj="FewshotBaseline.pkl"):
        if isinstance(load_obj, str):
            self.fasterRCNN.load_state_dict(torch.load(load_obj))
        else:
            self.fasterRCNN.load_state_dict(load_obj)

    def __call__(self,imgs,labels=None,classes=None):
        # we only train classes in range [start_cls, end_cls]
        return self.fasterRCNN(imgs, labels)













