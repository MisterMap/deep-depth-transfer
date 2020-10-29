from collections import OrderedDict
import re

import torch
import torch.nn as nn


def load_undeepvo_checkpoint(model, path, strict=False):
    undeepvo_state_dict = torch.load(path, map_location=torch.device('cpu'))

    model_state_dict = OrderedDict(
        (re.sub('pose_net', '_pose_net', k) if 'pose_net' in k else k, v) for k, v in undeepvo_state_dict.items())
    model_state_dict = OrderedDict(
        (re.sub('depth_net', '_depth_net', k) if 'depth_net' in k else k, v) for k, v in model_state_dict.items())

    model.load_state_dict(model_state_dict, strict)


def freeze_feature_extractor(model):
    for layer in model.parameters():
        if not isinstance(layer, torch.nn.BatchNorm2d):
            layer.requires_grad = False
    if model.__getattr__('_log_scale') and model.__getattr__('_log_min_depth'):
        model._log_scale.requires_grad = True
        model._log_min_depth.requires_grad = True
        model._log_pose_scale.requires_grad = True


def unfreeze_last_layer(model):
    model._pose_net.transl3.requires_grad = True
    model._pose_net.rot3.requires_grad = True
    model._depth_net._last_conv.requires_grad = True


def init_parameters(model):
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
