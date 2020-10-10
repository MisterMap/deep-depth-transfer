from collections import OrderedDict
import re

import torch


def load_undeepvo_checkpoint(model, path, strict=False):
    undeepvo_state_dict = torch.load(path, map_location=torch.device('cpu'))

    model_state_dict = OrderedDict(
        (re.sub('pose_net', '_pose_net', k) if 'pose_net' in k else k, v) for k, v in undeepvo_state_dict.items())
    model_state_dict = OrderedDict(
        (re.sub('depth_net', '_depth_net', k) if 'depth_net' in k else k, v) for k, v in model_state_dict.items())

    model.load_state_dict(model_state_dict, strict)
