import importlib
import re

# def create_model(opt=None, leader=False, trans_fusion_info=None):
#
#     arch = re.sub('\d', '', opt.model)
#
#
#     if opt.dataset == 'LIDC_IDRI':
#         arch += '_lidc'
#         num_classes = 2
#
#
#     model_filename = f'models.{arch}'
#     model_lib = importlib.import_module(model_filename)
#
#     model_cls = None
#     for name, cls in model_lib.__dict__.items():
#         if name.lower() == opt.model.lower():
#             model_cls = cls
#     inplanes = 3
#     outplanes = 16
#     model = model_cls(inplanes, outplanes, num_classes=num_classes, leader=leader, trans_fusion_info=trans_fusion_info)
#     return model


import importlib
import re
# from models.resnet_lidc import BasicBlock
# from models.DWCGhost_lidc import GhostNet
def create_model(opt=None, leader=False, trans_fusion_info=None):

    arch = re.sub('\d', '', opt.model)

    if opt.dataset == 'LIDC_IDRI':
        arch += '_lidc'
        num_classes = 2

    model_filename = f'models.{arch}'
    model_lib = importlib.import_module(model_filename)

    model_cls = None
    for name, cls in model_lib.__dict__.items():
        if name.lower() == opt.model.lower():
            model_cls = cls

    # model = model_cls(num_classes=num_classes, leader=leader, trans_fusion_info=trans_fusion_info)
    model = model_cls(num_classes=num_classes, leader=leader, trans_fusion_info=trans_fusion_info)

    return model
