# Copyright (c) OpenMMLab. All rights reserved.
from math import cos, pi

from mmcv.parallel import is_module_wrapper
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class AdapterHook(Hook):
    """Hook for BYOL.

    This hook includes momentum adjustment in BYOL following:

    .. math::
        m = 1 - (1 - m_0) * (cos(pi * k / K) + 1) / 2

    where :math:`k` is the current step, :math:`K` is the total steps.

    Args:
        end_momentum (float): The final momentum coefficient
            for the target network. Defaults to 1.
        update_interval (int, optional): The momentum update interval of the
            weights. Defaults to 1.
    """

    def __init__(self, **kwargs):
        pass

    # def before_train_iter(self, runner):
    #     assert hasattr(runner.model.module, 'momentum'), \
    #         "The runner must have attribute \"momentum\" in BYOL."
    #     assert hasattr(runner.model.module, 'base_momentum'), \
    #         "The runner must have attribute \"base_momentum\" in BYOL."
    #     if self.every_n_iters(runner, self.update_interval):
    #         cur_iter = runner.iter
    #         max_iter = runner.max_iters
    #         base_m = runner.model.module.base_momentum
    #         m = self.end_momentum - (self.end_momentum - base_m) * (
    #             cos(pi * cur_iter / float(max_iter)) + 1) / 2
    #         runner.model.module.momentum = m

    def before_train_iter(self, runner):
        # print(runner)
        # print(runner.model)
        runner.model.module.backbone_1.apply_adapters()
        runner.model.module.backbone_2.apply_adapters()
        # runner.model.module.arithmetic_sum()
        

    def after_train_iter(self, runner):
        print(runner.model.module.backbone_1.conv1.weight.grad)
        print(runner.model.module.backbone_1.weight_scales[0].grad)
        # print(runner.model.module.head.fc_cls.weight.grad)
        
        runner.model.module.backbone_1.restore_original_weights()
        runner.model.module.backbone_2.restore_original_weights()
        
