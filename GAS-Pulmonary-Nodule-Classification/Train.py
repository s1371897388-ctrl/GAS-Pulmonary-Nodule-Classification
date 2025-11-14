import utils.util as util
from options import options

import os
import random
import numpy as np
import torch

from Trainer import Trainer

def set_seed(seed=42):
    """固定随机种子以保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    opt = options.parse()
    
    # 固定随机种子（从命令行参数获取，默认42）
    set_seed(opt.seed)
    util.mkdirs(os.path.join(opt.checkpoints_dir, opt.name))
    logger = util.get_logger(os.path.join(opt.checkpoints_dir, opt.name, 'logger.log'))

    Trainer(opt, logger).train()

# 作者：孙海滨
# 日期：2024/5/22
