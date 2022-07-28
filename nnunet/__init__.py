from __future__ import absolute_import
import torch
import numpy as np
import warnings
import random
#print("\n\nPlease cite the following paper when using nnUNet:\n\nIsensee, F., Jaeger, P.F., Kohl, S.A.A. et al. "
#      "\"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.\" "
#      "Nat Methods (2020). https://doi.org/10.1038/s41592-020-01008-z\n\n")
#print("If you have questions or suggestions, feel free to open an issue at https://github.com/MIC-DKFZ/nnUNet\n")

from . import *

#torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore", category=UserWarning)
#torch.manual_seed(0)
#random.seed(0)
#np.random.seed(0)