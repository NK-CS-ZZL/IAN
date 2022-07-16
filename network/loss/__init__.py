from .losses import (CharbonnierLoss, GANLoss, L1Loss, MSELoss, PerceptualLoss,
                     WeightedTVLoss, g_path_regularize, gradient_penalty_loss,
                     r1_penalty, GrayLoss, SSIMLoss, CosineDistLoss, CELoss, ColorSSIMLoss, GradientLoss)

__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss', 'PerceptualLoss',
    'GANLoss', 'gradient_penalty_loss', 'r1_penalty', 'g_path_regularize', 
    'GrayLoss', 'SSIMLoss', 'CosineDistLoss' 'CELoss', 'ColorSSIMLoss', 'GradientLoss'
]