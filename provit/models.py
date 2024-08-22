import torchvision.models.vision_transformer as vt
import torch
import PRoViT.imagenet.models

# See: https://pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights
accuracies = {
    'vit_b_16'      : (0.81068, 0.95318),
    'vit_l_16'      : (0.79662, 0.94638),
    'vit_b_32'      : (0.75912, 0.92466),
    'vit_l_32'      : (0.76972, 0.9307),
    'deit'          : (0.81742, 0.95586),
    'resnet152'     : (0.78312, 0.94046),
    'vgg19'         : (0.72376, 0.90876)
}

def vit_l_16(dtype=torch.float32, device='cpu'):
    """Return the pre-trained ViT-L/16 model along with its top1 and top5 accuracy on the 
    ILSVRC2012 validation set (drawdown set).
    """

    acc1, acc5 = accuracies['vit_l_16']
    return vt.vit_b_16(pretrained=True).to(dtype=dtype, device=device), acc1, acc5

def vit_b_16(dtype=torch.float32, device='cpu'):
    """Return the pre-trained ViT-B/16 model along with its top1 and top5 accuracy on the 
    ILSVRC2012 validation set (drawdown set).
    """

    acc1, acc5 = accuracies['vit_b_16']
    return vt.vit_b_16(pretrained=True).to(dtype=dtype, device=device), acc1, acc5

def vit_b_32(dtype=torch.float32, device='cpu'):
    """Return the pre-trained ViT-B/32 model along with its top1 and top5 accuracy on the 
    ILSVRC2012 validation set (drawdown set).
    """
    
    acc1, acc5 = accuracies['vit_b_32']
    return vt.vit_b_32(pretrained=True).to(dtype=dtype, device=device), acc1, acc5



def vit_l_32(dtype=torch.float32, device='cpu'):
    """Return the pre-trained ViT-L/32 model along with its top1 and top5 accuracy on the 
    ILSVRC2012 validation set (drawdown set).
    """

    acc1, acc5 = accuracies['vit_l_32']
    return vt.vit_l_32(pretrained=True).to(dtype=dtype, device=device), acc1, acc5

def deit(dtype=torch.float32, device='cpu'):
    """Return the pre-trained DeiT model along with its top1 and top5 accuracy on the
    ILSVRC2012 validation set (drawdown set).
    """

    acc1, acc5 = accuracies['deit']
    return torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True).to(dtype=dtype, device=device), acc1, acc5

def resnet152(dtype=torch.float32, device='cpu'):
    """Return the pre-trained ResNet152 model along with its top1 and top5 accuracy on
    the ILSVRC2012 validation set (drawdown set).
    """
    
    acc1, acc5 = accuracies['resnet152']
    return PRoViT.imagenet.models.resnet152(pretrained=True).to(dtype=dtype, device=device), acc1, acc5

def vgg19(dtype=torch.float32, device='cpu'):
    """Return the pre-trained VGG19 model along with its top1 and top5 accuracy on the 
    ILSVRC2012 validation set (drawdown set).
    """

    acc1, acc5 = accuracies['vgg19']
    return PRoViT.imagenet.models.vgg19(pretrained=True).to(dtype=dtype, device=device), acc1, acc5

def get_encoder_output(net, x):
    """Given a Vision Transformer {net} and input {x}, compute the forward pass up to the
    final layer of the Vision Transformer. This function returns what will be the input 
    to the final layer.    
    https://pytorch.org/vision/main/_modules/torchvision/models/vision_transformer.html
    https://github.com/facebookresearch/deit/blob/main/models_v2.py 
    """
    
    if isinstance(net, vt.VisionTransformer):
        x = net._process_input(x)
        n = x.shape[0]
        batch_class_token = net.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = net.encoder(x)
        return x[:, 0]

    else:
        x = net.forward_features(x)
        return x[:, 0, :]