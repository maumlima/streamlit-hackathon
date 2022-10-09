import argparse
import torch
from model_unet import HackathonModel
from dataset import HackathonDataset

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib import image

from collections import defaultdict

def ret(img): 
    img = torch.tensor(img)

    input = defaultdict()
    input['img'] = img[None, :]

    from model_unet import HackathonModel
    model = HackathonModel.load_from_checkpoint('model_weights/unet.ckpt')
    model.eval()

    segmented = model(input)

    #plt.imshow(segmented.cpu().detach().numpy(), cmap='Greys')
    #plt.savefig('eval_outs/seg')

    from model_efficient import HackathonModel
    model = HackathonModel.load_from_checkpoint('model_weights/efficientnet.ckpt')
    model.eval()

    has_silo = model(input)
    out = torch.nn.Sigmoid()(has_silo)
    
    return(out, segmented.cpu().detach().numpy())
    
