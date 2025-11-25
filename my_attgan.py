# Import Generator and Discriminator (ATTGAN repo). Default values

MAX_DIM = 64 * 16  # 1024

import torch
import torch.nn as nn
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock
from torchsummary import summary
from my_attgan_blocks import Generator, Discriminators

class AttGAN():
    def __init__(self, args):
        """Args is a dictionary loaded from the setting.txt file"""
    
        self.enc_dim = args.get('enc_dim')
        self.enc_layers = args.get('enc_layers')
        self.enc_norm = args.get('enc_norm')
        self.enc_acti = args.get('enc_acti')
        self.dec_dim = args.get('dec_dim')
        self.dec_layers = args.get('dec_layers')
        self.dec_norm = args.get('dec_norm')
        self.dect_acti = args.get('dect_acti')
        self.n_attrs = args.get('n_attrs')
        self.shortcut_layers = args.get('shortcut_layers')
        self.inject_layers = args.get('inject_layers')
        self.img_size = args.get('img_size')
        self.dis_dim = args.get('dis_dim')
        self.dis_norm = args.get('dis_norm')
        self.dis_acti = args.get('dis_acti')
        self.dis_fc_dim = args.get('dis_fc_dim')
        self.dis_fc_norm = args.get('dis_fc_norm')
        self.dis_fc_acti = args.get('dis_fc_acti')
        self.dis_layers = args.get('dis_layers')
        
        self.G = Generator(
            self.enc_dim, self.enc_layers, self.enc_norm, self.enc_acti,
            self.dec_dim, self.dec_layers, self.dec_norm, self.dect_acti,
            self.n_attrs, self.shortcut_layers, self.inject_layers, self.img_size
        )
        
        self.D = Discriminators(
            self.dis_dim, self.dis_norm, self.dis_acti,
            self.dis_fc_dim, self.dis_fc_norm, self.dis_fc_acti, self.dis_layers, self.img_size
        )
        
    def eval(self):
        self.G.eval()
        self.D.eval()
        
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D' in states:
            self.D.load_state_dict(states['D'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D' in states:
            self.optim_D.load_state_dict(states['optim_D'])