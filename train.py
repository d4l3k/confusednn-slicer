import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
from os import path
import numpy as np

import gcode
import trimesh

class SpiralConv(nn.Module):
    def __init__(self, in_c, spiral_size,out_c,activation='elu',bias=True,device=None):
        super(SpiralConv,self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.device = device

        self.conv = nn.Linear(in_c*spiral_size,out_c,bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.02)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'identity':
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self,x,spiral_adj):
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()

        spirals_index = spiral_adj.view(bsize*num_pts*spiral_size) # [1d array of batch,vertx,vertx-adj]
        batch_index = torch.arange(bsize, device=self.device).view(-1,1).repeat([1,num_pts*spiral_size]).view(-1).long() # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index,spirals_index,:].view(bsize*num_pts,spiral_size*feats) # [bsize*numpt, spiral*feats]


        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize,num_pts,self.out_c)
        zero_padding = torch.ones((1,x.size(1),1), device=self.device)
        zero_padding[0,-1,0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat

class SpiralAutoencoder(nn.Module):
    def __init__(self, filters_enc, filters_dec, latent_size, sizes, spiral_sizes, spirals, D, U, device, activation = 'elu'):
        super(SpiralAutoencoder,self).__init__()
        self.latent_size = latent_size
        self.sizes = sizes
        self.spirals = spirals
        self.filters_enc = filters_enc
        self.filters_dec = filters_dec
        self.spiral_sizes = spiral_sizes
        self.D = D
        self.U = U
        self.device = device
        self.activation = activation

        self.conv = []
        input_size = filters_enc[0][0]
        for i in range(len(spiral_sizes)-1):
            if filters_enc[1][i]:
                self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[1][i],
                                            activation=self.activation, device=device).to(device))
                input_size = filters_enc[1][i]

            self.conv.append(SpiralConv(input_size, spiral_sizes[i], filters_enc[0][i+1],
                                        activation=self.activation, device=device).to(device))
            input_size = filters_enc[0][i+1]

        self.conv = nn.ModuleList(self.conv)

        self.fc_latent_enc = nn.Linear((sizes[-1]+1)*input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1]+1)*filters_dec[0][0])

        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes)-1):
            if i != len(spiral_sizes)-2:
                self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                             activation=self.activation, device=device).to(device))
                input_size = filters_dec[0][i+1]

                if filters_dec[1][i+1]:
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[1][i+1]
            else:
                if filters_dec[1][i+1]:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation=self.activation, device=device).to(device))
                    input_size = filters_dec[0][i+1]
                    self.dconv.append(SpiralConv(input_size,spiral_sizes[-2-i], filters_dec[1][i+1],
                                                 activation='identity', device=device).to(device))
                    input_size = filters_dec[1][i+1]
                else:
                    self.dconv.append(SpiralConv(input_size, spiral_sizes[-2-i], filters_dec[0][i+1],
                                                 activation='identity', device=device).to(device))
                    input_size = filters_dec[0][i+1]

        self.dconv = nn.ModuleList(self.dconv)

    def encode(self,x):
        bsize = x.size(0)
        S = self.spirals
        D = self.D

        j = 0
        for i in range(len(self.spiral_sizes)-1):
            x = self.conv[j](x,S[i].repeat(bsize,1,1))
            j+=1
            if self.filters_enc[1][i]:
                x = self.conv[j](x,S[i].repeat(bsize,1,1))
                j+=1
            x = torch.matmul(D[i],x)
        x = x.view(bsize,-1)
        return self.fc_latent_enc(x)

    def decode(self,z):
        bsize = z.size(0)
        S = self.spirals
        U = self.U

        x = self.fc_latent_dec(z)
        x = x.view(bsize,self.sizes[-1]+1,-1)
        j=0
        for i in range(len(self.spiral_sizes)-1):
            x = torch.matmul(U[-1-i],x)
            x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1))
            j+=1
            if self.filters_dec[1][i+1]:
                x = self.dconv[j](x,S[-2-i].repeat(bsize,1,1))
                j+=1
        return x


    def forward(self,x):
        bsize = x.size(0)
        z = self.encode(x)
        x = self.decode(z)
        return x

X_BOUND = 300
Y_BOUND = 300

class gcode_dataset(Dataset):
    def __init__(self, root_dir):
        gcode_files = glob.glob(path.join(root_dir, "DSA_1u.gcode"))
        stl_files = set(glob.glob(path.join(root_dir, "*.stl")))
        self.examples = []
        for gcode_file in gcode_files:
            base_file, _ = path.splitext(gcode_file)
            stl_file = base_file + ".stl"
            if stl_file not in stl_files:
                print(f"couldn't find {stl_file}")
                continue
            print(f"parsing {stl_file}")
            obj = trimesh.load_mesh(file_obj=stl_file, file_type='stl')
            assert obj.is_watertight, "must be watertight mesh"
            print(obj.bounds)
            min, max = obj.bounds
            center = (min + max) / 2
            center[2] = min[2]
            obj.apply_translation(-center + (X_BOUND/2, Y_BOUND/2, 0))
            print(obj.bounds)

            layers = gcode.GCode.from_file(gcode_file).split_layers()
            heights = list(layers.keys())
            planes = obj.section_multiplane((0,0,0), (0,0,1), heights)
            for height, plane in zip(heights, planes):
                commands = layers[height]
                self.examples.append((plane, height, commands))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        path, height, commands = self.examples[idx]
        assert path, "must have a path for this slice"
        out = np.array([
            (cmd.X, cmd.Y, cmd.E, cmd.F)
            for cmd in commands
        ])
        print(path, height)
        return path, out

dataset = gcode_dataset("data")
print("examples: ", len(dataset))
for layer, out in dataset:
    assert len(out) > 0, 'must have examples'
    layer.show()
