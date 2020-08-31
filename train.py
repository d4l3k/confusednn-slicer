import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch import optim
import glob
from os import path
import numpy as np
from torchvision import transforms

import gcode
import trimesh
from tqdm import tqdm


class SpiralConv(nn.Module):
    def __init__(
        self, in_c, spiral_size, out_c, activation="elu", bias=True, device=None
    ):
        super(SpiralConv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.device = device

        self.conv = nn.Linear(in_c * spiral_size, out_c, bias=bias)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.02)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "identity":
            self.activation = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x, spiral_adj):
        bsize, num_pts, feats = x.size()
        _, _, spiral_size = spiral_adj.size()

        spirals_index = spiral_adj.view(
            bsize * num_pts * spiral_size
        )  # [1d array of batch,vertx,vertx-adj]
        batch_index = (
            torch.arange(bsize, device=self.device)
            .view(-1, 1)
            .repeat([1, num_pts * spiral_size])
            .view(-1)
            .long()
        )  # [0*numpt,1*numpt,etc.]
        spirals = x[batch_index, spirals_index, :].view(
            bsize * num_pts, spiral_size * feats
        )  # [bsize*numpt, spiral*feats]

        out_feat = self.conv(spirals)
        out_feat = self.activation(out_feat)

        out_feat = out_feat.view(bsize, num_pts, self.out_c)
        zero_padding = torch.ones((1, x.size(1), 1), device=self.device)
        zero_padding[0, -1, 0] = 0.0
        out_feat = out_feat * zero_padding

        return out_feat


class SpiralAutoencoder(nn.Module):
    def __init__(
        self,
        filters_enc,
        filters_dec,
        latent_size,
        sizes,
        spiral_sizes,
        spirals,
        D,
        U,
        device,
        activation="elu",
    ):
        super(SpiralAutoencoder, self).__init__()
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
        for i in range(len(spiral_sizes) - 1):
            if filters_enc[1][i]:
                self.conv.append(
                    SpiralConv(
                        input_size,
                        spiral_sizes[i],
                        filters_enc[1][i],
                        activation=self.activation,
                        device=device,
                    ).to(device)
                )
                input_size = filters_enc[1][i]

            self.conv.append(
                SpiralConv(
                    input_size,
                    spiral_sizes[i],
                    filters_enc[0][i + 1],
                    activation=self.activation,
                    device=device,
                ).to(device)
            )
            input_size = filters_enc[0][i + 1]

        self.conv = nn.ModuleList(self.conv)

        self.fc_latent_enc = nn.Linear((sizes[-1] + 1) * input_size, latent_size)
        self.fc_latent_dec = nn.Linear(latent_size, (sizes[-1] + 1) * filters_dec[0][0])

        self.dconv = []
        input_size = filters_dec[0][0]
        for i in range(len(spiral_sizes) - 1):
            if i != len(spiral_sizes) - 2:
                self.dconv.append(
                    SpiralConv(
                        input_size,
                        spiral_sizes[-2 - i],
                        filters_dec[0][i + 1],
                        activation=self.activation,
                        device=device,
                    ).to(device)
                )
                input_size = filters_dec[0][i + 1]

                if filters_dec[1][i + 1]:
                    self.dconv.append(
                        SpiralConv(
                            input_size,
                            spiral_sizes[-2 - i],
                            filters_dec[1][i + 1],
                            activation=self.activation,
                            device=device,
                        ).to(device)
                    )
                    input_size = filters_dec[1][i + 1]
            else:
                if filters_dec[1][i + 1]:
                    self.dconv.append(
                        SpiralConv(
                            input_size,
                            spiral_sizes[-2 - i],
                            filters_dec[0][i + 1],
                            activation=self.activation,
                            device=device,
                        ).to(device)
                    )
                    input_size = filters_dec[0][i + 1]
                    self.dconv.append(
                        SpiralConv(
                            input_size,
                            spiral_sizes[-2 - i],
                            filters_dec[1][i + 1],
                            activation="identity",
                            device=device,
                        ).to(device)
                    )
                    input_size = filters_dec[1][i + 1]
                else:
                    self.dconv.append(
                        SpiralConv(
                            input_size,
                            spiral_sizes[-2 - i],
                            filters_dec[0][i + 1],
                            activation="identity",
                            device=device,
                        ).to(device)
                    )
                    input_size = filters_dec[0][i + 1]

        self.dconv = nn.ModuleList(self.dconv)

    def encode(self, x):
        bsize = x.size(0)
        S = self.spirals
        D = self.D

        j = 0
        for i in range(len(self.spiral_sizes) - 1):
            x = self.conv[j](x, S[i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_enc[1][i]:
                x = self.conv[j](x, S[i].repeat(bsize, 1, 1))
                j += 1
            x = torch.matmul(D[i], x)
        x = x.view(bsize, -1)
        return self.fc_latent_enc(x)

    def decode(self, z):
        bsize = z.size(0)
        S = self.spirals
        U = self.U

        x = self.fc_latent_dec(z)
        x = x.view(bsize, self.sizes[-1] + 1, -1)
        j = 0
        for i in range(len(self.spiral_sizes) - 1):
            x = torch.matmul(U[-1 - i], x)
            x = self.dconv[j](x, S[-2 - i].repeat(bsize, 1, 1))
            j += 1
            if self.filters_dec[1][i + 1]:
                x = self.dconv[j](x, S[-2 - i].repeat(bsize, 1, 1))
                j += 1
        return x

    def forward(self, x):
        bsize = x.size(0)
        z = self.encode(x)
        x = self.decode(z)
        return x


X_BOUND = 300
Y_BOUND = 300
PITCH = 0.4
MAX_LEN = 200
#MAX_LEN = 50
LAYER_HEIGHT = 0.3


class gcode_dataset(Dataset):
    def __init__(self, root_dir):
        self.transform = transforms.Compose([transforms.ToTensor(),])
        gcode_files = glob.glob(path.join(root_dir, "DSA_*.gcode"))
        stl_files = set(glob.glob(path.join(root_dir, "*.stl")))
        self.examples = []
        for gcode_file in tqdm(gcode_files):
            base_file, _ = path.splitext(gcode_file)
            stl_file = base_file + ".stl"
            if stl_file not in stl_files:
                print(f"couldn't find {stl_file}")
                continue
            print(f"parsing {stl_file}")
            obj = trimesh.load_mesh(file_obj=stl_file, file_type="stl")
            if not obj.is_watertight:
                print(f"{stl_file} is not watertight!")
            min, max = obj.bounds
            center = (min + max) / 2
            center[2] = min[2]
            obj.apply_translation(-center + (X_BOUND / 2, Y_BOUND / 2, 0))
            assert np.allclose(obj.bounds[0][2], 0)

            layers = gcode.GCode.from_file(gcode_file).normalize().split_layers()
            offset = LAYER_HEIGHT/2
            heights = list(height - offset for height in layers.keys())
            planes = obj.section_multiplane((0, 0, 0), (0, 0, 1), heights)
            for height, plane in zip(heights, planes):
                code = layers[height+offset]
                if not plane:
                    print(f"{stl_file}: missing plane this slice {height}")
                    continue
                assert len(code.commands) > 0
                assert height >= 0
                assert plane.area > 0
                self.examples.append((plane, height, code))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        plane, height, code = self.examples[idx]
        #if len(commands) > MAX_LEN:
        #    print(f'longer than max len {len(commands)}')
        out = np.zeros((MAX_LEN, 2))
        rasterized = code.rasterize(MAX_LEN)
        assert len(rasterized.commands) == MAX_LEN
        for i, cmd in enumerate(rasterized.commands):
            out[i, :] = (cmd.X, cmd.Y) #, cmd.E, cmd.F)
        out -= (X_BOUND / 2, Y_BOUND/2) #, 0, 5000)
        out /= (X_BOUND / 2, Y_BOUND/2) #, 1, 10000)
        img = trimesh.path.raster.rasterize(
            plane,
            pitch=PITCH,
            resolution=(X_BOUND / PITCH, Y_BOUND / PITCH),
            origin=(0, 0),
            fill=True,
            width=None,
        )
        grid = self.transform(img) * 2 - 1
        assert np.allclose(grid.max(), 1) and np.allclose(grid.min(), -1)

        dense = torch.tensor([height/200], dtype=torch.float)
        label = torch.tensor(out, dtype=torch.float)
        return grid, dense, label

class Encoder(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        self.output_size = output_size
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 32, 5),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(32 * 19 * 19, output_size)

    def forward(self, image):
        x = self.cnn(image)
        x = self.fc(x.view(len(image), -1))
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, output_size, hidden_dim=200):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=2)

        # The linear layer that maps from hidden state space to tag space
        self.fc2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 200),
            nn.Linear(200, 150),
            nn.Linear(150, output_size),
        )

    def forward(self, x):
        x = x.expand((MAX_LEN, len(x), self.embedding_dim))
        lstm_out, _ = self.rnn(x)
        x = lstm_out.view(-1, self.hidden_dim)
        tag_space = self.fc2(x)
        x = tag_space.view(-1, MAX_LEN, self.output_size)
        return x

class DecoderCNN(nn.Module):
    def __init__(self, embedding_dim, output_size, hidden_dim=200):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.cnn = nn.Sequential(
            nn.ConvTranspose1d(embedding_dim, 500, 4, stride=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(500, 250, 5, stride=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(250, 100, 4, stride=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(100, 60, 3, stride=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(60, 60, 3, stride=2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(60, 60, 4, stride=2),
        )

        # The linear layer that maps from hidden state space to tag space
        self.fc2 = nn.Sequential(
            nn.Linear(60, 200),
            nn.Linear(200, 150),
            nn.Linear(150, output_size),
        )

    def forward(self, x):
        batch_size = len(x)
        x = self.cnn(x.unsqueeze(2))
        if x.shape[2] != MAX_LEN:
            print(x.shape)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size * MAX_LEN, -1)
        x = self.fc2(x)
        x = x.view(batch_size, MAX_LEN, -1)
        return x

class Net(nn.Module):
    def __init__(self, embedding_dim=1000, dense_dim=1, output_size=2):
        super().__init__()

        self.encoder = Encoder(embedding_dim)
        self.decoder = DecoderCNN(embedding_dim + dense_dim, output_size)

    def forward(self, image, dense):
        x = self.encoder(image)
        x = self.decoder(torch.cat((x, dense), dim=1))
        return x


device = torch.device("cuda:0")
dataset = gcode_dataset("data")
print("examples: ", len(dataset))

trainloader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=True, num_workers=16
)

model = Net().to(device)
model.train()
print(model)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
#loss_function = torch.nn.SmoothL1Loss()
loss_function = torch.nn.MSELoss()

def write_gcode(f, out):
    with open(f, 'w') as f:
        f.write("M107 ; fan off\n")
        f.write("M104 S200 ; set temp\n")
        f.write("G28 ; home all axes\n")
        f.write("G1 Z5 F5000 ; lift nozzle\n")
        f.write("M109 S200 ; heat and wait\n")
        f.write("G21 ; millimeters\n")
        f.write("G90 ; absolute coordinates\n")
        f.write("M83 ; extruder relative\n")
        for x, y in out:
            x = x * (X_BOUND / 2) + X_BOUND / 2
            y = y * (Y_BOUND / 2) + Y_BOUND / 2
            f.write(f"G1 X{x:.3f} Y{y:.3f} E1 F5000\n")


best_loss = 1000000
for epoch in range(10000):
    summed_loss = 0.0
    num_examples = 0
    pbar = tqdm(trainloader)
    for img_cpu, dense, label in pbar:
        img = img_cpu.to(device)
        dense = dense.to(device)
        label = label.to(device)

        model.zero_grad()
        out = model(img, dense)

        loss = loss_function(out, label)
        loss.backward()
        optimizer.step()

        summed_loss += loss.item() * len(label)
        num_examples += len(label)
        cur_loss = summed_loss/num_examples
        pbar.set_description(
            f"epoch {epoch} - loss {cur_loss}",
            refresh=False,
        )
    if cur_loss < best_loss:
        best_loss = cur_loss
        print(f"new best loss {best_loss}")
        normalized = img_cpu[0] / 0.5 + 1
        im = transforms.ToPILImage()(normalized)
        im.save(f'image{epoch}.png', format='png')
        write_gcode(f'label{epoch}.gcode', label[0])
        write_gcode(f'out{epoch}.gcode', out[0])

    scheduler.step(cur_loss)
    #print(label)
