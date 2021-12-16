import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
from layers import PairNorm

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        #print(input.shape)
        #print(self.weight.shape)
        #print(adj.shape)
        #print(support.shape)
        
        
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, model, num_classes, in_channel=300, t=0, adj_file=None,trial=None):
        super(GCNResnet, self).__init__()
        self.parts= 10   # the number of regions
        self.map_threshold = 0.5  #the alpha threshold
        self.num_classes = num_classes
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

        # self.pooling = nn.MaxPool2d(14, 14)
        self.pool = nn.MaxPool2d(14,14)
        self.cov_channel = 2048

        imd = trial.suggest_categorical('Intermediete Value Layer ',[256,512,1024,2048])

        self.dropout = nn.Dropout()
        self.gc1 = GraphConvolution(in_channel, imd)
        self.norm1 = PairNorm()
        self.gc2 = GraphConvolution(imd, 2048)
        self.relu = nn.LeakyReLU(0.2)
        self.cov = nn.Conv2d(2048, self.parts, 1)
        self.fc = nn.Linear(2048*self.parts, 2048, False)
        # self.sigmoid=nn.Sigmoid()
        # self.softmax=nn.Softmax()

        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())

       # image normalization
       # self.image_normalization_mean = [0.447, 0.423, 0.388]   # nuswide dataset
       # self.image_normalization_std = [0.294, 0.282, 0.298]

        self.image_normalization_mean = [0.485, 0.456, 0.406]  #  coco dataset
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        feature = self.features(feature)
        #feature = self.pooling(feature)
        w = feature.size()
        weights = torch.sigmoid(self.cov(feature))  # Mk   K (H*W)
        batch, parts, width, height = weights.size()
        #print("batch:")
        #print(batch)
        weights_layout = weights.view(batch, -1)
        threshold_value, _ = weights_layout.max(dim=1)  # calculate AT_max

        local_max, _ = weights.view(batch, parts, -1).max(dim=2)  # calculate Mv
        threshold_value = self.map_threshold*threshold_value.view(batch, 1).expand(batch, parts)
        weights = weights * local_max.ge(threshold_value).view(batch, parts, 1, 1).float().expand(batch, parts, width,
                                                                                                  height)
        blocks = []
        for k in range(self.parts-1):
            Y = feature * weights[:, k, :, :].unsqueeze(dim=1).expand(w[0], self.cov_channel, w[2], w[3])
            blocks.append(self.pool(Y).squeeze().view(-1, self.cov_channel))
        Y = feature
        blocks.append(self.pool(Y).squeeze().view(-1, self.cov_channel)) 
        block = torch.cat(blocks, dim=1)

        feature = self.fc(block)
       
        feature = feature.view(feature.size(0), -1)

        inp = inp[0]
        adj = gen_adj(self.A).detach()

        x = inp
        #x = self.dropout(x)
        x = self.gc1(inp, adj)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        # x = self.softmax(x)
        # x = self.sigmoid(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]



def attention_gcn_pairnorm(num_classes, t, pretrained=True, adj_file=None, in_channel=300,trial = None):
    model = models.resnet101(pretrained=pretrained)
    return GCNResnet(model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel,trial = trial)
