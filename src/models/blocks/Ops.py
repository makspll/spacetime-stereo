import torch.nn as nn
import torch.nn.functional as F
from .ConvNd import Conv4d, BatchNorm4d

class ConvBR(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding,dim=2, bn=True, relu=True):
        super(ConvBR, self).__init__()
        self.relu = relu
        self.use_bn = bn

        add_kwargs = {
        }
        if dim == 2:
            self.conv_layer = nn.Conv2d
            self.batch_layer = nn.BatchNorm2d
        elif dim == 3:
            self.conv_layer = nn.Conv3d
            self.batch_layer = nn.BatchNorm3d
        elif dim == 4:
            self.conv_layer = Conv4d
            self.batch_layer = BatchNorm4d
            add_kwargs["kernel_initializer"]= lambda w: nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
 

        self.conv = self.conv_layer(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False, **add_kwargs)
        self.bn = self.batch_layer(c_out)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, self.conv_layer) and not self.conv_layer is Conv4d:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, self.batch_layer):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        self._initialize_weights()

    def forward(self, x):
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
