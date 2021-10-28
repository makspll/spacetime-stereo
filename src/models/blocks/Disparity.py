import torch
import torch.nn as nn
import torch.nn.functional as F

class DisparityRegression(nn.Module):
    def __init__(self, max_disp):
        super(DisparityRegression, self).__init__()
        self.max_disp = max_disp

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = torch.reshape(torch.arange(0, self.max_disp, device=torch.cuda.current_device(), dtype=torch.float32),[1,self.max_disp,1,1])
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out

class DisparitySelector(nn.Module):
    def __init__(self, max_disp=192):
        super().__init__()
        self.max_disp = max_disp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(max_disp=self.max_disp)

    def forward(self, x):
        x = F.interpolate(x, [self.max_disp, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)      
        x = self.disparity(x)
        return x
