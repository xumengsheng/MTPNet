from torch import nn
from torch.nn import init
import torch
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from fvcore.nn import FlopCountAnalysis, flop_count_table



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 输入特征图的尺寸被放大两倍
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),  # 对放大后的特征图进行特征提取和变换，输出通道变为ch_out
		    nn.BatchNorm2d(ch_out),  # 加速训练过程并提高模型的稳定性
			nn.ReLU(inplace=True)  # 增加非线性，有助于模型学习复杂的映射关系
        )

    def forward(self,x):
        x = self.up(x)
        return x

# @ARCH_REGISTRY.register()
class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=3,multi_stage=False):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        self.activation=nn.Sequential(nn.Sigmoid())
        # init_weights(self)
        self.apply(self._init_weights)


    def _init_weights(self,m):
        init_type='normal'
        gain=0.02
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)  # [1,64,512,512]

        x2 = self.Maxpool(x1)  # [1,64,256,256]
        x2 = self.Conv2(x2)  # [1,128,256,256]
        
        x3 = self.Maxpool(x2)  # [1,128,128,128]
        x3 = self.Conv3(x3)  # [1,256,128,128]

        x4 = self.Maxpool(x3)  # [1,256,64,64]
        x4 = self.Conv4(x4)  # [1,512,64,64]

        x5 = self.Maxpool(x4)  # [1,512,32,32]
        x5 = self.Conv5(x5)  # [1,1024,32,32]

        # decoding + concat path
        d5 = self.Up5(x5)  # [1,1024,64,64],UP5改变图像尺寸
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)  # [1,512,64,64]，Up_conv5改变输入输出通道
        
        d4 = self.Up4(d5)  # [1,512，128，128]
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)  # [1,256,128,128]

        d3 = self.Up3(d4)  # [1,256,256,256]
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)  # [1,128,256,256]

        d2 = self.Up2(d3)  # [1,128,512,512]
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)  # [1,64,512,512]

        d1 = self.Conv_1x1(d2)  # [1,3,512,512]
        d1 = self.activation(d1)
        return d1

if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型实例并移动到设备
    model = U_Net(
        img_ch=3,
        output_ch=3,

    ).to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_m = total_params / 1e6  # 转换为百万

    print(f"模型参数: {params_m:.2f} M")

    # 尝试计算FLOPs，如果失败则跳过
    try:
        # 创建随机输入并移动到设备
        input_tensor = torch.randn(1, 3, 256, 256).to(device)

        # 计算FLOPs
        flops = FlopCountAnalysis(model, input_tensor)
        flops.unsupported_ops_warnings(False)  # 关闭警告
        flops_g = flops.total() / 1e9  # 转换为 GFLOPs
        print(f"计算量: {flops_g:.2f} GFLOPs")
    except Exception as e:
        print(f"发生错误: {e}")

        print(f"计算量: {flops_g:.2f} GFLOPs")