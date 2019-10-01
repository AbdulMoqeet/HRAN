from model import common
import torch.nn as nn

import torch.nn


def make_model(args, parent=False):    return HRAN(args)


class HRAB(nn.Module):
    def __init__(self, conv=common.default_conv, n_feats=64):
        super(HRAB, self).__init__()


        kernel_size_1 = 3

        reduction = 4

        self.conv_du_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv(n_feats, n_feats // reduction, 1),
            nn.LeakyReLU(inplace=True),
            conv(n_feats // reduction, n_feats, 1),
            nn.Sigmoid()
        )

        self.conv_3 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(n_feats, n_feats, kernel_size_1, dilation=2)

        self.conv_3_1 = conv(n_feats*2, n_feats, kernel_size_1)
        self.conv_3_2_1 = conv(n_feats*2, n_feats, kernel_size_1, dilation=2)

        self.LR = nn.LeakyReLU(inplace=True)

        self.conv_11 = conv(n_feats*2, n_feats, 1)



    def forward(self, x):

        res_x = x

        a =  self.conv_du_1(x)
        b1 = self.LR(self.conv_3(x))
        b2 = self.LR(self.conv_3_2(x)) + b1
        B = torch.cat([ b1, b2 ],1)

        b1 = self.conv_3_1(B)
        b2 = self.LR(self.conv_3_2_1(B)) + b1

        B = torch.cat([b1,b2], 1)

        B = self.conv_11(B)


        output = a*B

        output = output + res_x

        return output

class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [HRAB( n_feats=n_feat) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class HRAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(HRAN, self).__init__()

        n_feats = args.n_feats
        self.n_blocks =  args.n_resblocks
        n_resgroups = args.n_resgroups

        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        self.n_blocks = n_blocks

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        modules_head_2 = [conv(n_feats, n_feats, kernel_size)]

        # define body module
        modules_body = nn.ModuleList()

        # define body module
        modules_body = [ ResidualGroup(conv, n_feats, kernel_size, n_blocks) for _ in range(n_resgroups)]


        modules_tail = [
            conv(n_feats, n_feats, kernel_size),
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]
        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)



        self.head_1 = nn.Sequential(*modules_head)
        self.head_2 = nn.Sequential(*modules_head_2)
        self.fusion = nn.Sequential(*[nn.Conv2d(n_feats * 2, n_feats, 1, padding=0, stride=1)])
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head_1(x)
        res = x

        x = self.head_2(x)

        res_x = x

        HRAB_out = []
        for i in range(4):
            x = self.body[i](x)
            HRAB_out.append(x)
 

        while len(HRAB_out) > 2:
            fusions = []
            for i in range(0, len(HRAB_out), 2):
                fusions.append( self.fusion(torch.cat((HRAB_out[i], HRAB_out[i+1]), 1)))

            HRAB_out = fusions

        res = res + self.fusion(torch.cat(HRAB_out, 1))
        x = self.tail(res)
        x = self.add_mean(x)
        return x

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
