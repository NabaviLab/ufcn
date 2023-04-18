import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import layers
import ufc

class UFCN(nn.Module):

    def __init__(self, img_ch = 1, output_ch = 1, block1Ch = 64, block2Ch = 128, block3Ch = 256, block4Ch = 512, block5Ch = 1024, activation = 'Relu', threshold = 0.0):
        super(UFCN, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.Conv1 = layers.conv_block(inputD = img_ch, outputD = block1Ch, activation = activation, threshold = threshold)
        self.Conv2 = layers.conv_block(inputD = block1Ch, outputD = block2Ch, activation = activation, threshold = threshold)
        self.Conv3 = layers.conv_block(inputD = block2Ch, outputD = block3Ch, activation = activation, threshold = threshold)
        self.Conv4 = layers.conv_block(inputD = block3Ch, outputD = block4Ch, activation = activation, threshold = threshold)
        self.Conv5 = layers.conv_block(inputD = block4Ch, outputD = block5Ch, activation = activation, threshold = threshold)

        self.up5 = layers.up_conv(inputD = block5Ch, outputD = block4Ch, activation = activation, threshold = threshold)
        self.att5 = layers.Attention_block(d_input = block4Ch, e_input = block4Ch, output = block3Ch, f_d = 128)
        self.Up_conv5 = layers.conv_block(inputD = block5Ch, outputD = block4Ch, activation = activation, threshold = threshold)

        self.up4 = layers.up_conv(inputD = block4Ch, outputD = block3Ch, activation = activation, threshold = threshold)
        self.att4 = layers.Attention_block(d_input = block3Ch, e_input = block3Ch, output = block2Ch, f_d = 256)
        self.Up_conv4 = layers.conv_block(inputD = block4Ch, outputD = block3Ch, activation = activation, threshold = threshold)      

        self.up3 = layers.up_conv(inputD = block3Ch, outputD = block2Ch, activation = activation, threshold = threshold)
        self.att3 = layers.Attention_block(d_input = block2Ch, e_input = block2Ch, output = block1Ch, f_d = 512)
        self.Up_conv3 = layers.conv_block(inputD = block3Ch, outputD = block2Ch, activation = activation, threshold = threshold)
    
        self.up2 = layers.up_conv(inputD = block2Ch, outputD = block1Ch, activation = activation, threshold = threshold)
        self.att2 = layers.Attention_block(d_input = block1Ch, e_input = block1Ch, output = 32, f_d = 1024)
        self.Up_conv2 = layers.conv_block(inputD = block2Ch, outputD = block1Ch, activation = activation, threshold = threshold)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size = 1, stride = 1, padding = 0)

        self.agr = nn.Conv2d(64, 1, 1)
        self.sigmoid = nn.Sigmoid()
       # self.mask = nn.Threshold(0.5,0)
        
    def forward(self, x1, x2):
   
        x1_1 = self.Conv1(x1)
        x2_1 = self.Conv1(x2)

        x1_2 = self.Maxpool(x1_1)
        x2_2 = self.Maxpool(x2_1)
        x1_2 = self.Conv2(x1_2)
        x2_2 = self.Conv2(x2_2)

        x1_3 = self.Maxpool(x1_2)
        x2_3 = self.Maxpool(x2_2)
        x1_3 = self.Conv3(x1_3)
        x2_3 = self.Conv3(x2_3)
      
        x1_4 = self.Maxpool(x1_3)
        x2_4 = self.Maxpool(x2_3)
        x1_4 = self.Conv4(x1_4)
        x2_4 = self.Conv4(x2_4)
   
        x1_5 = self.Maxpool(x1_4)
        x2_5 = self.Maxpool(x2_4)
        x1_5 = self.Conv5(x1_5)
        x2_5 = self.Conv5(x2_5)

        diff_1 = ufc.sub(x1_1, x2_1)
        diff_2 = ufc.sub(x1_2, x2_2)
        diff_3 = ufc.sub(x1_3, x2_3)
        diff_4 = ufc.sub(x1_4, x2_4)
        diff_5 = ufc.sub(x1_5, x2_5)


        up_5 = self.up5(x1_5)
        at_4, layerlabel4 = self.att5(diff_4,up_5)
        up_5 = torch.cat((at_4, up_5), dim = 1)
        up_5 = self.Up_conv5(up_5)
 
        up_4 = self.up4(up_5)
        at_3, layerlabel3 = self.att4(diff_3, up_4)
        up_4 = torch.cat((at_3, up_4), dim = 1)
        up_4 = self.Up_conv4(up_4)
    
        up_3 = self.up3(up_4)
        at_2, layerlabel2 = self.att3(diff_2, up_3)
        up_3 = torch.cat((at_2, up_3), dim = 1)
        up_3 = self.Up_conv3(up_3)
     
        up_2 = self.up2(up_3)
        at_1, layerlabel1 = self.att2(diff_1,  up_2)
        up_2 = torch.cat((at_1, up_2), dim = 1)
        up_2 = self.Up_conv2(up_2)
      
        output = self.Conv_1x1(up_2)
    
        blended = self.agr(at_1)
        sig = self.sigmoid(blended)
        mask = torch.where(sig > 0.5, 1, 0)     
   
        return output, mask, [layerlabel4, layerlabel3, layerlabel2]

 





