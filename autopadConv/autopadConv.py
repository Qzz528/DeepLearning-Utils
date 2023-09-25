# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math


class Conv1d_autopad(nn.Conv1d):  
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=None,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
   
        self.cutlast = False
        if padding == None:#保持数据尺寸不变或按stride倍数减小
            if stride == 1:
                if dilation*(kernel_size-1)%2==0:
                    padding = dilation*(kernel_size-1)//2
                else:
                    padding = math.ceil(dilation*(kernel_size-1)/2)
                    self.cutlast = True
            else:
                padding = (dilation*(kernel_size-1)-stride+2)//2
            
        super(Conv1d_autopad,self).__init__(in_channels,
                                             out_channels,
                                             kernel_size,
                                             stride,
                                             padding,
                                             dilation,
                                             groups,
                                             bias,
                                             padding_mode)    
    def forward(self, x):
        x = self._conv_forward(x, self.weight, self.bias)
        if self.cutlast:
            x = x[:,:,:-1]
            
        return x

    

class Conv2d_autopad(nn.Conv2d):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1,1),
                 padding=None,
                 dilation=(1,1),
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        
        self.cutlast = [False, False]
        
        kernel_size = list(kernel_size) if type(kernel_size)!= int else [kernel_size]*2
        stride = list(stride) if type(stride)!= int else [stride]*2
        dilation = list(dilation) if type(dilation)!= int else [dilation]*2
        if padding == None:
            padding = [None,None]
        else:
            padding = list(padding) if type(padding)!= int else [padding]*2


        for i in range(2):#2dims
            if padding[i] == None:#保持数据尺寸不变或按stride倍数减小
                
                if stride[i] == 1:
                    if dilation[i]*(kernel_size[i]-1)%2==0:
                        padding[i] = dilation[i]*(kernel_size[i]-1)//2
                    else:
                        padding[i] = math.ceil(dilation[i]*(kernel_size[i]-1)/2)
                        self.cutlast[i] = True
                else:
                    padding[i] = (dilation[i]*(kernel_size[i]-1)-stride[i]+2)//2
            
        super(Conv2d_autopad,self).__init__(in_channels,
                                             out_channels,
                                             kernel_size,
                                             stride,
                                             padding,
                                             dilation,
                                             groups,
                                             bias,
                                             padding_mode)
        
        
        
    def forward(self, x):
        x = self._conv_forward(x, self.weight, self.bias)
        if self.cutlast[0]:
            x = x[:,:,:-1,:]
        if self.cutlast[1]:
            x = x[:,:,:,:-1]            
            
        return x



class DeConv1d_autopad(nn.ConvTranspose1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=None,
                 output_padding = None,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros'):
   
        if output_padding == None:#保持数据尺寸不变或按stride倍数减小
            if padding != None:
                output_padding = 2*padding + stride - dilation*(kernel_size-1) -1
            else:    
                for padding in range(dilation*(kernel_size-1)):
                    output_padding = 2*padding + stride - dilation*(kernel_size-1) -1
                    if (output_padding >=0) and (output_padding<stride):
                        break
                                
        super(DeConv1d_autopad,self).__init__(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride,
                                              padding,
                                              output_padding,
                                              groups,
                                              bias,
                                              dilation,
                                              padding_mode)
      
        
class DeConv2d_autopad(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=None,
                 output_padding = None,
                 groups=1,
                 bias=True,
                 dilation=1,
                 padding_mode='zeros'):
        
        kernel_size = list(kernel_size) if type(kernel_size)!= int else [kernel_size]*2
        stride = list(stride) if type(stride)!= int else [stride]*2
        dilation = list(dilation) if type(dilation)!= int else [dilation]*2
        if padding == None:
            padding = [None,None]
        else:
            padding = list(padding) if type(padding)!= int else [padding]*2
        
        if output_padding == None:
            output_padding = [None,None]
        else:
            output_padding = list(output_padding) if type(output_padding)!= int else [output_padding]*2
                

        for i in range(2):#2dims
            if output_padding[i] == None:#保持数据尺寸不变或按stride倍数减小
                if padding[i] != None:
                    output_padding[i] = 2*padding[i] + stride[i] - dilation[i]*(kernel_size[i]-1) -1
                else:    
                    for padding[i] in range(dilation[i]*(kernel_size[i]-1)):
                        output_padding[i] = 2*padding[i] + stride[i] - dilation[i]*(kernel_size[i]-1) -1
                        if (output_padding[i] >=0) and (output_padding[i]<stride[i]):
                            break
                    
            
        super(DeConv2d_autopad,self).__init__(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride,
                                              padding,
                                              output_padding,
                                              groups,
                                              bias,
                                              dilation,
                                              padding_mode)        
        

    
if __name__ == '__main__':
    
    print('---conv1d---')
    x = torch.randn(1,1,100)
    
    #当padding设置为非None，同torch中的卷积
    c = Conv1d_autopad(in_channels=1,
                        out_channels=1,
                        kernel_size = 2,
                        stride = 1,
                        padding = 0, 
                        dilation = 3)
    print(c(x).shape) #torch.Size([1, 1, 97])
    
    #当padding设置为None，将自动填充使输入数据长度变为1/stride
    c = Conv1d_autopad(in_channels=1,
                        out_channels=1,
                        kernel_size = 2,
                        stride = 1,
                        padding = None, 
                        dilation = 3)
    print(c(x).shape) #torch.Size([1, 1, 100])
    
    c = Conv1d_autopad(in_channels=1,
                        out_channels=1,
                        kernel_size = 5,
                        stride = 4,
                        padding = None, 
                        dilation = 3)
    print(c) #Conv1d_autopad(1, 1, kernel_size=(2,), stride=(4,), dilation=(3,))
    print(c(x).shape) #torch.Size([1, 1, 25])     
    print('*'*20)
    
    
    print('---conv2d---')
    x = torch.randn(1,1,300,400)
    #二维同理，padding为None则h,w两个维度自动填充，否则则指定为None的维度填充
    #padding非None则同torch的卷积
    #所有入参如为int，则h,w两个维度均为此参数
    c = Conv2d_autopad(in_channels=1, 
                       out_channels=1, 
                       kernel_size = 2,
                       stride = 1,
                       padding = None, 
                       dilation=4)
    print(c(x).shape) #torch.Size([1, 1, 300, 400])
    
    c = Conv2d_autopad(in_channels=1, 
                       out_channels=1, 
                       kernel_size = 2,
                       stride = (3,2),
                       padding = None, 
                       dilation=4)
    print(c) #Conv2d_autopad(1, 1, kernel_size=[2, 2], stride=[3, 2], padding=[1, 2], dilation=[4, 4])
    print(c(x).shape) #torch.Size([1, 1, 100, 200])
    
    c = Conv2d_autopad(in_channels=1, 
                       out_channels=1, 
                       kernel_size = 2,
                       stride = 1,
                       padding = (None,0), 
                       dilation=1)
    print(c) #Conv2d_autopad(1, 1, kernel_size=[2, 2], stride=[1, 1], padding=[1, 0], dilation=[1, 1])
    print(c(x).shape) #torch.Size([1, 1, 300, 399])
    print('*'*20)
    
    
    print('---convtranspose1d&2d---')
    #反卷积如果padding非None,output_padding为None,则自动计算output_padding
    #将输入输出长度变为stride倍
    #也可以padding,output_padding均设置为None,全部自动计算
    #如果padding,output_padding均不是None，则同torch的反卷积
    ####由于反卷积机制，某些情况求出的填充值无法实现
    x = torch.randn(1,1,100)
    dc = DeConv1d_autopad(in_channels=1, 
                          out_channels=1,
                          kernel_size = 4,
                          stride = 3,
                          padding = None,
                          output_padding = None)
    print(dc) #DeConv1d_autopad(1, 1, kernel_size=(4,), stride=(3,), padding=(1,), output_padding=(1,))
    print(dc(x).shape) #torch.Size([1, 1, 300])
    
    
    x = torch.randn(1,1,100,200)
    dc = DeConv2d_autopad(in_channels=1, 
                          out_channels=1,
                          kernel_size = 3,
                          stride =(3,2),
                          padding = None,
                          output_padding = None)
    print(dc) #DeConv2d_autopad(1, 1, kernel_size=[3, 3], stride=[3, 2], padding=[0, 1], dilation=[1, 1], output_padding=[0, 1])
    print(dc(x).shape) #torch.Size([1, 1, 300, 400])

    