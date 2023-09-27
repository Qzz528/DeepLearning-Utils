# -*- coding: utf-8 -*-
'非极大值抑制，目标检测框筛选'

'torchvision有现成的nms方法'

'import torchvision'
'keep = torchvision.ops.batched_nms(boxes,scores,idxs,iou_threshold)'
#boxes，scores，idxs均为tensor类。
#boxes格式为（框数，4），其中4是x1y1x2y2型的框坐标。
#scores格式为（框数，），表示对应boxes的置信度。
#idxs格式为（框数，），表示对应boxes中的物体类别。
#每个框将对应一个boxes中的框坐标，一个scores中的置信度，以及一个idxs中的物体类别。
#iou_threshold为阈值，该NMS时iou以上的框会进行滤除。
#返回值为要保留框的序数index
'print(boxes[keep])' #保留的框的坐标
'print(scores[keep])' #保留的框的置信度
'print(idxs[keep])' #保留的框对应的物体类别


'一个基于原理的IOU及NMS代码（易于理解，但代码运算效率低）'

#入参及返回，除了类型为numpy的array，而非torch的tensor外，格式用法与torchvision中的nms完全一致。
import numpy as np
#计算两个框的iou交并比
def iou(box1,box2):
    #box1,box2:(4,) |box1,box2 type x1y1x2y2
    b1x1,b1y1,b1x2,b1y2 = box1
    b2x1,b2y1,b2x2,b2y2 = box2
    #交集框的坐标
    ix1,iy1,ix2,iy2 = max(b1x1,b2x1),max(b1y1,b2y1),min(b1x2,b2x2),min(b1y2,b2y2)
    #求面积
    s1 = (b1x2-b1x1)*(b1y2-b1y1)
    s2 = (b2x2-b2x1)*(b2y2-b2y1)
    si = (ix2-ix1)*(iy2-iy1)
    return si/(s1+s2-si)

def nms(boxes,scores,idxs,iou_threshold):
    #boxes:(n,4)|scores:(n,)|idxs:(n,)
    #boxes type x1y1x2y2
    cls_list = list(set(idxs))
    keep_idx = []
    for i in cls_list: #对每个种类的box进行nms
        #从所有box中跳出指定类别的
        cls_idx = np.argwhere(idxs==i)[:,0]
        cls_boxes = boxes[cls_idx]
        cls_scores = scores[cls_idx]
        
        cls_keep_idx = []
        cls_left_idx = list(np.argsort(cls_scores)[::-1]) #按照scores从大到小排列
        while len(cls_left_idx):
            max_idx = cls_left_idx[0] #scores最大的保留
            cls_keep_idx.append(max_idx)
            cls_left_idx.remove(max_idx)
            for no_max_idx in cls_left_idx: #非极大值进行对比。iou过大的筛除
                if iou(cls_boxes[max_idx],cls_boxes[no_max_idx]) > iou_threshold:
                    cls_left_idx.remove(no_max_idx)
        keep_idx.extend(cls_idx[cls_keep_idx]) #保留的序号加入列表 #序号是针对所有类box的序号
        keep_idx.sort()
    return keep_idx


if __name__ == "__main__":

    import torchvision
    import torch

    boxes = np.array([[2,2,8,8],
                     [1,1,9,9],
                     [8,8,9,9]],float)
    scores = np.array([0.9,0.8,0.2],float)
    idxs = np.array([1,1,1],int)   #三个box同类物体
    # idxs = np.array([1,2,1],int) #三个box不同类物体
    iou_thresh = 0.5

    #torchvision需要数据是tensor格式
    # keep_index1 = np.array(
    # torchvision.ops.batched_nms(torch.from_numpy(boxes),
    #                             torch.from_numpy(scores),
    #                             torch.from_numpy(idxs),
    #                             iou_thresh)
    # )

    keep_index2 = nms(boxes,scores,idxs,0.5)

    #两者效果相同
    # print(keep_index1)
    print(keep_index2)
