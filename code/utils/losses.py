import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
# from metrics import dice_coef
# from metrics import dice
from collections import OrderedDict
import warnings
from utils.util import dequeue_and_enqueue
warnings.filterwarnings("ignore")



def ConstraLoss(inputs, targets):

    m=nn.AdaptiveAvgPool2d(1)
    input_pro = m(inputs)
    input_pro = input_pro.view(inputs.size(0),-1) #N*C
    targets_pro = m(targets)
    targets_pro = targets_pro.view(targets.size(0),-1)#N*C
    input_normal = nn.functional.normalize(input_pro,p=2,dim=1) # 正则化
    targets_normal = nn.functional.normalize(targets_pro,p=2,dim=1)
    res = (input_normal - targets_normal)
    res = res * res
    loss = torch.mean(res)
    return loss

    
def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def dice_loss1(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target)
    z_sum = torch.sum(score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def entropy_loss(p, C=2):
    # p N*C*W*H*D
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1) / \
        torch.tensor(np.log(C)).cuda()
    ent = torch.mean(y1)

    return ent


def softmax_dice_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    n = input_logits.shape[1]
    dice = 0
    for i in range(0, n):
        dice += dice_loss1(input_softmax[:, i], target_softmax[:, i])
    mean_dice = dice / n

    return mean_dice


def entropy_loss_map(p, C=2):
    ent = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                       keepdim=True)/torch.tensor(np.log(C)).cuda()
    return ent


def softmax_mse_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax-target_softmax)**2
    return mse_loss


def softmax_kl_loss(input_logits, target_logits, sigmoid=False):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    if sigmoid:
        input_log_softmax = torch.log(torch.sigmoid(input_logits))
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)

    # return F.kl_div(input_log_softmax, target_softmax)
    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='mean')
    # mean_kl_div = torch.mean(0.2*kl_div[:,0,...]+0.8*kl_div[:,1,...])
    return kl_div


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    return torch.mean((input1 - input2)**2)

# def compute_max_loss(predict, target, percent, pred_teacher):
#     batch_size, num_class, h, w = predict.shape

#     with torch.no_grad():
#         prob = torch.softmax(pred_teacher, dim=1)   #得到概率
#         entropy = calculate_entropy(prob)  #计算熵
#         reliable_thresh = np.percentile(entropy[target!=255].detach().cpu().numpy().flatten(), percent)  #计算阈值

#         unreliable_mask = entropy.ge(reliable_thresh).bool()*target.ne(255).bool()  #找到熵大于阈值的像素点,即为不可靠像素点

#         target[unreliable_mask] = 255  #将不可靠像素点的标签设置为255
#         weight =  batch_size * h * w / torch.sum(target != 255)
#     #计算交叉熵损失
#     # loss = weight*F.cross_entropy(predict, target, ignore_index=255)
#     loss = F.cross_entropy(predict, target, ignore_index=255)


#     return loss
def compute_max_loss(predict, target, percent, pred_teacher):
    batch_size, num_class, h, w = predict.shape

    with torch.no_grad():
        prob = torch.softmax(pred_teacher, dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

        thresh = np.percentile(
            entropy[target != 255].detach().cpu().numpy().flatten(), percent
        )
        thresh_mask = entropy.ge(thresh).bool() * (target != 255).bool()

        target[thresh_mask] = 255
        weight = batch_size * h * w / torch.sum(target != 255)

    loss = weight * F.cross_entropy(predict, target, ignore_index=255)
    return loss

def calculate_entropy(probs):
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=1)
    return entropy


      
class EntropyMinimization(nn.Module):
    def __init__(self, reduction='mean'):
        super(EntropyMinimization, self).__init__()
        self.reduction = reduction

    def forward(self, inputs):
        P = torch.softmax(inputs, dim=1)
        logP = torch.log_softmax(inputs, dim=1)
        PlogP = P * logP
        loss_ent = -1.0 * PlogP.sum(dim=1)
        if self.reduction == 'mean':
            loss_ent = torch.mean(loss_ent)
        elif self.reduction == 'sum':
            loss_ent = torch.sum(loss_ent)
        else:
            pass

        return loss_ent
    
def compute_contra_memobank_loss(
    rep, #特征表示
    label_l,#有标签数据
    label_u,#无标签数据
    prob_l,#有标签数据的预测概率
    prob_u,#无标签数据的预测概率
    low_mask,#低熵掩码
    high_mask,#高熵掩码
    cfg,#配置参数
    memobank,#存储库
    queue_prtlis,#队列指针
    queue_size,#队列大小
    rep_teacher,#教师网络的特征表示，通常用于提供辅助信息或监督信号
    momentum_prototype=None,#动量原型，用于保存动态更新的原型特征
    i_iter=0,
):
    # current_class_threshold: delta_p (0.3)
    # current_class_negative_threshold: delta_n (1)
    
    current_class_threshold = cfg["current_class_threshold"]#当前类别阈值用于筛选低熵像素的阈值，设定为0.3
    current_class_negative_threshold = cfg["current_class_negative_threshold"]#当前类别负阈值，用于筛选高熵像素的阈值，设定为1
    low_rank, high_rank = cfg["low_rank"], cfg["high_rank"] #低秩高秩的设定
    temp = cfg["temperature"] #温度参数，控制softmax
    num_queries = cfg["num_queries"] #查询数量
    num_negatives = cfg["num_negatives"]#负样本数量

    num_feat = rep.shape[1] #获取特征表示
    num_labeled = label_l.shape[0] #有标签数据的数量
    num_segments = label_l.shape[1]#标签数据中的分段数量

    low_valid_pixel = torch.cat((label_l, label_u), dim=0) * low_mask  #低熵掩码中为 1 的位置保留原始数据，而对为 0 的位置置为 0，实现筛选作用
    high_valid_pixel = torch.cat((label_l, label_u), dim=0) * high_mask #对高熵掩码中为 1 的位置保留原始数据，而对为 0 的位置置为 0

    # print("low_valid_pixel.shape:",low_valid_pixel.shape)  #low_valid_pixel.shape: torch.Size([2, 224, 224])

    #维度变换
    rep = rep.permute(0, 2, 3, 1)
    rep_teacher = rep_teacher.permute(0, 2, 3, 1)

    # print("rep.shape:",rep.shape)  #rep.shape: torch.Size([2, 56, 96, 56])

    #用于存储不同类别的特征信息、低熵像素数据、每个类别中低熵像素的数量以及类别中心等信息
    seg_feat_all_list = []
    seg_feat_low_entropy_list = []  # candidate anchor pixels
    seg_num_list = []  # the number of low_valid pixels in each class 低熵有效像素
    seg_proto_list = []  # the center of each class

    #对有标签数据的预测概率 prob_l 和无标签数据的预测概率 prob_u 进行排序操作，按照概率值从大到小进行排序
    _, prob_indices_l = torch.sort(prob_l, 1, True)
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)  # (num_labeled, h, w, num_cls)

    _, prob_indices_u = torch.sort(prob_u, 1, True)
    prob_indices_u = prob_indices_u.permute(
        0, 2, 3, 1
    )  # (num_unlabeled, h, w, num_cls)

    #将有标签数据和无标签数据的预测概率整合到一个张量中
    prob = torch.cat((prob_l, prob_u), dim=0)  # (batch_size, num_cls, h, w)

    #存储有效类别和新的关键信息
    valid_classes = []
    new_keys = []
    #循环处理每个类别
    for i in range(num_segments):
        #针对每个类别，从低熵和高熵中选择二进制掩码
        low_valid_pixel_seg = low_valid_pixel[:, i]  # select binary mask for i-th class
        high_valid_pixel_seg = high_valid_pixel[:, i]

        # print("low_valid_pixel[:, i].shape:",low_valid_pixel[:, i].shape)  #low_valid_pixel_seg.shape: torch.Size([2, 224, 224])

        #从整合的预测概率中提取特定类别的掩码
        prob_seg = prob[:, i, :, :]
        # print("prob_seg.shape:",prob_seg.shape)   #prob_seg.shape: torch.Size([2, 224, 224])
        # print("low_valid_pixel_seg.bool().shape:",low_valid_pixel_seg.bool().shape)  #torch.Size([2, 224, 224])
        #保留概率高于阈值且为低熵有效像素的部分
        rep_mask_low_entropy = (
            prob_seg > current_class_threshold #大于低熵阈值
        ) * low_valid_pixel_seg.bool()
        #保留概率低于阈值且为高熵有效像素的部分
        rep_mask_high_entropy = (
            prob_seg < current_class_negative_threshold
        ) * high_valid_pixel_seg.bool()

        #提取出低熵有效像素对应的特征数据，并将其添加到列表中
        seg_feat_all_list.append(rep[low_valid_pixel_seg.bool()])
        #提取出低熵有效像素且概率高于阈值的特征数据，并将其添加到列表中
        seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy])

        # positive sample: center of the class 正样本，可能是所有锚点的中心
        #对于每个类别 计算正样本中心，被选为正样本的是低熵有效像素的教师网络特征数据
        seg_proto_list.append(
            torch.mean(
                rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True
            )
        )

        # generate class mask for unlabeled data 生成无标签数据的类别掩码
        # prob_i_classes = prob_indices_u[rep_mask_high_entropy[num_labeled :]]
        #针对高熵有效像素的掩码 rep_mask_high_entropy，选择其中从第 num_labeled 个样本之后的像素作为无标签数据
        #对指定范围内的预测概率索引是否等于当前类别 i 进行计算，生成一个布尔类型的类别掩码 class_mask_u，如果是i则为TRUE
        class_mask_u = torch.sum(
            #无标签数据的预测概率索引张量 prob_indices_u 中选择对应类别的预测概率索引
            prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3
        ).bool()

        # generate class mask for labeled data
        #针对高熵有效像素的掩码 rep_mask_high_entropy，选择其中前 num_labeled 个样本作为有标签数据
        # label_l_mask = rep_mask_high_entropy[: num_labeled] * (label_l[:, i] == 0) #高熵有效像素且标签为非当前类别 i 的样本
        # prob_i_classes = prob_indices_l[label_l_mask]
        #生成有标签数据中关于类别i的类别掩码
        class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool()

        #有标签数据中，高熵有效像素且标签为非当前类别 i 的样本与无标签数据的类别掩码进行拼接
        class_mask = torch.cat(
            (class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0
        )

        #负样本掩码 表示的是在高熵像素中属于类别i但在总体类别掩码中不属于i
        negative_mask = rep_mask_high_entropy * class_mask

        keys = rep_teacher[negative_mask].detach() #从教师特征表示中提取对应特征数据作为负样本的关键信息
        #将负样本的特征数据添加到循环队列中
        # print("keys.shape:",keys.shape)  #keys.shape: torch.Size([2, 96])
        # print("memobank[i].shape:",memobank[i][0].shape)  #memobank[i].shape: torch.Size([0, 96])
        new_keys.append(
            dequeue_and_enqueue(
                keys=keys,
                queue=memobank[i],
                queue_ptr=queue_prtlis[i],
                queue_size=queue_size[i],
            )
        )
        #若有低熵有效像素
        if low_valid_pixel_seg.sum() > 0:
            seg_num_list.append(int(low_valid_pixel_seg.sum().item())) #则将其低熵有效像素的数量到 seg_num_list 列表中
            valid_classes.append(i) #记录其类别信息

    if (
        len(seg_num_list) <= 1
    ):  # in some rare cases, a small mini-batch might only contain 1 or no semantic class   处理一些特殊情况，例如一个小的 mini-batch 只包含了一个或没有语义类别的情况
        if momentum_prototype is None:
            return new_keys, torch.tensor(0.0) * rep.sum() #返回新的关键信息和全0张量
        else:
            return momentum_prototype, new_keys, torch.tensor(0.0) * rep.sum() #返回动量原型，新的关键信息和全0张量

    else:
        reco_loss = torch.tensor(0.0).cuda() #初始化重构损失为全0张量
        #将有效类别原型列表 seg_proto_list 中的张量拼接起来，得到一个形状为 [valid_seg, 256] 的张量 seg_proto，即有效类别的原型
        seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256] 
        valid_seg = len(seg_num_list)  # number of valid classes  有效类别的数量

        #初始化原型为全零张量 存储原型信息
        prototype = torch.zeros(
            (prob_indices_l.shape[-1], num_queries, 1, num_feat)
        ).cuda()

        #对于每个有效类别
        for i in range(valid_seg):
            if (
                len(seg_feat_low_entropy_list[i]) > 0 
                and memobank[valid_classes[i]][0].shape[0] > 0
            ): #确保当前类别存在低熵特征列表并且记忆库中有相关数据
                # select anchor pixel
                #从seg_feat_low_entropy_list(低熵特征列表)中选择指定数量的索引作为锚点像素
                seg_low_entropy_idx = torch.randint(
                    len(seg_feat_low_entropy_list[i]), size=(num_queries,)
                )
                #锚点特征
                anchor_feat = (
                    seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().cuda()
                )
            else:#在某些情况下，当前查询类别中的所有查询都很容易
                # in some rare cases, all queries in the current query class are easy
                reco_loss = reco_loss + 0 * rep.sum()
                continue

            # apply negative key sampling from memory bank (with no gradients)
            #记忆库中负采样
            with torch.no_grad():
                negative_feat = memobank[valid_classes[i]][0].clone().cuda() #从记忆库中复制有效类别i作为负样本特征

                high_entropy_idx = torch.randint( #从负样本特征中选择一定数量的索引作为负样本
                    len(negative_feat), size=(num_queries * num_negatives,)
                )
                negative_feat = negative_feat[high_entropy_idx] #实现采样操作，从负样本中选择指定负样本
                negative_feat = negative_feat.reshape( #对负样本进行形状变换
                    num_queries, num_negatives, num_feat
                )
                positive_feat = ( #正样本特征
                    seg_proto[i] #取出有效类别i的原型特征
                    .unsqueeze(0)#扩展
                    .unsqueeze(0)
                    .repeat(num_queries, 1, 1)#将扩展后的特征复制多份，保持第一维是num_queries
                    .cuda()
                )  # (num_queries, 1, num_feat)

                if momentum_prototype is not None: #若动量原型不为空
                    if not (momentum_prototype == 0).all():#若动量原型全为0
                        if i_iter == 0:
                            ema_decay = 0.999
                        else:
                            ema_decay = min(1 - 1 / i_iter, 0.999)
                        # ema_decay = min(1 - 1 / i_iter, 0.999) #计算EMA 衰减率，取 1 - 1 / i_iter 和 0.999 中的较小值
                        positive_feat = (#将正样本特征根据EMA进行更新，得到新的正样本特征
                            1 - ema_decay
                        ) * positive_feat + ema_decay * momentum_prototype[
                            valid_classes[i]
                        ]#动量原型中对应当前有效类别的信息

                    prototype[valid_classes[i]] = positive_feat.clone()#更新原型

                all_feat = torch.cat(
                    (positive_feat, negative_feat), dim=1
                )  # (num_queries, 1 + num_negative, num_feat) #将负样本特征和正样本特征拼接

            #计算了锚点特征 anchor_feat 与所有特征 all_feat 之间的余弦相似度。
            seg_logits = torch.cosine_similarity(
                anchor_feat.unsqueeze(1), all_feat, dim=2
            )

            reco_loss = reco_loss + F.cross_entropy(
                seg_logits / temp, torch.zeros(num_queries).long().cuda()
            )

        if momentum_prototype is None:
            return new_keys, reco_loss / valid_seg #如果不存在动量原型，则返回更新后的键（new_keys）和归一化的重构损失
        else:
            return prototype, new_keys, reco_loss / valid_seg #更新后的原型、键和归一化的重构损失
def simple_contrastive_loss(
    rep, label_l,label_u,prob_l,prob_u, low_mask, high_mask, cfg, memobank, queue_prtlis, queue_size, rep_teacher
):
    current_class_threshold = cfg["current_class_threshold"]
    current_class_negative_threshold = cfg["current_class_negative_threshold"]
    low_rank, high_rank = cfg["low_rank"], cfg["high_rank"]
    temp = cfg["temperature"]
    num_queries = cfg["num_queries"]
    num_negatives = cfg["num_negatives"]
    num_feat = rep.shape[1]
    num_labeled = label_l.shape[0]
    num_segments = label_l.shape[1]

    label=torch.cat((label_l, label_u), dim=0)

    # print("label.shape:",label.shape)

    low_valid_pixel = label * low_mask
    high_valid_pixel = label * high_mask

    rep = rep.permute(0, 2, 3, 1)
    rep_teacher = rep_teacher.permute(0, 2, 3, 1)

    seg_feat_low_entropy_list, seg_proto_list, seg_num_list = [], [], []

    _, prob_indices_l = torch.sort(prob_l, 1, True)
    prob_indices_l = prob_indices_l.permute(0, 2, 3, 1)

    _, prob_indices_u = torch.sort(prob_u, 1, True)
    prob_indices_u = prob_indices_u.permute(0, 2, 3, 1)
    prob = torch.cat((prob_l, prob_u), dim=0)

    valid_classes = []
    new_keys = []
    for i in range(label.shape[1]):
        low_valid_pixel_seg = low_valid_pixel[:, i]
        high_valid_pixel_seg = high_valid_pixel[:, i]
        prob_seg = prob[:, i, :, :]

        rep_mask_low_entropy = (prob_seg > current_class_threshold) * low_valid_pixel_seg.bool()
        rep_mask_high_entropy = (prob_seg < current_class_negative_threshold) * high_valid_pixel_seg.bool()
        seg_feat_low_entropy_list.append(rep[rep_mask_low_entropy])

        seg_proto_list.append(
            torch.mean(rep_teacher[low_valid_pixel_seg.bool()].detach(), dim=0, keepdim=True)
        )

        class_mask_u = torch.sum(prob_indices_u[:, :, :, low_rank:high_rank].eq(i), dim=3).bool()

        class_mask_l = torch.sum(prob_indices_l[:, :, :, :low_rank].eq(i), dim=3).bool()

        class_mask = torch.cat((class_mask_l * (label_l[:, i] == 0), class_mask_u), dim=0)

        negative_mask = rep_mask_high_entropy * class_mask
        keys = rep_teacher[negative_mask].detach()
        new_keys.append(dequeue_and_enqueue(
            keys=keys, queue=memobank[i], queue_ptr=queue_prtlis[i], queue_size=queue_size[i]
        ))

        if low_valid_pixel_seg.sum() > 0:
            seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
            valid_classes.append(i)

    if len(seg_num_list) <= 1:
        return new_keys,torch.tensor(0.0)*rep.sum()
    else:

        reco_loss = torch.tensor(0.0).cuda()
        contra_loss=torch.tensor(0.0).cuda()
        for i in range(len(seg_num_list)):
            if (len(seg_feat_low_entropy_list[i]) > 0 and memobank[valid_classes[i]][0].shape[0] > 0):
                seg_low_entropy_idx = torch.randint(len(seg_feat_low_entropy_list[i]), size=(num_queries,))
                anchor_feat = seg_feat_low_entropy_list[i][seg_low_entropy_idx].clone().cuda()
            else:
                reco_loss = reco_loss + 0 * rep.sum()
                continue

            with torch.no_grad():
                negative_feat = memobank[valid_classes[i]][0].clone().cuda()
                high_entropy_idx = torch.randint(len(negative_feat), size=(num_queries * num_negatives,))
                negative_feat = negative_feat[high_entropy_idx]
                # print("negative_feat.shape",negative_feat.shape)    #negative_feat.shape torch.Size([12800, 96])
                negative_feat = negative_feat.reshape(num_queries, num_negatives, num_feat)

                positive_feat = seg_proto_list[i].repeat(num_queries, 1).cuda()

            all_feat = torch.cat((positive_feat.unsqueeze(1), negative_feat), dim=1)
            seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)

            reco_loss = reco_loss + (
                F.cross_entropy(seg_logits / temp, torch.zeros(seg_logits.shape[0]).long().cuda())
            ).mean()
            
            contra_loss=reco_loss / len(seg_num_list)

    return contra_loss,new_keys

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    # Using function "sum" and "mean" are depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


###############################################
# BCE = torch.nn.BCELoss()

def weighted_loss(pred, mask):
    BCE = torch.nn.BCELoss(reduction = 'none')
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask).float()
    wbce = BCE(pred, mask)
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()  



def calc_loss(pred, target, bce_weight=0.5):
    bce = weighted_loss(pred, target)
    # dl = 1 - dice_coef(pred, target)
    # loss = bce * bce_weight + dl * bce_weight

    return bce


def loss_sup(logit_S1, logit_S2, labels_S1, labels_S2):
    loss1 = calc_loss(logit_S1, labels_S1)
    loss2 = calc_loss(logit_S2, labels_S2)

    return loss1 + loss2



def loss_diff(u_prediction_1, u_prediction_2, batch_size):
    a = weighted_loss(u_prediction_1, Variable(u_prediction_2, requires_grad=False))
#     print('a',a.size())
    a = a.item()

    b = weighted_loss(u_prediction_2, Variable(u_prediction_1, requires_grad=False))
    b = b.item()

    loss_diff_avg = (a + b)
#     print('loss_diff_avg',loss_diff_avg)
#     print('loss_diff batch size',batch_size)
#     return loss_diff_avg / batch_size
    return loss_diff_avg 



###############################################
#contrastive_loss

class ConLoss(torch.nn.Module):
#for unlabel data
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mask_dtype = torch.bool
#         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
#         width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)  #batch * dim * np  # batch * np * dim
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))  #(batch * np) * 1 * dim #(batch * np) * dim * 1  #(batch * np) * 1
        l_pos = l_pos.view(-1, 1) #(batch * np) * 1

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)  #batch * np * dim
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))  # batch * np * np

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))
        l_neg = l_neg_curbatch.view(-1, npatches)  #(batch * np) * np

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  #(batch * np) * (np+1)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
    
    
# class MocoLoss(torch.nn.Module):
# #for unlabel data
#     def __init__(self, temperature=0.07):

#         super(MocoLoss, self).__init__()
#         self.temperature = temperature
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

#     def forward(self, feat_q, feat_k, queue):
#         assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
#         feat_q = F.normalize(feat_q, dim=-1, p=1)
#         feat_k = F.normalize(feat_k, dim=-1, p=1)
#         batch_size = feat_q.shape[0]
#         dim = feat_q.shape[1]
#         K = len(queue)
# #         print('K',K)

#         feat_k = feat_k.detach()

#         # pos logit
#         l_pos = torch.bmm(feat_q.view(batch_size,1,dim),feat_k.view(batch_size,dim,1))  #batch_size * 1
#         l_pos = l_pos.view(-1, 1)
#         feat_k = feat_k.transpose(0,1)
# #         print('feat_k',feat_k.size())
#         # neg logit
#         if K == 0:
#             l_neg = torch.mm(feat_q.view(batch_size,dim), feat_k)
#         else:
            
#             queue_tensor = torch.cat(queue,dim = 1)
# #             print('queue_tensor.size()',queue_tensor.size())
        
#             l_neg = torch.mm(feat_q.view(batch_size,dim), queue_tensor) #batch_size * K
# #         print(l_pos.size())
# #         print(l_neg.size())

#         out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  #batch_size * (K+1)
        
# #         print(1)
        
#         loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
#                                                         device=feat_q.device))
# #         print(2)
        
#         queue.append(feat_k)
        
#         if K >= 10:
#             queue.pop(0)

#         return loss,queue
    
    
class contrastive_loss_sup(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(contrastive_loss_sup, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mask_dtype = torch.bool
#         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
#         width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
#         l_pos = torch.zeros((batch_size*2304,1)).cuda()
#         l_pos = torch.zeros((batch_size*1024,1)).cuda()
#         l_pos = torch.zeros((batch_size*784,1)).cuda()
        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))
        l_neg = l_neg_curbatch.view(-1, npatches)
        l_pos = torch.zeros((l_neg.size(0),1)).cuda()
        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
    
def info_nce_loss(feats1,feats2):
#     imgs, _ = batch
#     imgs = torch.cat(imgs, dim=0)

    # Encode all images
#     feats = self.convnet(imgs)
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(feats1[:,None,:], feats2[None,:,:], dim=-1)
    # Mask out cosine similarity to itself
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
    cos_sim.masked_fill_(self_mask, -9e15)
    # Find positive example -> batch_size//2 away from the original example
    pos_mask = self_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
    # InfoNCE loss
    cos_sim = cos_sim / 0.07
    nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
    nll = nll.mean()

    # Logging loss
#     self.log(mode+'_loss', nll)
    # Get ranking position of positive example
#     comb_sim = torch.cat([cos_sim[pos_mask][:,None],  # First position positive example
#                               cos_sim.masked_fill(pos_mask, -9e15)],
#                              dim=-1)
#     sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
#     # Logging ranking metrics
#     self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
#     self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
#     self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())

    return nll

class contrastive_loss_sup(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(contrastive_loss_sup, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
#         self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.mask_dtype = torch.bool
#         self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
#         width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))  
        l_pos = l_pos.view(-1, 1) 
        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))
        l_neg = l_neg_curbatch.view(-1, npatches)
#         l_pos = torch.zeros((l_neg.size(0),1)).cuda()
        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss

class MocoLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, use_queue = True, max_queue = 1):

        super(MocoLoss, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.use_queue = use_queue
        self.mask_dtype = torch.bool
        self.queue = OrderedDict()
        self.idx_list = []
        self.max_queue = max_queue

    def forward(self, feat_q, feat_k, idx):
        num_enqueue = 0
        num_update = 0
        num_dequeue = 0
        mid_pop = 0
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        dim = feat_q.shape[1]
        batch_size = feat_q.shape[0]
        feat_q = feat_q.reshape(batch_size,-1)  
        feat_k = feat_k.reshape(batch_size,-1)

        K = len(self.queue)
#         print(K)

        feat_k = feat_k.detach()

        # pos logit
        l_pos = F.cosine_similarity(feat_q,feat_k,dim=1)        
        l_pos = l_pos.view(-1, 1)

        # neg logit
        if K == 0 or not self.use_queue:
            l_neg = F.cosine_similarity(feat_q[:,None,:], feat_k[None,:,:], dim=-1)
        else:
            for i in range(0,batch_size):
                if str(idx[i].item()) in self.queue.keys():
                    self.queue.pop(str(idx[i].item()))
                    mid_pop += 1
            queue_tensor = torch.cat(list(self.queue.values()),dim = 0)
            l_neg = F.cosine_similarity(feat_q[:,None,:], queue_tensor.reshape(-1,feat_q.size(1))[None,:,:], dim=-1)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  #batch_size * (K+1)
        
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        
        if self.use_queue:
            for i in range(0,batch_size):
                if str(idx[i].item()) not in self.queue.keys():
                    self.queue[str(idx[i].item())] = feat_k[i].clone()[None,:]
                    num_enqueue += 1
                else:
                    self.queue[str(idx[i].item())] = feat_k[i].clone()[None,:]
                    num_update += 1
                if len(self.queue) >= 1056 + 1:
                    self.queue.popitem(False)

                    num_dequeue += 1

#         print('queue length, mid pop, enqueue, update queue, dequeue: ', len(self.queue), mid_pop, num_enqueue, num_update, num_dequeue)

        return loss

class ConLoss_queue(torch.nn.Module):
#for unlabel data
    def __init__(self, temperature=0.07, use_queue = True, max_queue = 1):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ConLoss_queue, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool
        self.queue = OrderedDict()
        self.idx_list = []
        self.max_queue = max_queue


    def forward(self, feat_q, feat_k):
        num_enqueue = 0
        num_update = 0
        num_dequeue = 0
        mid_pop = 0
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
#         width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)  #batch * dim * np  # batch * np * dim
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))  #(batch * np) * 1 * dim #(batch * np) * dim * 1  #(batch * np) * 1
        l_pos = l_pos.view(-1, 1) #(batch * np) * 1

        # neg logit

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_size, -1, dim)  #batch * np * dim
        feat_k = feat_k.reshape(batch_size, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))  # batch * np * np

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -float('inf'))
        l_neg = l_neg_curbatch.view(-1, npatches)  #(batch * np) * np

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  #(batch * np) * (np+1)

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss
    

class MocoLoss_list(torch.nn.Module):
    def __init__(self, temperature=0.07, use_queue = True):

        super(MocoLoss_list, self).__init__()
        self.temperature = temperature
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.use_queue = use_queue
        self.queue = []
        self.mask_dtype = torch.bool
        self.idx_list = []

    def forward(self, feat_q, feat_k, idx):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        dim = feat_q.shape[1]
        batch_size = feat_q.shape[0]
        feat_q = feat_q.reshape(batch_size,-1)  #转成向量
        feat_k = feat_k.reshape(batch_size,-1)

        K = len(self.queue)
#         print('K',K)

        feat_k = feat_k.detach()

        # pos logit
        l_pos = F.cosine_similarity(feat_q,feat_k,dim=1)        
        l_pos = l_pos.view(-1, 1)

        # neg logit
        if K == 0 or not self.use_queue:
            l_neg = F.cosine_similarity(feat_q[:,None,:], feat_k[None,:,:], dim=-1)
        else:            
            queue_tensor = torch.cat(self.queue,dim = 0)
            print(queue_tensor.size())
            l_neg = F.cosine_similarity(feat_q[:,None,:], queue_tensor.reshape(-1,feat_q.size(1))[None,:,:], dim=-1)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature  #batch_size * (K+1)
        
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))
        if self.use_queue:
            self.queue.append(feat_k.clone())
#             for i in range(0,24):
#                 if idx[i] not in self.idx_list and len(self.queue) <512:
# #                     print(idx[i].item())
# #                     print(self.idx_list)
#                     self.idx_list.append(idx[i].item())                    
#                     self.queue.append(feat_k[i].clone()[None,:])
#                     print('LIST',len(self.idx_list))
#                     print('1',feat_k[i][None,:].size())
#                 elif idx[i] in self.idx_list:
#                     print('duplicate')
            if K >= 512:
#                 print('pop')
                self.queue.pop(0)
#                 self.idx_list.pop(0)

        return loss