import argparse
import datetime
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from networks.vision_mamba import MambaUnet as ViM_seg
from config import get_config
from dataloaders import utils
import yaml
from dataloaders.dataset import (BaseDataSets, RandomGenerator, BaseDataSets_Synapse,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume
from utils.losses import compute_contra_memobank_loss, compute_max_loss
from utils.util import label_onehot

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Prostate', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='Prostate/Semi_LMamba_UNet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mambaunet', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=1,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=4,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {'1':14,'2':28, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1311}
    # elif "Prostate":
    #     ref_dict = {"2": 27, "4": 53, "8": 120,
    #                 "12": 179, "16": 256, "21": 312, "42": 623}
    elif "Prostate" in dataset:
        ref_dict = {"2": 47, "4": 111, "8": 238,
                    "12": 351, "16": 438, "21": 563, "35": 940}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    cfg=yaml.load(open(args.cfg, 'r'), Loader=yaml.Loader)
    model_teacher = ViM_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    model_teacher.load_from(config)

    model2 = ViM_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes).cuda()
    model2.load_from(config)

    for param in model_teacher.parameters():
        param.detach_()

    # build class-wise memory bank
    memobank = []  
    queue_ptrlis = []  
    queue_size = [] 
    for i in range(num_classes):
        memobank.append([torch.zeros(0, 96)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000

    # build prototype
    prototype = torch.zeros(
        (
            num_classes,
            cfg["trainer"]["contrastive"]["num_queries"],
            1,
            96,
        )
    ).cuda()
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

   
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance2 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]
            labeled_batch = label_batch[:args.labeled_bs]

            # noise = torch.clamp(torch.randn_like(
            #     unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            # ema_inputs = unlabeled_volume_batch + noise
            outputs2 = model2(volume_batch)
            pred_all, rep_all = outputs2["pred"], outputs2["rep"]  
            outputs_soft2 = torch.softmax(pred_all, dim=1)
            consistency_weight = get_current_consistency_weight(
                iter_num // 150)
            with torch.no_grad():
                # outputs_teacher = model_teacher(ema_inputs)
                outputs_teacher = model_teacher(volume_batch)
                pred_all_teacher, rep_all_teacher = outputs_teacher["pred"], outputs_teacher["rep"]
                outputs_soft_teacher = torch.softmax(pred_all_teacher, dim=1)

            loss_ce = ce_loss(pred_all[:args.labeled_bs],
                              label_batch[:][:args.labeled_bs].long())
            loss_dice = dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            loss_sup=0.5 * (loss_dice + loss_ce)

            
            pseudo_outputs_teacher = torch.argmax(
                outputs_soft_teacher[args.labeled_bs:].detach(), dim=1, keepdim=False)     
            pseudo_outputs2 = torch.argmax(
                outputs_soft2[args.labeled_bs:].detach(), dim=1, keepdim=False)
            
            
            logits_u_teacher, label_u_teacher = torch.max(outputs_soft_teacher[args.labeled_bs:].detach(), dim=1)
            logits_u_aug, label_u_aug = torch.max(outputs_soft2[args.labeled_bs:].detach(), dim=1)


            pseudo_supervision1 = dice_loss(
                outputs_soft_teacher[args.labeled_bs:], pseudo_outputs2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(
                outputs_soft2[args.labeled_bs:], pseudo_outputs_teacher.unsqueeze(1))
            
            
            drop_percent = cfg["trainer"]["max_loss"].get("drop_percent", 100)  #80
            percent_unreliable = (100 - drop_percent) * (1 - epoch_num / max_epoch)  
            drop_percent = 100 - percent_unreliable 

            
            max_loss=(
                compute_max_loss(
                    pred_all[args.labeled_bs:],
                    label_u_teacher.clone(),
                    drop_percent,
                    pred_all_teacher[args.labeled_bs:].detach(),
                )
                * cfg["trainer"]["max_loss"].get("loss_weight", 1)
            )

            

            entropy_minimization = losses.EntropyMinimization(reduction='mean')
            min_loss = entropy_minimization(pred_all_teacher[args.labeled_bs:].detach())

            # contrastive loss using unreliable pseudo labels
            contra_flag = "none"
            if cfg["trainer"].get("contrastive", False):
                cfg_contra = cfg["trainer"]["contrastive"]
                contra_flag = "{}:{}".format(
                    cfg_contra["low_rank"], cfg_contra["high_rank"]
                )
                alpha_t = cfg_contra["low_entropy_threshold"] * (
                    1 - epoch_num / max_epoch
                )

                with torch.no_grad():
                    prob = torch.softmax(pred_all_teacher[args.labeled_bs:], dim=1)   
                    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1) 

                    low_thresh = np.percentile(
                        entropy[label_u_teacher != 255].cpu().numpy().flatten(), alpha_t
                    )
                    low_entropy_mask = (
                        entropy.le(low_thresh).float() * (label_u_teacher != 255).bool()
                    )

                    high_thresh = np.percentile(
                        entropy[label_u_teacher != 255].cpu().numpy().flatten(),
                        100 - alpha_t,
                    )
                    high_entropy_mask = (
                        entropy.ge(high_thresh).float() * (label_u_teacher != 255).bool()
                    )

                    low_mask_all = torch.cat(
                        (
                            (labeled_batch.unsqueeze(1) != 255).float(),
                            low_entropy_mask.unsqueeze(1),
                        )
                    )
                    low_mask_all = F.interpolate(
                        low_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )

                    if cfg_contra.get("negative_high_entropy", True):
                        contra_flag += " high"  
                        high_mask_all = torch.cat(
                            (
                                (labeled_batch.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                    else:
                        contra_flag += " low"
                        high_mask_all = torch.cat(
                            (
                                (labeled_batch.unsqueeze(1) != 255).float(),
                                torch.ones(logits_u_teacher.shape)
                                .float()
                                .unsqueeze(1)
                                .cuda(),
                            ),
                        )
                    high_mask_all = F.interpolate(
                        high_mask_all, size=pred_all.shape[2:], mode="nearest"
                    )  # down sample

                    # down sample and concat
                    label_l_small = F.interpolate(
                        label_onehot(labeled_batch, num_classes),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )
                    label_u_small = F.interpolate(
                        label_onehot(label_u_teacher, num_classes),
                        size=pred_all.shape[2:],
                        mode="nearest",
                    )
                    
                    prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                            rep_all,
                            label_l_small.long(),
                            label_u_small.long(),
                            outputs_soft_teacher[:args.labeled_bs].detach(),
                            outputs_soft_teacher[args.labeled_bs:].detach(),
                            low_mask_all,
                            high_mask_all,
                            cfg_contra,
                            memobank,
                            queue_ptrlis,
                            queue_size,
                            rep_all_teacher.detach(),
                            prototype,
                        )
                    
                    contra_loss = contra_loss * cfg_contra.get("loss_weight", 1)




            loss_total=loss_sup +consistency_weight * (pseudo_supervision1 + pseudo_supervision2) + contra_loss + max_loss +min_loss
            optimizer2.zero_grad()

            loss_total.backward()

            optimizer2.step()


            update_ema_variables(model2, model_teacher, args.ema_decay, iter_num)
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/loss_sup',
                              loss_sup, iter_num)
            writer.add_scalar('loss/max_loss',
                               max_loss, iter_num)
            writer.add_scalar('loss/min_loss',
                                min_loss, iter_num)
            writer.add_scalar('loss/constra_loss',
                              contra_loss, iter_num)
            
           
            writer.add_scalar('loss/loss', loss_total, iter_num)
            logging.info('iteration %d : loss_total : %f loss_sup : %f max_loss : %f min_loss : %f contra_loss : %f ' % (
                iter_num, loss_total.item(), loss_sup.item(), (max_loss*consistency_weight).item(), min_loss.item(), contra_loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    pred_all_teacher, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model_teacher_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    pred_all, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model2.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes,model_type=args.model, patch_size=args.patch_size)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/model2_val_{}_dice'.format(class_i+1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model2_val_{}_hd95'.format(class_i+1),
                                      metric_list[class_i, 1], iter_num)

                performance2 = np.mean(metric_list, axis=0)[0]

                mean_hd952 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model2_val_mean_dice',
                                  performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95',
                                  mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model2_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model2.pth'.format(args.model))
                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))
                model2.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num,args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)