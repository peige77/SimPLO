
import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Prostate', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='Prostate/Semi_UNet', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet_u2pl', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=16,
                    help='labeled data')
args = parser.parse_args()


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, jc, hd95, asd
    else:
        return 0, 0, 0, 0


def test_single_volume(case, net, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    # h5f = h5py.File(FLAGS.root_path + "/Synapse/test_vol_h5/{}.npy.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (224 / x, 224 / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        outputs = net(input)
        with torch.no_grad():
            if FLAGS.model == "unet_urpc":
                out_main, _, _, _ = outputs
            elif FLAGS.model == "mambaunet":
                out_main = outputs["pred"]
            elif FLAGS.model == "mambaunet_origin":
                out_main = outputs["pred"]
            elif FLAGS.model == "unet_u2pl":
                out_main = outputs["pred"]
            else:
                out_main = net(input)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)




            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / 224, y / 224), order=0)
            prediction[ind] = pred

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)
    # metric_list = []
    # for i in range(1, args.num_classes):
    #     metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    if np.sum(prediction == 2)==0:
        second_metric = 0,0,0,0
    else:
        second_metric = calculate_metric_percase(prediction == 2, label == 2)

    if np.sum(prediction == 3)==0:
        third_metric = 0,0,0,0
    else:
        third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    # sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    # sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    # sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric,second_metric,third_metric

def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    # snapshot_path = "../model/{}_{}_labeled/{}".format(
    snapshot_path = "../model/{}_{}/{}".format(        
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    # test_save_path = "../model/{}_{}_labeled/{}_predictions/".format(
    test_save_path = "../model/{}_{}/{}_predictions/".format(        
        FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    net = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    save_mode_path = os.path.join(
        snapshot_path, '{}_best_model2.pth'.format(FLAGS.model))
        # snapshot_path, 'mambaunet_best_model2.pth')
    print(save_mode_path)
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    for case in tqdm(image_list):
        first_metric ,second_metric,second_metric= test_single_volume(
            case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(second_metric)
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    # avg_metric = metric_list / len(image_list)
    return avg_metric,test_save_path


# if __name__ == '__main__':
#     FLAGS = parser.parse_args()
#     metric, test_save_path = Inference(FLAGS)
#     print(metric)
#     print((metric[0]+metric[1]+metric[2])/3)
#     with open(test_save_path+'../performance.txt', 'w') as f:
#         f.writelines('metric is {} \n'.format(metric))
#         f.writelines('average metric is {}\n'.format((metric[0]+metric[1]+metric[2])/3))
if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric, test_save_path = Inference(FLAGS)
    print(metric)
    print((metric[0]+metric[1]+metric[2])/3)
    
    performance_txt_path = os.path.join(os.path.dirname(test_save_path), '../performance.txt')
    
    mode = 'a' if os.path.exists(performance_txt_path) else 'w'
    
    with open(performance_txt_path, mode) as f:
        f.writelines('model \n')
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format((metric[0]+metric[1]+metric[2])/3))

