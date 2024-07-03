import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


def calculate_metric_percase(pred, gt):
    # pred[pred > 0] = 1
    # gt[gt > 0] = 1
    # if pred.sum() > 0:
    #     dice = metric.binary.dc(pred, gt)
    #     hd95 = metric.binary.hd95(pred, gt)
    #     return dice, hd95
    # else:
    #     return 0, 0
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes,model_type, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        outputs = net(input)
        with torch.no_grad():
            if model_type =='mambaunet':
                out = torch.argmax(torch.softmax(
                    outputs["pred"], dim=1), dim=1).squeeze(0)
            elif model_type =='mambaunet_origin':
                out = torch.argmax(torch.softmax(
                    outputs["pred"], dim=1), dim=1).squeeze(0)
            elif model_type =='unet_u2pl':
                out = torch.argmax(torch.softmax(
                    outputs["pred"], dim=1), dim=1).squeeze(0)
            else:
                out = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
    # image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # if len(image.shape) == 3:
    #     prediction = np.zeros_like(label)
    #     for ind in range(image.shape[0]):
    #         slice = image[ind, :, :]
    #         x, y = slice.shape[0], slice.shape[1]
    #         if x != patch_size[0] or y != patch_size[1]:
    #             slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
    #         input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
    #         net.eval()
    #         with torch.no_grad():
    #             outputs = net(input)
    #             out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
    #             out = out.cpu().detach().numpy()
    #             if x != patch_size[0] or y != patch_size[1]:
    #                 pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
    #             else:
    #                 pred = out
    #             prediction[ind] = pred
    # else:
    #     input = torch.from_numpy(image).unsqueeze(
    #         0).unsqueeze(0).float().cuda()
    #     net.eval()
    #     with torch.no_grad():
    #         out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
    #         prediction = out.cpu().detach().numpy()
    # metric_list = []
    # for i in range(1, classes):
    #     metric_list.append(calculate_metric_percase(prediction == i, label == i))
    # return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list
