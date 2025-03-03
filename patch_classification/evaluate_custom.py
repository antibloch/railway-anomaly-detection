import os
import numpy as np

import presets
import torch
import torch.utils.data
import torchvision
import utils
from torch import nn
import json
from PIL import Image
from PIL import ImageDraw

from dataset import FishyrailsCroppedDataset, RailSem19CroppedDatasetLikeFishyrails
from autoencoder_networks import AeSegParam02
from torchgeometry.losses.ssim import SSIM
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from patchclass_networks import PatchClassModel, PatchSegModelLight
from torchvision.transforms import functional as F
import cv2
import time
from skimage.filters import sato
from skimage import io





def min_distance_between_contours(contour1, contour2):
    min_dist = float('inf')
    
    for pt1 in contour1:
        for pt2 in contour2:
            dist = np.linalg.norm(pt1[0] - pt2[0])  # Euclidean distance
            if dist < min_dist:
                min_dist = dist

    return min_dist



def process_image(image_a):
    # Ensure the image is already grayscale and uint8
    assert image_a.dtype == np.uint8, "Image must be of type uint8"
    assert len(image_a.shape) == 2, "Image must be grayscale with shape (224, 224)"

    # Apply Canny Edge Detection directly
    edged = cv2.Canny(image_a, 30, 200)

    # Find contours
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    print("Number of Contours found =", len(contours))

    # Convert grayscale image to BGR for visualization
    image_with_contours = cv2.cvtColor(image_a, cv2.COLOR_GRAY2BGR)
    
    # Draw contours on the converted image
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 1)  # Thickness = 1 for clarity

    # Display images using matplotlib
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(edged, cmap='gray')
    ax[0].set_title("Canny Edges")
    ax[0].axis('off')

    ax[1].imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Contours on Image")
    ax[1].axis('off')

    plt.show()

    return image_with_contours 


def intersection_img(image_a, image_b):
    return np.logical_and(image_a == 255, image_b == 255).astype(np.uint8) * 255


def union_img(image_a, image_b):
    return np.logical_or(image_a == 255, image_b == 255).astype(np.uint8) * 255


def residual_img(image_a, image_b):
    union = union_img(image_a, image_b)
    intersection = intersection_img(image_a, image_b)
    return cv2.subtract(union, intersection)


def compliment_img(image_a, image_b):
    intersection = intersection_img(image_a, image_b)
    return cv2.subtract(image_a, intersection)




def get_tubes(ae_image):
    ae_image_grey = cv2.cvtColor(ae_image, cv2.COLOR_RGB2GRAY)

    g_otsu, _, l_img = get_components(ae_image_grey, p = 0.15)
    g_otsu = (g_otsu > 0).astype(np.uint8)
    tube_image = sato(g_otsu, sigmas=range(1, 5, 2), black_ridges=False)

    tube_image_norm = cv2.normalize(tube_image, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8 (OpenCV requires uint8 for thresholding)
    tube_image_uint8 = np.uint8(tube_image_norm)
    
    p = 0.55
    _ , ae_tube = cv2.threshold(tube_image_uint8,int(p*255),255,cv2.THRESH_BINARY)

    return ae_tube


def get_components(output_seg_np, p = 0.55):

    _,greyscale_otsu = cv2.threshold(output_seg_np,int(p*255),255,cv2.THRESH_BINARY)
    

    binary_image = (greyscale_otsu > 0).astype(np.uint8)  # Convert 255 to 1
    num_labels, labels = cv2.connectedComponents(binary_image)
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    original_labelled = labeled_img.copy()

    uniques= np.unique(labeled_img)

    sizes = []
    for i in range(1, num_labels):  # start from 1 to ignore the background
        size = np.sum(labels == i)
        sizes.append(size)
    idx_max = np.argmax(sizes) + 1  # add 1 because sizes index starts from 0
    max_label = idx_max

    labeled_img[labels != max_label] = 0
    labeled_img[labels == max_label] = 255

    labelled_img = cv2.cvtColor(labeled_img, cv2.COLOR_BGR2GRAY)

    return greyscale_otsu, labelled_img, original_labelled




def find_object_in_rails(rail_image, object_img, threshold_intersect= 2 , threshold_std = 0.9):

    if np.sum(object_img==255) < 0.05 * object_img.shape[0] * object_img.shape[1]:
        return False
    
    rail_image_ref = rail_image.copy()
    object_img_ref = object_img.copy()

    # find connected components in the object image
    num_labels, labels = cv2.connectedComponents(object_img)

    # find connected components in the rail image
    num_labels_rail, labels_rail = cv2.connectedComponents(rail_image)


    binary_image = (rail_image > 0).astype(np.uint8)  # Convert 255 to 1
    num_labels, labels = cv2.connectedComponents(binary_image)
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0


    # Initialize pix_image as an empty image (all zeros)
    pix_image = np.zeros_like(rail_image, dtype=np.uint8)
    
    
    dilation_record_per_component = []
    
    
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        pix_image[labels == i] = 255  # Set pixels belonging to label `i` to 255

        dilation_kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel of ones

        for dil_amount in range(100):
            if dil_amount > 0:
                # Apply dilation
                pix_image = cv2.dilate(pix_image, dilation_kernel, iterations=1)

            amount_intersection = np.sum(intersection_img(pix_image, object_img)==255)

            if amount_intersection > threshold_intersect:
                dilation_record_per_component.append(dil_amount)
                
                break


    std_dilations = np.std(dilation_record_per_component)

    if std_dilations > threshold_std:
        return True
    
    else:
        return False
    




            


def evaluate(args, ae_model, model, data_loader, device, num_classes, vis_path=None, mean=None, std=None):
    if ae_model:
        ae_model.eval()
    if model:
        model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    if vis_path:
        utils.mkdir(vis_path)
    header = "Test:"

    stat_book = dict()
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    stat_book["image_log"] = list()
    stat_book["images_fp"] = list()
    stat_book["images_fn_obstacle"] = list()
    stat_book["images_fn_correct"] = list()
    stat_book["conf_obstacle"] = list()
    stat_book["conf_correct"] = list()
    for thr_idx in range(len(thresholds)):
        stat_book["image_log"].append(list())
        stat_book["images_fp"].append(list())
        stat_book["images_fn_obstacle"].append(list())
        stat_book["images_fn_correct"].append(list())
        stat_book["conf_obstacle"].append({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        stat_book["conf_correct"].append({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    stat_book["auroc_data"] = np.zeros((1001, 2))

    overall_max = 1
    print(f"Overall Max: {overall_max}")


    with torch.no_grad():


        current_time = time.time()
        
        
        image_curr = cv2.imread(args.img)
        image_curr = cv2.cvtColor(image_curr, cv2.COLOR_BGR2RGB)
        image_curr_ref = image_curr.copy()
        image_curr_ref = cv2.resize(image_curr_ref, (224, 224))


        image_tensor = torch.from_numpy(image_curr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

        # image_tensor = image_tensor / 255.0
        image_tensor = image_tensor / 255.0
        image_tensor_ref= torchvision.transforms.functional.resize(image_tensor, (224, 224)).to(device)

        image_tensor_ref = torchvision.transforms.functional.normalize(image_tensor_ref, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        image_fishy = image_tensor_ref.clone()
        image_orig = image_tensor_ref.clone()
        target_fishy = torch.zeros(224, 224)
        target_orig = torch.zeros(224, 224)




        # Go over fishy:
        for mode in ["fishy"]:
            #time_prep_start = time.time()
            # Prepare everything
            if mode == "fishy":
                image, target_seg = image_fishy.to(device), target_fishy.to(device)
            elif mode == "orig":
                image, target_seg = image_orig.to(device), target_orig.to(device)

            target_seg_orig = target_orig.clone().to(device)

            # Mask for evaluation (discard background)
            if args.rails_only > 0:
                evaluation_mask = target_seg_orig == 1
                target_seg[torch.logical_not(evaluation_mask)] = 0
                evaluation_mask = evaluation_mask.squeeze()
            else:
                evaluation_mask = torch.logical_or(target_seg == 2, target_seg == 1).squeeze()

            # Visualize original image
            if args.g_act == "tanh":
                image_target_ae, _ = presets.denormalize_tanh(image, image)  # (-1, 1)
                image_vis, _ = presets.re_convert_tanh(image_target_ae, image_target_ae)
            else:
                image_target_ae, _ = presets.denormalize(image, image)  # (0, 1)
                image_vis, _ = presets.re_convert(image_target_ae, image_target_ae)
            VIS_INPUT = image_vis

            if ae_model and args.ae_model != "patchsegmodellight":

                # image_curr = cv2.imread("sample_rail.jpeg")
                # image_curr = cv2.cvtColor(image_curr, cv2.COLOR_BGR2RGB)


                # image_tensor = torch.from_numpy(image_curr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)

                # # image_tensor = image_tensor / 255.0
                # image_tensor = image_tensor / 255.0
                # image= torchvision.transforms.functional.resize(image_tensor, (224, 224)).to(device)



                # Run  AE inference
                with torch.no_grad():
                    outputs = ae_model(image)
                output_ae = outputs["out_aa"]


                ae_tensor= output_ae

                ae_image = ae_tensor[0].permute(1, 2, 0).cpu().numpy()

                ae_image = (ae_image * 255).clip(0, 255).astype(np.uint8)

                # Post-process AE image for PatchSeg
                if args.g_act == "tanh":
                    image_ae = (output_ae / 2) + 0.5
                else:
                    image_ae = output_ae
                image_ae = torchvision.transforms.functional.normalize(image_ae, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

                # Visualize AE image
                if args.g_act == "tanh":
                    image_ae_vis, _ = presets.re_convert_tanh(output_ae, output_ae)  # no de-normalization
                else:
                    image_ae_vis, _ = presets.re_convert(output_ae, output_ae)  # no de-normalization
                VIS_OUTPUT = image_ae_vis
            else:
                VIS_OUTPUT = None

            # Prepare input for PatchSeg model
            if "patch30" in args.model:
                image_r1 = torch.zeros_like(image)
                image_r1[::, :-27] = image[::, 27:]
                image_r2 = torch.zeros_like(image)
                image_r2[::, :-54] = image[::, 54:]
                image_l1 = torch.zeros_like(image)
                image_l1[::, 27:] = image[::, :-27]
                image_l2 = torch.zeros_like(image)
                image_l2[::, 54:] = image[::, :-54]
                image_ae_r1 = torch.zeros_like(image_ae)
                image_ae_r1[::, :-27] = image_ae[::, 27:]
                image_ae_r2 = torch.zeros_like(image_ae)
                image_ae_r2[::, :-54] = image_ae[::, 54:]
                image_ae_l1 = torch.zeros_like(image_ae)
                image_ae_l1[::, 27:] = image_ae[::, :-27]
                image_ae_l2 = torch.zeros_like(image_ae)
                image_ae_l2[::, 54:] = image_ae[::, :-54]
                input_seg = torch.cat((image_l2, image_l1, image, image_r1, image_r2, image_ae_l2, image_ae_l1, image_ae, image_ae_r1, image_ae_r2), dim=1)
            elif "patch15" in args.model:
                image_r1 = torch.zeros_like(image)
                image_r1[::, :-27] = image[::, 27:]
                image_r2 = torch.zeros_like(image)
                image_r2[::, :-54] = image[::, 54:]
                image_l1 = torch.zeros_like(image)
                image_l1[::, 27:] = image[::, :-27]
                image_l2 = torch.zeros_like(image)
                image_l2[::, 54:] = image[::, :-54]
                input_seg = torch.cat((image_l2, image_l1, image, image_r1, image_r2), dim=1)
            elif "patch6" in args.model:
                input_seg = torch.cat((image, image_ae), dim=1)

                # ref_image = image.permute(0, 2, 3, 1).cpu().numpy()
                # ref_image = (ref_image[0] * 255).astype(np.uint8)
                # # ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR)
                # plt.imshow(ref_image)
                # plt.show()
            else:
                input_seg = image

            # Inference
            if "patchclassmodel" in args.model:
                with torch.no_grad():
                    # image_tensor= input_seg[:,0:3,:,:]

                    # ref_image = image_tensor[0].permute(1, 2, 0).cpu().numpy()

                    # ref_image = (ref_image * 255).clip(0, 255).astype(np.uint8)



                    
                    output_seg = model(input_seg)["out"]
                    output_seg = nn.functional.softmax(output_seg, dim=1)
                    output_seg = output_seg[0, 0, ::]

                    print(f"Output_seg: {output_seg.shape}, max: {torch.max(output_seg)}, min: {torch.min(output_seg)}")
                    # fdgjd
            elif args.model == "deeplabv3_resnet50":
                with torch.no_grad():
                    output_seg = model(input_seg)["out"]
                    output_seg = nn.functional.softmax(output_seg, dim=1)
                    output_seg = output_seg[0, 0, ::]
            elif args.model == "mse":
                if args.g_act == "tanh":
                    image_target_ae = (image_target_ae / 2) + 0.5
                    output_ae = (output_ae / 2) + 0.5
                output_seg = torch.squeeze(torch.sqrt(torch.square(image_target_ae - output_ae)))
                output_seg = torch.mean(output_seg, dim=0)

                
            elif args.model == "ssim":
                ssim = SSIM(11)
                if args.g_act == "tanh":
                    image_target_ae = (image_target_ae / 2) + 0.5
                    output_ae = (output_ae / 2) + 0.5
                output_seg = torch.squeeze(ssim(image_target_ae, output_ae))*2 # SSIM output is in range (0, 0.5)
                output_seg = torch.mean(output_seg, dim=0)
            elif args.model == "patchsegmodellight" and args.ae_model == "patchsegmodellight": # Student Teacher
                with torch.no_grad():
                    outputs_teacher = ae_model(input_seg)["descriptor"]
                    normalized_teacher = F.normalize(outputs_teacher, mean=mean, std=std).clone().detach()
                    outputs_student = model(input_seg)["descriptor"]
                    output_seg = torch.squeeze(torch.sqrt(torch.square(normalized_teacher - outputs_student)))
                    # print(f"Output_seg: {output_seg.shape}, max: {torch.max(output_seg)}, min: {torch.min(output_seg)}")
                    # Make sure output_seg is in range (0, 1)
                    output_seg = torch.mean(output_seg, dim=0)

            # Make sure segmentation outputs are in range (0, 1):
            

            kernel = torch.tensor(
                np.ones((args.k_d, args.k_d)) * 1 / (args.k_d * args.k_d)
            ).view(1, 1, args.k_d, args.k_d).type(torch.FloatTensor).cuda()

            # Ensure output_seg has correct shape: (batch_size, channels, height, width)
            if output_seg.dim() == 3:  # If missing batch dimension
                output_seg = output_seg.unsqueeze(0)  # Add batch dim

            if output_seg.dim() == 2:  # If missing both batch & channel dims
                output_seg = output_seg.unsqueeze(0).unsqueeze(0)

            # Apply convolution
            output_seg = torch.nn.functional.conv2d(output_seg, kernel, padding=args.k_d // 2)

            output_seg = output_seg.squeeze().squeeze()


            output_seg = (output_seg - torch.min(output_seg))/ (torch.max(output_seg)-torch.min(output_seg))


            # output_seg [output_seg > 0.0] = 1.0
            # output_seg [output_seg <= 0.0] = 0.0
            final_time = time.time()

            print("----------------------------------------------------")
            print(f"Time Inference: {final_time-current_time} seconds")
            print("----------------------------------------------------")

            output_seg_np=output_seg.cpu().numpy()
            output_seg_np = (output_seg_np * 255).clip(0, 255)

            _ , object_img, _ = get_components(output_seg_np)

            ae_tube = get_tubes(ae_image)

            # things = process_image(ae_tube)

            # plt.imshow(object_img)
            # plt.show()

            # plt.imshow(ae_tube)
            # plt.show()

            object_plus_rail = union_img(object_img, ae_tube)
            object_bool = find_object_in_rails(ae_tube, object_img, threshold_intersect= 20 , threshold_std = 0.9)

            
            plt.figure()
            plt.subplot(1, 6, 1)
            
            plt.imshow(image_curr_ref)
            plt.title("Original Image")
            
            plt.subplot(1, 6, 2)
            plt.imshow(ae_image)
            plt.title("Autoencoder Output")

            plt.subplot(1, 6, 3)
            plt.imshow(ae_tube, cmap='gray')
            plt.title("Autoencoder -> Tubes")

            plt.subplot(1, 6, 4)
            plt.imshow(output_seg_np, cmap='gray')
            plt.title("Autoencoder + Differentiator -> Segmentation")

            plt.subplot(1, 6, 5)
            plt.imshow(object_img, cmap='gray')
            plt.title("Post Processed Segmentation")

            plt.subplot(1, 6, 6)
            plt.imshow(object_plus_rail, cmap='gray')
            plt.title(f"Object Found on Rail : {object_bool}")

            plt.show()

             


def main(args):

    print(args)
    device = torch.device(args.device)
    print(f"Running on device: {device}")

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    if "Railsem19" in args.data_path:
        dataset_test = RailSem19CroppedDatasetLikeFishyrails(args.data_path, mode="train")
    else:
        dataset_test = FishyrailsCroppedDataset(args.data_path)
    print("Dataset loaded.")

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=1
    )

    print("Dataloader created.")

    # Gan model
    if args.ae_model == "AeSegParam02_8810":
        ae_model = AeSegParam02(c_seg=8, c_ae=8, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_model == "AeSegParam02_8410":
        ae_model = AeSegParam02(c_seg=8, c_ae=4, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_model == "AeSegParam02_8210":
        ae_model = AeSegParam02(c_seg=8, c_ae=2, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_model == "AeSegParam02_8110":
        ae_model = AeSegParam02(c_seg=8, c_ae=1, c_param=1, mode="none", ratio=args.color_space_ratio, act=args.g_act)
    elif args.ae_model == "patchsegmodellight":
        ae_model = PatchSegModelLight(in_channels=3, out_channels=512, stages=args.stages, patch_only=False)
    else:
        ae_model = None
        print(f"No autoencoder used!")

    if ae_model:
        ae_model.to(device)
        ae_checkpoint = torch.load(args.ae_checkpoint, map_location="cpu")
        ae_model.load_state_dict(ae_checkpoint["model"], strict=False)
        print("AE Model loaded.")
        if args.ae_model == "patchsegmodellight" and args.model == "patchsegmodellight":
            args.mean_std_dir = os.path.dirname(args.ae_checkpoint)
            args.mean_std_suffix = os.path.basename(args.ae_checkpoint[:-4])
            with open(os.path.join(args.mean_std_dir, f"{args.mean_std_suffix}_mean.npy"), "rb") as file:
                mean = np.load(file)
            with open(os.path.join(args.mean_std_dir, f"{args.mean_std_suffix}_std.npy"), "rb") as file:
                std = np.load(file)
            mean = torch.from_numpy(mean).to(device)
            print(f"Mean shape: {mean.shape}")
            std = torch.from_numpy(std).to(device)
            print(f"Std shape: {std.shape}")
        else:
            mean = None
            std = None
    else:
        mean = None
        std = None

    # Segmentation model
    if args.model == "deeplabv3_resnet50":
        model = torchvision.models.segmentation.__dict__[args.model](
                pretrained=False,
                pretrained_backbone=False,
                num_classes=2,
                aux_loss=False,
            )
        #model.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    elif args.model == "patchsegmodellight_patch30":
        model = PatchSegModelLight(stages=args.stages, in_channels=30)
    elif args.model == "patchclassmodel_patch30":
        model = PatchClassModel(stages=args.stages, in_channels=30)
    elif args.model == "patchclassmodel_patch15":
        model = PatchClassModel(stages=args.stages, in_channels=15)
    elif args.model == "patchclassmodel_patch6":
        model = PatchClassModel(stages=args.stages, in_channels=6)
    elif args.model == "patchclassmodel_patch3":
        model = PatchClassModel(stages=args.stages, in_channels=3)
    elif args.model == "patchsegmodellight":
        model = PatchSegModelLight(in_channels=3, out_channels=512, stages=args.stages, patch_only=False)
    else:
        model = None
        print("No seg model!")

    if model:

        model.to(device)

        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)

    evaluate(args, ae_model, model, data_loader_test, device=device, num_classes=num_classes, vis_path=args.output_path, mean=mean, std=std)
    return


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--data_path", default="./datasets/FishyrailsCroppedDebug/FishyrailsCroppedDebug.h5", type=str, help="dataset path")
    parser.add_argument("--model", default="patchclassmodel_patch6", type=str, help="model name")
    parser.add_argument("--output_path", default="./evaluation", type=str, help="output directory")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu)")
    parser.add_argument("--checkpoint", default="./trained_models/patchclass_model_10.pth", type=str,
                        help="path of checkpoint")
    parser.add_argument("--ae_checkpoint", default="./trained_models/ganaesegparam02_8810_01000_017_model_199.pth", type=str, help="path of checkpoint")
    parser.add_argument("--ae_model", default="AeSegParam02_8810", type=str, help="Autoencoder Type")
    parser.add_argument("--color_space_ratio", default=0.1, type=float, help="color space ratio for each channel, NOT relevant for our experiments")
    parser.add_argument("--max_num_images", default=4000, type=int, help="max number of images to be evaluated")
    parser.add_argument("--g_act", default="tanh", type=str, help="generator activation")
    parser.add_argument("--theta_visualize", default=0.0, type=float, help="which obstacle threshold to visualize")
    parser.add_argument(
        "--seg_pretrained",
        dest="seg_pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument("--stages", default=1, type=int, help="Number of stages of the Patch Classification network. Stage 0 corresponds to patch size 13, 1 to 21, 2 to 29, 3 to 35, and 4 to 51.")
    parser.add_argument("--k_d", default=21, type=int, help="patch density size k_d")
    parser.add_argument("--rails_only", default=1, type=int, help="whether to evaluate on rails only")
    parser.add_argument("--img", required=True, type=str, help="target image path")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
