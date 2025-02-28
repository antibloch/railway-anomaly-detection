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
        for idx, (image_fishy, target_fishy, image_orig, target_orig) in enumerate(metric_logger.log_every(data_loader, 100, header)):

            print(image_fishy.shape)
            print(target_fishy.shape)
            print(image_orig.shape)
            print(target_orig.shape)

            current_time = time.time()
            
            
            image_curr = cv2.imread("sample_rail.jpeg")
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
            target_fishy = torch.zeros_like(target_fishy)
            target_orig = torch.zeros_like(target_orig)

            # print(image_fishy.shape)
            # print(target_fishy.shape)
            # print(image_orig.shape)
            # print(target_orig.shape)
            # aslkjdj

            # plt.imshow(image_curr)
            # plt.show()



            print(f"Image {idx} ...")
            if idx > args.max_num_images:
                break
            # Go over fishy:
            for mode in ["fishy", "orig"]:
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

                    # plt.imshow(ae_image)
                    # plt.show()
                    # sdfhn

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


                        # print(ref_image.shape)
                        # plt.imshow(ref_image)
                        # plt.show()
                        # sdfi

                        
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

                print(f"Time Inference: {final_time-current_time}")

                output_seg_np=output_seg.cpu().numpy()
                output_seg_np = (output_seg_np * 255).clip(0, 255)

                greyscale_otsu, labelled_img, _ = get_components(output_seg_np)


                # kernel = np.ones((5, 5), np.uint8)  # 5x5 kernel of ones

                # # Apply dilation
                # greyscale_otsu = cv2.dilate(greyscale_otsu, kernel, iterations=1)

                image_curr_ref_grey = cv2.cvtColor(image_curr_ref, cv2.COLOR_RGB2GRAY)

                ae_tube = get_tubes(ae_image)

                things = process_image(ae_tube)
                



                # print(np.min(ae_tube))
                # skd


                intersection = intersection_img(ae_tube, labelled_img)
                union = union_img(ae_tube, labelled_img)

                residual = residual_img(ae_tube, labelled_img)

                comp = compliment_img(ae_tube, labelled_img)

                plt.imshow(union,cmap='gray')
                plt.show()


                sdfjg




                


                plt.imshow(ae_image_grey)
                plt.show()

                plt.imshow(g_otsu)
                plt.show()

                plt.imshow(tube_image)
                plt.show()

                plt.imshow(g_tube)
                plt.show()

                sdjif


                differential_image = np.abs(image_curr_ref_grey- ae_image_grey)

    
         
                plt.figure()
                plt.subplot(1, 5, 1)
                
                plt.imshow(image_curr_ref)
                plt.title("Original Image")
                
                plt.subplot(1, 5, 2)
                plt.imshow(ae_image)
                plt.title("Autoencoder Output")

                plt.subplot(1, 5, 3)
                plt.imshow(differential_image, cmap='gray')
                plt.title("Autoencoder - Original")

                plt.subplot(1, 5, 4)
                plt.imshow(output_seg_np, cmap='gray')
                plt.title("Segmented Image")

                plt.subplot(1, 5, 5)
                plt.imshow(labelled_img, cmap='gray')
                plt.title("Post Processed Image")

                plt.show()

                
                sjkd


                # Check if segmentation outputs are in range (0, 1):
                if torch.max(output_seg) > 1 or torch.min(output_seg) < 0:
                    print("ERROR: Output segmentation out of range!")
                    return
                #time_prep_end = time.time()
                #print(f"Time Inference: {time_prep_end-time_prep_start}")

                # Compute whether an obstacle can be found in groundtruth:
                target_seg_masked = target_seg.clone().type(torch.FloatTensor)
                target_seg_masked[target_seg == 1] = 0
                target_seg_masked[target_seg == 2] = 1
                kernel = torch.tensor(
                    np.ones((args.k_d, args.k_d)) * 1 / (args.k_d * args.k_d)).view(1, 1,
                                                                                                                args.k_d,
                                                                                                                args.k_d).type(
                    torch.FloatTensor)
                patch_density_target = torch.nn.functional.conv2d(target_seg_masked, kernel, padding='same')
                max_patch_density_target = torch.max(patch_density_target)
                if max_patch_density_target > 0.3:
                    has_obstacle = 1
                    # Get bounding box
                    target_seg_masked_pil = presets.torch_mask_to_pil(target_seg_masked)
                    l_bb, u_bb, r_bb, d_bb = target_seg_masked_pil.getbbox()
                else:
                    has_obstacle = 0

                # Binned data for AUROC
                if has_obstacle:
                    for i in range(224):
                        for j in range(224):
                            if evaluation_mask[i, j]:
                                val = int(output_seg[i, j] * 1000)
                                if target_seg[0, 0, i, j] == 2:
                                    stat_book["auroc_data"][val, 1] += 1
                                else:
                                    stat_book["auroc_data"][val, 0] += 1

                for thr_idx, thr in enumerate(thresholds):
                    visualization_images = list()
                    visualization_images.append(VIS_INPUT)
                    if VIS_OUTPUT:
                        visualization_images.append(VIS_OUTPUT)
                    #time_loc_start = time.time()
                    # Compute whether an obstacle can be found in seg output based on patch density:
                    output_seg_masked = output_seg.clone()
                    output_seg_masked[torch.logical_not(evaluation_mask)] = 0
                    kernel = kernel.cuda() if args.device == "cuda" else kernel
                    patch_density = torch.nn.functional.conv2d(output_seg_masked.unsqueeze(0).unsqueeze(1), kernel, padding='same')
                    max_patch_density = torch.max(patch_density)
                    patch_density[patch_density <= thr] = 0
                    patch_density.squeeze()
                    if max_patch_density > thr:
                        found_obstacle = 1
                        # Compute centroid based on patch density
                        x = torch.linspace(0, 223, steps=224).unsqueeze(0)
                        x = x.repeat(224, 1)
                        y = torch.linspace(0, 223, steps=224).unsqueeze(1)
                        y = y.repeat(1, 224)

                        x = x.cuda() if args.device == "cuda" else x
                        y = y.cuda() if args.device == "cuda" else y


                        centroid_x = int(
                            torch.floor(torch.sum(patch_density * x) / torch.sum(patch_density)).type(torch.LongTensor))
                        centroid_y = int(
                            torch.floor(torch.sum(patch_density * y) / torch.sum(patch_density)).type(torch.LongTensor))
                    else:
                        found_obstacle = 0
                    #time_loc_end = time.time()
                    #print(f"Loc time: {time_loc_end - time_loc_start}")

                    # Check whether found obstacle was correct
                    if found_obstacle == 1 and has_obstacle == 1 and l_bb < centroid_x < r_bb and u_bb < centroid_y < d_bb:
                        found_correct = True
                    else:
                        found_correct = False

                    # Now fill statistics book
                    image_data = {"idx": idx, "mode": mode, "has_obstacle": has_obstacle, "found_obstacle": found_obstacle, "found_correct": found_correct}
                    stat_book["image_log"][thr_idx].append(image_data)

                    # Fill confusion matrices:
                    if has_obstacle == 1:
                        if found_obstacle == 1:
                            stat_book["conf_obstacle"][thr_idx]["tp"] += 1
                        else:
                            stat_book["conf_obstacle"][thr_idx]["fn"] += 1
                            stat_book["images_fn_obstacle"][thr_idx].append(idx)
                        if found_correct == 1:
                            stat_book["conf_correct"][thr_idx]["tp"] += 1
                        else:
                            stat_book["conf_correct"][thr_idx]["fn"] += 1
                            stat_book["images_fn_correct"][thr_idx].append(idx)
                    else:  # if there is no obstacle, it does not matter if it was classified correctly or not
                        if found_obstacle == 1:
                            stat_book["conf_obstacle"][thr_idx]["fp"] += 1
                            stat_book["conf_correct"][thr_idx]["fp"] += 1
                            stat_book["images_fp"][thr_idx].append(idx)
                        else:
                            stat_book["conf_obstacle"][thr_idx]["tn"] += 1
                            stat_book["conf_correct"][thr_idx]["tn"] += 1

                    # Visualize Output Patchseg
                    output_seg_vis_gray = output_seg / torch.max(output_seg) * 255

                    # output_seg_vis_gray = output_seg_vis_gray.cpu().numpy().astype(np.uint8)
                    # plt.imshow(output_seg_vis_gray)
                    # plt.show()
                  


                    output_seg_vis_gray = presets.torch_mask_to_pil(output_seg_vis_gray)
                    output_seg_vis = Image.new("RGB", output_seg_vis_gray.size)
                    output_seg_vis.paste(output_seg_vis_gray)
                    visualization_images.append(output_seg_vis)

                    # Visualize Patch Density
                    patch_density_vis_gray = patch_density * 255
                    patch_density_vis_gray = presets.torch_mask_to_pil(patch_density_vis_gray)
                    patch_density_vis = Image.new("RGB", patch_density_vis_gray.size)
                    patch_density_vis.paste(patch_density_vis_gray)
                    draw = ImageDraw.Draw(patch_density_vis)
                    if found_obstacle == 1:
                        if found_correct == 1:
                            draw.ellipse((centroid_x-3 , centroid_y-3, centroid_x+3 , centroid_y+3), fill="green")
                        else:
                            draw.ellipse((centroid_x - 3, centroid_y - 3, centroid_x + 3, centroid_y + 3), fill="red")
                    if has_obstacle == 1:
                        draw.rectangle((l_bb, u_bb, r_bb, d_bb), outline="blue")
                    visualization_images.append(patch_density_vis)

                    # Visualized output seg masked
                    output_seg_masked_vis_gray = output_seg
                    output_seg_masked_vis_gray[torch.logical_not(evaluation_mask)] = 0.5
                    output_seg_masked_vis_gray = output_seg_masked_vis_gray * 255
                    output_seg_masked_vis_gray = presets.torch_mask_to_pil(output_seg_masked_vis_gray)
                    output_seg_masked_vis = Image.new("RGB", output_seg_masked_vis_gray.size)
                    output_seg_masked_vis.paste(output_seg_masked_vis_gray)
                    draw = ImageDraw.Draw(output_seg_masked_vis)
                    if found_obstacle == 1:
                        draw.text((0, 0), f"Max patch %: {max_patch_density:.2f} --> 1", (0, 255, 0))
                    else:
                        draw.text((0, 0), f"Max patch %: {max_patch_density:.2f} --> 0", (255, 0, 0))


                    # plt.imshow(output_seg_masked_vis)
                    # plt.show()
                    visualization_images.append(output_seg_masked_vis)

                    # Visualize target segmentation
                    target_obs_seg_vis_gray = torch.zeros_like(target_seg)
                    target_obs_seg_vis_gray[target_seg == 2] = 255 # obstacle white
                    target_obs_seg_vis_gray[target_seg == 0] = 127 # background gray
                    target_obs_seg_vis_gray = presets.torch_mask_to_pil(target_obs_seg_vis_gray)
                    target_obs_seg_vis = Image.new("RGB", target_obs_seg_vis_gray.size)
                    target_obs_seg_vis.paste(target_obs_seg_vis_gray)
                    draw = ImageDraw.Draw(target_obs_seg_vis)
                    if has_obstacle == 1:
                        draw.text((0, 0), f"Max patch %: {max_patch_density_target:.2f} --> 1", (0, 255, 0))
                    else:
                        draw.text((0, 0), f"Max patch %: {max_patch_density_target:.2f} --> 0", (255, 0, 0))
                    visualization_images.append(target_obs_seg_vis)

                    # Stack horizontally
                    if args.theta_visualize == thr:
                        img_row = np.hstack((np.asarray(i) for i in visualization_images))
                        image_final = Image.fromarray(img_row)
                        image_final.save(os.path.join(vis_path, f"Thr{int(thr*100)}_{idx:04}_{mode}_visualization.jpeg"), format="jpeg")

        # Compute global metrics:
        fpr_list, tpr_list, thr_list, auc = roc_curve(stat_book["auroc_data"])
        roc_auc = auc(fpr_list, tpr_list)
        fig, ax = plt.subplots(1, 1)
        ax.plot(fpr_list, tpr_list, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(vis_path, f"ROC.pdf"))

        output_string = ""

        f1_scores = list()

        # Print confusion matrices:
        for thr_idx, thr in enumerate(thresholds):
            for key in ["conf_obstacle", "conf_correct"]:
                tp = stat_book[key][thr_idx]["tp"]
                fp = stat_book[key][thr_idx]["fp"]
                tn = stat_book[key][thr_idx]["tn"]
                fn = stat_book[key][thr_idx]["fn"]
                tot = tp + fp + tn + fn
                pp = tp + fp
                pn = tn + fn
                p = tp + fn
                n = tn + fp
                if p != 0:
                    tpr = tp / (tp + fn) # True Positive Rate (Sensitivity, Recall): ratio of (correct) detection if there was an obstacle (SAFETY-CRITICAL)
                else:
                    tpr = 99999
                if pn != 0:
                    npv = tn / (fn + tn) # Negative Predictive Value: ratio of correct non-detections if there was no detection (SAFETY-CRITICAL)
                else:
                    npv = 99999
                if n != 0:
                    tnr = tn / (fp + tn) # True Negative Rate (Specificity, 1-FPR): ratio of (correct) non-detection if there was no obstacle
                else:
                    tnr = 99999
                if pp != 0:
                    ppv = tp / (tp + fp) # Positive Predictive Value (Precision): ratio of correct detection if there was a detection
                else:
                    ppv = 99999
                if tpr + ppv != 0:
                    f1 = 2 * (tpr * ppv) / (tpr + ppv) # F1 score
                else:
                    f1 = -1
                if key == "conf_correct":
                    f1_scores.append(f1)

                output_string += f"CONFMAT METRICS FOR {key} (Obstacle Patch Density Threshold: {thr}):\n\n"
                output_string += f"             \t Obstacle     No Obstacle\n"
                output_string += f"Detection:   \t  {tp:4d}          {fp:4d}           |   {pp:4d}\n"
                output_string += f"No Detection:\t  {fn:4d}          {tn:4d}           |   {pn:4d}\n"
                output_string += f"             \t----------------------------------------------\n"
                output_string += f"             \t  {p:4d}          {n:4d}           |   {tot:4d} images\n"
                output_string += f"\n"
                output_string += f"True Positive Rate (Sensitivity, Recall):\t {tpr:.3f} \t\t ratio of (correct) detection if there was an obstacle (SAFETY-CRITICAL)\n"
                output_string += f"Negative Predictive Value:               \t {npv:.3f} \t\t ratio of correct non-detections if there was no detection (SAFETY-CRITICAL)\n"
                output_string += f"True Negative Rate (Specificity, 1-FPR): \t {tnr:.3f} \t\t ratio of (correct) non-detection if there was no obstacle\n"
                output_string += f"Positive Predictive Value (Precision):   \t {ppv:.3f} \t\t ratio of correct detection if there was a detection\n"
                output_string += f"F1 Score:                                \t {f1:.3f} \t\t trade-off between precision and recall\n"
                output_string += f"\n\n"
                output_string += f"Image Ids for False Positive: {stat_book['images_fp'][thr_idx]}\n"
                output_string += f"Image Ids for False Negative Obstacle: {stat_book['images_fn_obstacle'][thr_idx]}\n"
                output_string += f"Image Ids for False Negative Correct: {stat_book['images_fn_correct'][thr_idx]}\n"
                output_string += f"\n\n"

        max_f1 = max(f1_scores)
        max_f1_idx = f1_scores.index(max_f1)
        max_f1_thr = thresholds[max_f1_idx]
        output_string += f"\n\nFINAL RESULTS:\n\n"
        output_string += f"AUROC: {roc_auc:.3f}\n"
        output_string += f"Max F1: {max_f1} with threshold {max_f1_thr}\n\n"
        output_string += f"\tfrom {f1_scores}\n"
        output_string += f"Image Ids for False Positive: {stat_book['images_fp'][max_f1_idx]}\n"
        output_string += f"Image Ids for False Negative Correct: {stat_book['images_fn_correct'][max_f1_idx]}\n"
        output_string += f"\n\n"

        print(output_string)
        with open(os.path.join(vis_path, "output.txt"), "w") as file:
            file.write(output_string)
        with open(os.path.join(vis_path, "args.txt"), 'w') as file:
            file.write(json.dumps(vars(args)))
    return

def roc_curve(roc_data):
    max_thres = roc_data.shape[0] # 1001 in our case (threshold is from >= 0 (all) to >= 1001 (none)
    fpr_list = np.empty((max_thres+1,))
    tpr_list = np.empty((max_thres+1,))
    thr_list = np.empty((max_thres+1,))
    # >= 1001:
    fpr_list[max_thres] = 0
    tpr_list[max_thres] = 0
    thr_list[max_thres] = max_thres
    # i in [1000, 1]
    for i in range(max_thres - 1, 0, -1):
        thr_list[i] = i
        tp = np.sum(roc_data[i:, 1])
        fp = np.sum(roc_data[i:, 0])
        tn = np.sum(roc_data[:i, 0])
        fn = np.sum(roc_data[:i, 1])
        tpr = float(tp / (tp + fn))
        fpr = float(fp / (fp + tn))
        tpr_list[i] = tpr
        fpr_list[i] = fpr
    # >= 0:
    fpr_list[0] = 1
    tpr_list[0] = 1
    thr_list[0] = 0

    return fpr_list, tpr_list, thr_list, auc


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

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
