# Dataset Generation (RailSem19Cropped, FishyrailsCropped)

This directory contains the code to create the *RailSem19Cropped* and *FishyrailsCropped* datasets.

# Set-up


## Virtual Environment (using Python 3.8)
```
chmod +x env.sh && ./env.sh
```

## Datasets 

### RailSem19
```
cd dataset_generation
python get_rs19_val.py
```

### PascalVOC

```
cd dataset_generation
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

tar -xvf VOCtrainval_11-May-2012.tar
rm -rf mv VOC2012/VOC2012 .
```

### ImageNet
```
kaggle competitions download -c imagenet-object-localization-challenge
```

## Dataset Creation

### Create Railsem19Croppedv1
```
# Generate region of interest crops
python generate_image_crops.py --max_images 999999 --mode rs19 --input_path rs19_val --output_path Railsem19Croppedv1

# Convert dataset to hdf5
python railsem19cropped2hdf5.py --input_path Railsem19Croppedv1 --output_name Railsem19Croppedv1
```

### Create FishyrailsCroppedv1
```
# Augment RailSem19 with obstacles
python fishyrails.py --max_images 1000 --max_obstacles 2000 --output_path Fishyrailsv1 --input_path_rs19 rs19_val --input_path_voc VOC2012

# Generate region of interest crops
python generate_image_crops.py --max_images 999999 --mode fishyrails --input_path Fishyrailsv1 --output_path FishyrailsCroppedv1

# Convert dataset to hdf5
python fishyrailscropped2hdf5.py --input_path FishyrailsCroppedv1 --output_name FishyrailsCroppedv1
```

### ImageNet
```
# Convert dataset to hdf5
python imagenet2hdf5.py --input_path /path/to/datasets/ImageNet --output_name ImageNet
```