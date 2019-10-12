import csv
import os
import cv2
import numpy as np
import random
import augmentor




####CAUTION the data is from pytorch tutorial ,
###download from url=https://download.pytorch.org/tutorial/faces.zip
##### and i find some of them are not labeled very well

csv_file='faces/face_landmarks.csv'

###parse the scv
label_file=csv.reader(open(csv_file,'r'))


for _,single_sample in enumerate(label_file):
    if _==0:
        ##drop the header in csvfile
        continue

    image_path=os.path.join('faces',single_sample[0])
    label=np.array(single_sample[1:]).reshape([-1,2]).astype(np.int)
    img=cv2.imread(image_path)
    for _index in range(label.shape[0]):
        x_y=label[_index]
        cv2.circle(img,center=(x_y[0],x_y[1]),color=(122,122,122),radius=2,thickness=2)

    cv2.imshow('raw',img)

    ##first make it rotate with label
    img = cv2.imread(image_path)
    label = np.array(single_sample[1:]).reshape([-1, 2]).astype(np.int)
    angle=random.uniform(-180,180)
    img,aug_label=augmentor.Rotate_aug(img,label=label,angle=angle)
    for _index in range(aug_label.shape[0]):
        x_y=aug_label[_index]
        cv2.circle(img,center=(x_y[0],x_y[1]),color=(122,122,122),radius=2,thickness=2)
    cv2.imshow('rotate with label',img)

    ##first make it rotate without label
    img = cv2.imread(image_path)
    label = np.array(single_sample[1:]).reshape([-1, 2]).astype(np.int)
    angle = random.uniform(-180, 180)
    img, _ = augmentor.Rotate_aug(img, angle=angle)
    cv2.imshow('rotate without label', img)

    ##first make it Affine_aug with label
    img = cv2.imread(image_path)
    label = np.array(single_sample[1:]).reshape([-1, 2]).astype(np.int)
    strength=random.uniform(0,100)
    img, aug_label = augmentor.Affine_aug(img,strength=strength,label=label)
    for _index in range(aug_label.shape[0]):
        x_y = aug_label[_index]
        cv2.circle(img, center=(x_y[0], x_y[1]), color=(122, 122, 122), radius=2, thickness=2)
    cv2.imshow('Affine transform with label', img)



    ###padding  with a target shape
    img = cv2.imread(image_path)
    label = np.array(single_sample[1:]).reshape([-1, 2]).astype(np.int)
    img,aug_label = augmentor.Fill_img(img,target_height=480,target_width=640,label=label)
    for _index in range(aug_label.shape[0]):
        x_y = aug_label[_index]
        cv2.circle(img, center=(x_y[0], x_y[1]), color=(122, 122, 122), radius=2, thickness=2)
    cv2.imshow('padding transform with label', img)

    ##blur
    img = cv2.imread(image_path)
    label = np.array(single_sample[1:]).reshape([-1, 2]).astype(np.int)
    strength = random.uniform(0, 60)
    img = augmentor.Blur_aug(img, ksize=(7,7))
    for _index in range(label.shape[0]):
        x_y = label[_index]
        cv2.circle(img, center=(x_y[0], x_y[1]), color=(122, 122, 122), radius=2, thickness=2)
    cv2.imshow('blur transform with label', img)

    ##img dropout
    img = cv2.imread(image_path)
    label = np.array(single_sample[1:]).reshape([-1, 2]).astype(np.int)
    strength = random.uniform(0, 60)
    img = augmentor.Img_dropout(img, max_pattern_ratio=0.4)
    for _index in range(label.shape[0]):
        x_y = label[_index]
        cv2.circle(img, center=(x_y[0], x_y[1]), color=(122, 122, 122), radius=2, thickness=2)
    cv2.imshow('img_dropout transform with label', img)

    ##mirror

    img = cv2.imread(image_path)
    label = np.array(single_sample[1:]).reshape([-1, 2]).astype(np.int)
    strength = random.uniform(0, 60)
    ####need symmetry to swap from left and right, the symmetry need change for u data
    symmetry=[(0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),(8,8),
              (17,26),(18,25),(19,24),(20,23),(21,22),
              (31,35),(32,34),
              (36,45),(37,44),(38,43),(39,42),(40,47),(41,46),
              (48,54),(49,53),(50,52),(55,59),(56,58),(60,64),(61,63),(65,67)]
    img,aug_label = augmentor.Mirror(img, label=label,symmetry=symmetry)
    for _index in range(aug_label.shape[0]):
        x_y = aug_label[_index]
        cv2.circle(img, center=(x_y[0], x_y[1]), color=(122, 122, 122), radius=2, thickness=2)
    cv2.imshow('flip transform with label', img)

    ###heatmaps

    label = np.array(single_sample[1:]).reshape([-1, 2]).astype(np.int).T
    heat_map_size=img.shape[0:2]
    heat=augmentor.produce_heat_maps(label,heat_map_size,1,1)
    augmentor.visualize_heatmap_target(heat)##visualise



    cv2.waitKey(0)



