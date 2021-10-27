# -*- coding:utf-8 -*-
from modified_deeplab_V3 import *
from PFB_measurement import Measurement
from random import shuffle, random

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 512,

                           "train_txt_path": "/yuhwan/yuhwan/Dataset/Segmentation/Crop_weed/datasets_IJRR2017/train.txt",

                           "val_txt_path": "/yuhwan/yuhwan/Dataset/Segmentation/Crop_weed/datasets_IJRR2017/val.txt",

                           "test_txt_path": "/yuhwan/yuhwan/Dataset/Segmentation/Crop_weed/datasets_IJRR2017/test.txt",
                           
                           "label_path": "/yuhwan/yuhwan/Dataset/Segmentation/Crop_weed/datasets_IJRR2017/raw_aug_gray_mask/",
                           
                           "image_path": "/yuhwan/yuhwan/Dataset/Segmentation/Crop_weed/datasets_IJRR2017/raw_aug_rgb_img/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "C:/Users/Yuhwan/Downloads/156/156",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 200,

                           "total_classes": 3,

                           "ignore_label": 0,

                           "batch_size": 4,

                           "sample_images": "/yuhwan/yuhwan/checkpoint/Segmenation/V3_5th_paper/BoniRob/sample_images",

                           "save_checkpoint": "/yuhwan/yuhwan/checkpoint/Segmenation/V3_5th_paper/BoniRob/checkpoint",

                           "save_print": "/yuhwan/yuhwan/checkpoint/Segmenation/V3_5th_paper/BoniRob/train_out.txt",

                           "test_images": "D:/[1]DB/[5]4th_paper_DB/crop_weed/V2/test_images",

                           "train": True})


optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
color_map = np.array([[255, 0, 0], [0, 0, 255], [0,0,0]], dtype=np.uint8)

def tr_func(image_list, label_list):

    h = tf.random.uniform([1], 1e-2, 30)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 30)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.random_brightness(img, max_delta=50.)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    no_img = img
    img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)

    return img, no_img, lab

def test_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0, 255)
    img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

def test_func2(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.clip_by_value(img, 0, 255)
    temp_img = img / 255
    img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68]) # 평균값 보정

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_png(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, temp_img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

def cal_loss(model, images, labels, crop_objectiness, weed_objectiness, total_objectiness, class_im_plain, crop_im_plain, weed_im_plain, ignore_label):

    with tf.GradientTape() as tape:
        # loss가 불안정함! 수정하자!!!기억해!!!!!!!!!!!!!!!!!!!!!!!!!
        
        batch_labels = tf.reshape(labels, [-1,])
        indices = tf.squeeze(tf.where(tf.not_equal(batch_labels, ignore_label)),1)
        batch_labels = tf.cast(tf.gather(batch_labels, indices), tf.float32)

        logits = run_model(model, images, True)
        raw_logits = tf.reshape(logits, [-1, 6])
        predict = tf.gather(raw_logits, indices)

        class_im_plain = tf.reshape(class_im_plain, [-1,])
        class_im_plain = tf.cast(tf.gather(class_im_plain, indices), tf.float32)
        crop_im_plain = tf.cast(tf.reshape(crop_im_plain, [-1,]), tf.float32)
        weed_im_plain = tf.cast(tf.reshape(weed_im_plain, [-1,]), tf.float32)

        ##########################################################################################################
        crop_obj_indices = tf.squeeze(tf.where(tf.equal(tf.reshape(crop_objectiness, [-1,]), 1)), 1)
        crop_logit_objectiness = tf.gather(raw_logits[:, 3], crop_obj_indices)
        crop_label_objectiness = tf.cast(tf.gather(tf.reshape(crop_objectiness, [-1,]), crop_obj_indices), tf.float32)
        crop_obj_loss = -crop_label_objectiness * tf.math.log(tf.nn.sigmoid(crop_logit_objectiness) + 1e-7)
        crop_obj_loss = tf.reduce_mean(crop_obj_loss)

        no_crop_obj_indices = tf.squeeze(tf.where(tf.not_equal(tf.reshape(crop_objectiness, [-1,]), 1)),1)
        no_crop_logit_objectiness = tf.gather(raw_logits[:, 3], no_crop_obj_indices)
        no_crop_label_objectiness = tf.cast(tf.gather(tf.reshape(crop_objectiness, [-1,]), no_crop_obj_indices), tf.float32)
        no_crop_obj_loss = -(1. - no_crop_label_objectiness) * tf.math.log(1 - tf.nn.sigmoid(no_crop_logit_objectiness) + 1e-7)
        no_crop_obj_loss = tf.reduce_mean(no_crop_obj_loss)

        crop_logit = raw_logits[:, 2]
        cast_crop_objectiness = tf.cast(tf.reshape(crop_objectiness, [-1,]), tf.float32)
        crop_loss = dice_loss(cast_crop_objectiness, crop_logit) \
            + tf.nn.sigmoid_cross_entropy_with_logits(cast_crop_objectiness, crop_logit) * crop_im_plain
        crop_loss = tf.reduce_mean(crop_loss) + crop_obj_loss + no_crop_obj_loss

        weed_obj_indices = tf.squeeze(tf.where(tf.equal(tf.reshape(weed_objectiness, [-1,]), 1)), 1)
        weed_logit_objectiness = tf.gather(raw_logits[:, 5], weed_obj_indices)
        weed_label_objectiness = tf.cast(tf.gather(tf.reshape(weed_objectiness, [-1,]), weed_obj_indices), tf.float32)
        weed_obj_loss = -weed_label_objectiness * tf.math.log(tf.nn.sigmoid(weed_logit_objectiness) + 1e-7)
        weed_obj_loss = tf.reduce_mean(weed_obj_loss)

        no_weed_obj_indices = tf.squeeze(tf.where(tf.not_equal(tf.reshape(weed_objectiness, [-1,]), 1)), 1)
        no_weed_logit_objectiness = tf.gather(raw_logits[:, 5], no_weed_obj_indices)
        no_weed_label_objectiness = tf.cast(tf.gather(tf.reshape(weed_objectiness, [-1,]), no_weed_obj_indices), tf.float32)
        no_weed_obj_loss = -(1. - no_weed_label_objectiness) * tf.math.log(1 - tf.nn.sigmoid(no_weed_logit_objectiness) + 1e-7)
        no_weed_obj_loss = tf.reduce_mean(no_weed_obj_loss)

        weed_logit = raw_logits[:, 4]
        cast_weed_objectiness = tf.cast(tf.reshape(weed_objectiness, [-1,]), tf.float32)
        weed_loss = dice_loss(cast_weed_objectiness, weed_logit) \
            + tf.nn.sigmoid_cross_entropy_with_logits(cast_weed_objectiness, weed_logit) * weed_im_plain
        weed_loss = tf.reduce_mean(weed_loss) + weed_obj_loss + no_weed_obj_loss
        ##########################################################################################################
        
        label_objectiness = tf.cast(tf.reshape(total_objectiness, [-1,]), tf.float32)
        logit_objectiness = raw_logits[:, 1]

        no_obj_indices = tf.squeeze(tf.where(tf.equal(tf.reshape(total_objectiness, [-1,]), 0)),1)
        no_logit_objectiness = tf.gather(logit_objectiness, no_obj_indices)
        no_obj_labels = tf.cast(tf.gather(label_objectiness, no_obj_indices), tf.float32)
        no_obj_loss = -(1. - no_obj_labels) * tf.math.log(1 - tf.nn.sigmoid(no_logit_objectiness) + 1e-7)
        no_obj_loss = tf.reduce_mean(no_obj_loss)

        obj_indices = tf.squeeze(tf.where(tf.not_equal(tf.reshape(total_objectiness, [-1,]), 0)),1)
        yes_logit_objectiness = tf.gather(logit_objectiness, obj_indices)
        yes_obj_labels = tf.cast(tf.gather(label_objectiness, obj_indices), tf.float32)
        obj_loss = -yes_obj_labels * tf.math.log(tf.nn.sigmoid(yes_logit_objectiness) + 1e-7)
        obj_loss = tf.reduce_mean(obj_loss)

        seg_loss = dice_loss(batch_labels, predict[:, 0]) \
            + tf.nn.sigmoid_cross_entropy_with_logits(batch_labels, predict[:, 0]) * class_im_plain
        seg_loss = tf.reduce_mean(seg_loss) + obj_loss + no_obj_loss
        
        loss = seg_loss + weed_loss + crop_loss

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# yilog(h(xi;θ))+(1−yi)log(1−h(xi;θ))
def main():
    tf.keras.backend.clear_session()

    model = DeepLabV3Plus(FLAGS.img_size, FLAGS.img_size, 34)
    out = model.get_layer("activation_decoder_2_upsample").output
    out = tf.keras.layers.Conv2D(6, (1,1), name="output_layer")(out)
    model = tf.keras.Model(inputs=model.input, outputs=out)
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        #elif isinstance(layer, tf.keras.layers.Conv2D):
        #    layer.kernel_regularizer = tf.keras.regularizers.l2(0.0005)

    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print("Restored!!")
    
    if FLAGS.train:
        count = 0
        output_text = open(FLAGS.save_print, "w")
        
        train_list = np.loadtxt(FLAGS.train_txt_path, dtype="<U200", skiprows=0, usecols=0)
        val_list = np.loadtxt(FLAGS.val_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        train_img_dataset = [FLAGS.image_path + data for data in train_list]
        val_img_dataset = [FLAGS.image_path + data for data in val_list]
        test_img_dataset = [FLAGS.image_path + data for data in test_list]

        train_lab_dataset = [FLAGS.label_path + data for data in train_list]
        val_lab_dataset = [FLAGS.label_path + data for data in val_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        val_ge = tf.data.Dataset.from_tensor_slices((val_img_dataset, val_lab_dataset))
        val_ge = val_ge.map(test_func)
        val_ge = val_ge.batch(1)
        val_ge = val_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img_dataset, train_lab_dataset))
            shuffle(A)
            train_img_dataset, train_lab_dataset = zip(*A)
            train_img_dataset, train_lab_dataset = np.array(train_img_dataset), np.array(train_lab_dataset)

            train_ge = tf.data.Dataset.from_tensor_slices((train_img_dataset, train_lab_dataset))
            train_ge = train_ge.shuffle(len(train_img_dataset))
            train_ge = train_ge.map(tr_func)
            train_ge = train_ge.batch(FLAGS.batch_size)
            train_ge = train_ge.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(train_ge)
            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, print_images, batch_labels = next(tr_iter)  
                batch_labels = batch_labels.numpy()
                batch_labels = np.where(batch_labels == FLAGS.ignore_label, 2, batch_labels)    # 2 is void
                batch_labels = np.where(batch_labels == 255, 0, batch_labels)
                batch_labels = np.where(batch_labels == 128, 1, batch_labels)
                crop_labels = batch_labels
                weed_labels = batch_labels

                crop_objectiness = np.where(crop_labels == 2, 0, crop_labels)  # crop 피사체가 있는곳은 1 없는곳은 0으로 만들어준것
                crop_objectiness = np.where(crop_labels == 1, 0, crop_objectiness) # crop 피사체가 있는곳은 1 없는곳은 0으로 만들어준것
                crop_objectiness = np.where(crop_labels == 0, 1, crop_objectiness) # crop 피사체가 있는곳은 1 없는곳은 0으로 만들어준것
                #plt.imshow(crop_objectiness[0, :, :, 0]*255, cmap="gray")
                #plt.show()

                weed_objectiness = np.where(weed_labels == 2, 0, weed_labels) # weed 피사체가 있는곳은 1 없는곳은 0으로 만들어준것
                weed_objectiness = np.where(weed_labels == 0, 0, weed_objectiness) # weed 피사체가 있는곳은 1 없는곳은 0으로 만들어준것
                weed_objectiness = np.where(weed_labels == 1, 1, weed_objectiness) # weed 피사체가 있는곳은 1 없는곳은 0으로 만들어준것
                #plt.imshow(weed_objectiness[0, :, :, 0]*255, cmap="gray")
                #plt.show()

                total_objectiness = np.where(batch_labels == 2, 0, 1)  # 피사체가 있는곳은 1 없는곳은 0으로 만들어준것
                #plt.imshow(total_objectiness[0, :, :, 0]*255, cmap="gray")
                #plt.show()

                class_imbal_labels = np.squeeze(batch_labels, -1)
                crop_imbal_labels = np.squeeze(crop_objectiness, -1)
                weed_imbal_labels = np.squeeze(weed_objectiness, -1)
                class_imbal_labels_buf = 0.
                crop_imbal_labels_buf = 0.
                weed_imbal_labels_buf = 0.
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels[i]
                    crop_imlab_label = crop_imbal_labels[i]
                    weed_imbal_label = weed_imbal_labels[i]

                    class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])
                    crop_imlab_label = np.reshape(crop_imlab_label, [FLAGS.img_size*FLAGS.img_size, ])
                    weed_imbal_label = np.reshape(weed_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])

                    count_c_i_lab = np.bincount(class_imbal_label, minlength=FLAGS.total_classes)
                    count_crop_i_lab = np.bincount(crop_imlab_label, minlength=FLAGS.total_classes-1)
                    count_weed_i_lab = np.bincount(weed_imbal_label, minlength=FLAGS.total_classes-1)

                    class_imbal_labels_buf += count_c_i_lab
                    crop_imbal_labels_buf += count_crop_i_lab
                    weed_imbal_labels_buf += count_weed_i_lab

                class_imbal_labels_buf /= FLAGS.batch_size
                class_imbal_labels_buf = class_imbal_labels_buf[0:FLAGS.total_classes-1]
                class_imbal_labels_buf = (np.max(class_imbal_labels_buf / np.sum(class_imbal_labels_buf)) + 1 - (class_imbal_labels_buf / np.sum(class_imbal_labels_buf)))
                class_im_plain = np.where(np.squeeze(batch_labels, -1) == 0, class_imbal_labels_buf[0], np.squeeze(batch_labels, -1))
                class_im_plain = np.where(np.squeeze(batch_labels, -1) == 1, class_imbal_labels_buf[1], class_im_plain)

                crop_imbal_labels_buf /= FLAGS.batch_size
                crop_imbal_labels_buf = (np.max(crop_imbal_labels_buf / np.sum(crop_imbal_labels_buf)) + 1 - (crop_imbal_labels_buf / np.sum(crop_imbal_labels_buf)))
                crop_im_plain = np.where(np.squeeze(crop_objectiness, -1) == 0, crop_imbal_labels_buf[0], np.squeeze(crop_objectiness, -1))
                crop_im_plain = np.where(np.squeeze(crop_objectiness, -1) == 1, crop_imbal_labels_buf[1], crop_im_plain)

                weed_imbal_labels_buf /= FLAGS.batch_size
                weed_imbal_labels_buf = (np.max(weed_imbal_labels_buf / np.sum(weed_imbal_labels_buf)) + 1 - (weed_imbal_labels_buf / np.sum(weed_imbal_labels_buf)))
                weed_im_plain = np.where(np.squeeze(weed_objectiness, -1) == 0, weed_imbal_labels_buf[0], np.squeeze(weed_objectiness, -1))
                weed_im_plain = np.where(np.squeeze(weed_objectiness, -1) == 1, weed_imbal_labels_buf[1], weed_im_plain)

                loss = cal_loss(model, batch_images, batch_labels, crop_objectiness, weed_objectiness, total_objectiness, class_im_plain, crop_im_plain, weed_im_plain, 2)
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step+1, tr_idx, loss))

                if count % 100 == 0:

                    logits = run_model(model, batch_images, False)
                    crop_weed_images = tf.nn.sigmoid(logits[:, :, :, 0])
                    crop_weed_objects = tf.nn.sigmoid(logits[:, :, :, 1])
                    crop_images = tf.nn.sigmoid(logits[:, :, :, 2])
                    crop_objects = tf.nn.sigmoid(logits[:, :, :, 3])
                    weed_images = tf.nn.sigmoid(logits[:, :, :, 4])
                    weed_objects = tf.nn.sigmoid(logits[:, :, :, 5])
                    for i in range(FLAGS.batch_size):
                        crop_weed_image = crop_weed_images[i]
                        crop_weed_image = np.where(crop_weed_image.numpy() >= 0.5, 1, 0)    # 1:weed, 0: crop
                        predict_temp = crop_weed_image
                        crop_weed_object = crop_weed_objects[i]
                        crop_weed_object = np.where(crop_weed_object.numpy() >= 0.5, 1, 2)  # 1:crop/weed object, 2: background
                        crop_image = crop_images[i]
                        crop_image = np.where(crop_image.numpy() >= 0.5, 1, 0)  # 1:crop, 0:background
                        predict_temp2 = crop_image
                        crop_object = crop_objects[i]
                        crop_object = np.where(crop_object.numpy() >= 0.5, 1, 2)    # 1:crop object, 0:background
                        weed_image = weed_images[i]
                        weed_image = np.where(weed_image.numpy() >= 0.5, 1, 0)  # 1:weed, 0: background
                        predict_temp3 = weed_image
                        weed_object = weed_objects[i]
                        weed_object = np.where(weed_object.numpy() >= 0.5, 1, 2)    # 1: weed object, 2: background

                        cw_object_predict_axis = np.where(crop_weed_object==2)
                        predict_temp[cw_object_predict_axis] = 2    # 이 모델은 전 버전에서 잘 나온것을 확인했음
                        predict_temp = np.expand_dims(predict_temp, -1)

                        c_object_predict_axis = np.where(crop_object==2)
                        c_object_predict_axis2 = np.where(crop_object!=2)
                        predict_temp2[c_object_predict_axis] = 2
                        predict_temp2[c_object_predict_axis2] = 0
                        predict_temp2 = np.expand_dims(predict_temp2, -1)

                        w_object_predict_axis = np.where(weed_object==2)
                        w_object_predict_axis2 = np.where(weed_object!=2)
                        predict_temp3[w_object_predict_axis] = 2
                        predict_temp3[w_object_predict_axis2] = 1
                        predict_temp3 = np.expand_dims(predict_temp3, -1)

                        predict_temp4 = np.concatenate([predict_temp, predict_temp2, predict_temp3], -1)


                        pred_mask_color = np.where(predict_temp4 == np.array([0,0,0], dtype=np.uint8),
                                                   np.array([255,0,0], dtype=np.uint8),
                                                   predict_temp4)
                        pred_mask_color = np.where(predict_temp4 == np.array([1,1,1], dtype=np.uint8),
                                                   np.array([0,0,255], dtype=np.uint8),
                                                   pred_mask_color)
                        pred_mask_color = np.where(predict_temp4 == np.array([2,2,2], dtype=np.uint8),
                                                   np.array([0,0,0], dtype=np.uint8),
                                                   pred_mask_color)
                        func1 = lambda f:f != np.array([0,0,0], dtype=np.uint8)
                        func2 = lambda f:f != np.array([1,1,1], dtype=np.uint8)
                        pred_mask_color = np.where(func1(predict_temp4)&func2(predict_temp4),
                                                   np.array([0,0,0], dtype=np.uint8),
                                                   pred_mask_color)

                        func3 = lambda r: r[:, :, 0] == 255 
                        func4 = lambda g: g[:, :, 1] == 0 
                        func5 = lambda b: b[:, :, 2] == 0

                        func6 = lambda r: r[:, :, 0] == 0 
                        func7 = lambda g: g[:, :, 1] == 0 
                        func8 = lambda b: b[:, :, 2] == 255

                        metric_pred = np.zeros([FLAGS.img_size, FLAGS.img_size], dtype=np.uint8) + 2
                        metric_pred = np.where(func3(pred_mask_color) & func4(pred_mask_color) & func5(pred_mask_color), 0, 2)
                        metric_pred = np.where(func6(pred_mask_color) & func7(pred_mask_color) & func8(pred_mask_color), 1, metric_pred)
                        pred_mask_color = color_map[metric_pred]

                        pred_mask_color2 = color_map[predict_temp]
                        pred_mask_color2 = np.squeeze(pred_mask_color2, 2)

                        label = batch_labels[i]
                        label = np.concatenate((label, label, label), -1)
                        label_mask_color = np.zeros([FLAGS.img_size, FLAGS.img_size, 3], dtype=np.uint8)
                        label_mask_color = np.where(label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
                        label_mask_color = np.where(label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_label.png", label_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_predict.png", pred_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_predict2.png", pred_mask_color2)
                    

                count += 1

            tr_iter = iter(train_ge)
            miou = 0.
            f1_score = 0.
            tdr = 0.
            sensitivity = 0.
            crop_iou = 0.
            weed_iou = 0.
            for i in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                batch_labels = tf.squeeze(batch_labels, -1)
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    logits = run_model(model, batch_image, False) # type을 batch label과 같은 type으로 맞춰주어야함
                    crop_weed_images = tf.nn.sigmoid(logits[:, :, :, 0])
                    crop_weed_objects = tf.nn.sigmoid(logits[:, :, :, 1])
                    crop_images = tf.nn.sigmoid(logits[:, :, :, 2])
                    crop_objects = tf.nn.sigmoid(logits[:, :, :, 3])
                    weed_images = tf.nn.sigmoid(logits[:, :, :, 4])
                    weed_objects = tf.nn.sigmoid(logits[:, :, :, 5])


                    crop_weed_image = crop_weed_images[0]
                    crop_weed_image = np.where(crop_weed_image.numpy() >= 0.5, 1, 0)    # 1:weed, 0: crop
                    predict_temp = crop_weed_image
                    crop_weed_object = crop_weed_objects[0]
                    crop_weed_object = np.where(crop_weed_object.numpy() >= 0.5, 1, 2)  # 1:crop/weed object, 2: background
                    crop_image = crop_images[0]
                    crop_image = np.where(crop_image.numpy() >= 0.5, 1, 0)  # 1:crop, 0:background
                    predict_temp2 = crop_image
                    crop_object = crop_objects[0]
                    crop_object = np.where(crop_object.numpy() >= 0.5, 1, 2)    # 1:crop object, 2:background
                    weed_image = weed_images[0]
                    weed_image = np.where(weed_image.numpy() >= 0.5, 1, 0)  # 1:weed, 0: background
                    predict_temp3 = weed_image
                    weed_object = weed_objects[0]
                    weed_object = np.where(weed_object.numpy() >= 0.5, 1, 2)    # 1: weed object, 2: background

                    cw_object_predict_axis = np.where(crop_weed_object==2)
                    predict_temp[cw_object_predict_axis] = 2    # 이 모델은 전 버전에서 잘 나온것을 확인했음
                    predict_temp = np.expand_dims(predict_temp, -1)

                    c_object_predict_axis = np.where(crop_object==2)
                    c_object_predict_axis2 = np.where(crop_object!=2)
                    predict_temp2[c_object_predict_axis] = 2
                    predict_temp2[c_object_predict_axis2] = 0
                    predict_temp2 = np.expand_dims(predict_temp2, -1)

                    w_object_predict_axis = np.where(weed_object==2)
                    w_object_predict_axis2 = np.where(weed_object!=2)
                    predict_temp3[w_object_predict_axis] = 2
                    predict_temp3[w_object_predict_axis2] = 1
                    predict_temp3 = np.expand_dims(predict_temp3, -1)

                    predict_temp4 = np.concatenate([predict_temp, predict_temp2, predict_temp3], -1)


                    pred_mask_color = np.where(predict_temp4 == np.array([0,0,0], dtype=np.uint8),
                                                np.array([255,0,0], dtype=np.uint8),
                                                predict_temp4)
                    pred_mask_color = np.where(predict_temp4 == np.array([1,1,1], dtype=np.uint8),
                                                np.array([0,0,255], dtype=np.uint8),
                                                pred_mask_color)
                    pred_mask_color = np.where(predict_temp4 == np.array([2,2,2], dtype=np.uint8),
                                                np.array([0,0,0], dtype=np.uint8),
                                                pred_mask_color)
                    func1 = lambda f:f != np.array([0,0,0], dtype=np.uint8)
                    func2 = lambda f:f != np.array([1,1,1], dtype=np.uint8)
                    pred_mask_color = np.where(func1(predict_temp4)&func2(predict_temp4),
                                                np.array([0,0,0], dtype=np.uint8),
                                                pred_mask_color)
                    func3 = lambda r: r[:, :, 0] == 255 
                    func4 = lambda g: g[:, :, 1] == 0 
                    func5 = lambda b: b[:, :, 2] == 0

                    func6 = lambda r: r[:, :, 0] == 0 
                    func7 = lambda g: g[:, :, 1] == 0 
                    func8 = lambda b: b[:, :, 2] == 255

                    metric_pred = np.zeros([FLAGS.img_size, FLAGS.img_size], dtype=np.uint8) + 2
                    metric_pred = np.where(func3(pred_mask_color) & func4(pred_mask_color) & func5(pred_mask_color), 0, 2)
                    metric_pred = np.where(func6(pred_mask_color) & func7(pred_mask_color) & func8(pred_mask_color), 1, metric_pred)

                    label = batch_labels[j]
                    label = tf.cast(label, tf.uint8).numpy()
                    label = np.where(label == FLAGS.ignore_label, 2, label)    # 2 is void
                    label = np.where(label == 255, 0, label)
                    label = np.where(label == 128, 1, label)
                    batch_label = label
                    label = np.concatenate((label, label, label), -1)
                    label_mask_color = np.zeros([FLAGS.img_size, FLAGS.img_size, 3], dtype=np.uint8)
                    label_mask_color = np.where(label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
                    label_mask_color = np.where(label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=metric_pred,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=FLAGS.total_classes).MIOU()
                    f1_score_, recall_ = Measurement(predict=metric_pred,
                                            label=batch_label,
                                            shape=[FLAGS.img_size*FLAGS.img_size, ],
                                            total_classes=FLAGS.total_classes).F1_score_and_recall()
                    tdr_ = Measurement(predict=metric_pred,
                                            label=batch_label,
                                            shape=[FLAGS.img_size*FLAGS.img_size, ],
                                            total_classes=FLAGS.total_classes).TDR()

                    miou += miou_
                    f1_score += f1_score_
                    sensitivity += recall_
                    tdr += tdr_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_
            print("=================================================================================================================================================")
            print("Epoch: %3d, train mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), train F1_score = %.4f, train sensitivity = %.4f, train TDR = %.4f" % (epoch, miou / len(train_img_dataset),
                                                                                                                                                 crop_iou / len(train_img_dataset),
                                                                                                                                                 weed_iou / len(train_img_dataset),
                                                                                                                                                  f1_score / len(train_img_dataset),
                                                                                                                                                  sensitivity / len(train_img_dataset),
                                                                                                                                                  tdr / len(train_img_dataset)))
            output_text.write("Epoch: ")
            output_text.write(str(epoch))
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.write("train mIoU: ")
            output_text.write("%.4f" % (miou / len(train_img_dataset)))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou / len(train_img_dataset)))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou / len(train_img_dataset)))
            output_text.write(", train F1_score: ")
            output_text.write("%.4f" % (f1_score / len(train_img_dataset)))
            output_text.write(", train sensitivity: ")
            output_text.write("%.4f" % (sensitivity / len(train_img_dataset)))
            output_text.write(", train TDR: ")
            output_text.write("%.4f" % (tdr / len(train_img_dataset)))
            output_text.write("\n")

            val_iter = iter(val_ge)
            miou = 0.
            f1_score = 0.
            tdr = 0.
            sensitivity = 0.
            crop_iou = 0.
            weed_iou = 0.
            for i in range(len(val_img_dataset)):
                batch_images, batch_labels = next(val_iter)
                batch_labels = tf.squeeze(batch_labels, -1)
                for j in range(1):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    logits = run_model(model, batch_image, False) # type을 batch label과 같은 type으로 맞춰주어야함
                    crop_weed_images = tf.nn.sigmoid(logits[:, :, :, 0])
                    crop_weed_objects = tf.nn.sigmoid(logits[:, :, :, 1])
                    crop_images = tf.nn.sigmoid(logits[:, :, :, 2])
                    crop_objects = tf.nn.sigmoid(logits[:, :, :, 3])
                    weed_images = tf.nn.sigmoid(logits[:, :, :, 4])
                    weed_objects = tf.nn.sigmoid(logits[:, :, :, 5])


                    crop_weed_image = crop_weed_images[0]
                    crop_weed_image = np.where(crop_weed_image.numpy() >= 0.5, 1, 0)    # 1:weed, 0: crop
                    predict_temp = crop_weed_image
                    crop_weed_object = crop_weed_objects[0]
                    crop_weed_object = np.where(crop_weed_object.numpy() >= 0.5, 1, 2)  # 1:crop/weed object, 2: background
                    crop_image = crop_images[0]
                    crop_image = np.where(crop_image.numpy() >= 0.5, 1, 0)  # 1:crop, 0:background
                    predict_temp2 = crop_image
                    crop_object = crop_objects[0]
                    crop_object = np.where(crop_object.numpy() >= 0.5, 1, 2)    # 1:crop object, 2:background
                    weed_image = weed_images[0]
                    weed_image = np.where(weed_image.numpy() >= 0.5, 1, 0)  # 1:weed, 0: background
                    predict_temp3 = weed_image
                    weed_object = weed_objects[0]
                    weed_object = np.where(weed_object.numpy() >= 0.5, 1, 2)    # 1: weed object, 2: background

                    cw_object_predict_axis = np.where(crop_weed_object==2)
                    predict_temp[cw_object_predict_axis] = 2    # 이 모델은 전 버전에서 잘 나온것을 확인했음
                    predict_temp = np.expand_dims(predict_temp, -1)

                    c_object_predict_axis = np.where(crop_object==2)
                    c_object_predict_axis2 = np.where(crop_object!=2)
                    predict_temp2[c_object_predict_axis] = 2
                    predict_temp2[c_object_predict_axis2] = 0
                    predict_temp2 = np.expand_dims(predict_temp2, -1)

                    w_object_predict_axis = np.where(weed_object==2)
                    w_object_predict_axis2 = np.where(weed_object!=2)
                    predict_temp3[w_object_predict_axis] = 2
                    predict_temp3[w_object_predict_axis2] = 1
                    predict_temp3 = np.expand_dims(predict_temp3, -1)

                    predict_temp4 = np.concatenate([predict_temp, predict_temp2, predict_temp3], -1)


                    pred_mask_color = np.where(predict_temp4 == np.array([0,0,0], dtype=np.uint8),
                                                np.array([255,0,0], dtype=np.uint8),
                                                predict_temp4)
                    pred_mask_color = np.where(predict_temp4 == np.array([1,1,1], dtype=np.uint8),
                                                np.array([0,0,255], dtype=np.uint8),
                                                pred_mask_color)
                    pred_mask_color = np.where(predict_temp4 == np.array([2,2,2], dtype=np.uint8),
                                                np.array([0,0,0], dtype=np.uint8),
                                                pred_mask_color)
                    func1 = lambda f:f != np.array([0,0,0], dtype=np.uint8)
                    func2 = lambda f:f != np.array([1,1,1], dtype=np.uint8)
                    pred_mask_color = np.where(func1(predict_temp4)&func2(predict_temp4),
                                                np.array([0,0,0], dtype=np.uint8),
                                                pred_mask_color)

                    func3 = lambda r: r[:, :, 0] == 255 
                    func4 = lambda g: g[:, :, 1] == 0 
                    func5 = lambda b: b[:, :, 2] == 0

                    func6 = lambda r: r[:, :, 0] == 0 
                    func7 = lambda g: g[:, :, 1] == 0 
                    func8 = lambda b: b[:, :, 2] == 255

                    metric_pred = np.zeros([FLAGS.img_size, FLAGS.img_size], dtype=np.uint8) + 2
                    metric_pred = np.where(func3(pred_mask_color) & func4(pred_mask_color) & func5(pred_mask_color), 0, 2)
                    metric_pred = np.where(func6(pred_mask_color) & func7(pred_mask_color) & func8(pred_mask_color), 1, metric_pred)

                    label = batch_labels[j]
                    label = tf.cast(label, tf.uint8).numpy()
                    label = np.where(label == FLAGS.ignore_label, 2, label)    # 2 is void
                    label = np.where(label == 255, 0, label)
                    label = np.where(label == 128, 1, label)
                    batch_label = label
                    label = np.concatenate((label, label, label), -1)
                    label_mask_color = np.zeros([FLAGS.img_size, FLAGS.img_size, 3], dtype=np.uint8)
                    label_mask_color = np.where(label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
                    label_mask_color = np.where(label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=metric_pred,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=FLAGS.total_classes).MIOU()
                    f1_score_, recall_ = Measurement(predict=metric_pred,
                                            label=batch_label,
                                            shape=[FLAGS.img_size*FLAGS.img_size, ],
                                            total_classes=FLAGS.total_classes).F1_score_and_recall()
                    tdr_ = Measurement(predict=metric_pred,
                                            label=batch_label,
                                            shape=[FLAGS.img_size*FLAGS.img_size, ],
                                            total_classes=FLAGS.total_classes).TDR()

                    miou += miou_
                    f1_score += f1_score_
                    sensitivity += recall_
                    tdr += tdr_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_
            print("Epoch: %3d, val mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), val F1_score = %.4f, val sensitivity = %.4f, val TDR = %.4f" % (epoch, miou / len(val_img_dataset),
                                                                                                                                         crop_iou / len(val_img_dataset),
                                                                                                                                         weed_iou / len(val_img_dataset),
                                                                                                                                         f1_score / len(val_img_dataset),
                                                                                                                                         sensitivity / len(val_img_dataset),
                                                                                                                                         tdr / len(val_img_dataset)))
            output_text.write("val mIoU: ")
            output_text.write("%.4f" % (miou / len(val_img_dataset)))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou / len(val_img_dataset)))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou / len(val_img_dataset)))
            output_text.write(", val F1_score: ")
            output_text.write("%.4f" % (f1_score / len(val_img_dataset)))
            output_text.write(", val sensitivity: ")
            output_text.write("%.4f" % (sensitivity / len(val_img_dataset)))
            output_text.write(", val TDR: ")
            output_text.write("%.4f" % (tdr / len(val_img_dataset)))
            output_text.write("\n")

            test_iter = iter(test_ge)
            miou = 0.
            f1_score = 0.
            tdr = 0.
            sensitivity = 0.
            crop_iou = 0.
            weed_iou = 0.
            for i in range(len(test_img_dataset)):
                batch_images, batch_labels = next(test_iter)
                batch_labels = tf.squeeze(batch_labels, -1)
                for j in range(1):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    logits = run_model(model, batch_image, False) # type을 batch label과 같은 type으로 맞춰주어야함
                    crop_weed_images = tf.nn.sigmoid(logits[:, :, :, 0])
                    crop_weed_objects = tf.nn.sigmoid(logits[:, :, :, 1])
                    crop_images = tf.nn.sigmoid(logits[:, :, :, 2])
                    crop_objects = tf.nn.sigmoid(logits[:, :, :, 3])
                    weed_images = tf.nn.sigmoid(logits[:, :, :, 4])
                    weed_objects = tf.nn.sigmoid(logits[:, :, :, 5])


                    crop_weed_image = crop_weed_images[0]
                    crop_weed_image = np.where(crop_weed_image.numpy() >= 0.5, 1, 0)    # 1:weed, 0: crop
                    predict_temp = crop_weed_image
                    crop_weed_object = crop_weed_objects[0]
                    crop_weed_object = np.where(crop_weed_object.numpy() >= 0.5, 1, 2)  # 1:crop/weed object, 2: background
                    crop_image = crop_images[0]
                    crop_image = np.where(crop_image.numpy() >= 0.5, 1, 0)  # 1:crop, 0:background
                    predict_temp2 = crop_image
                    crop_object = crop_objects[0]
                    crop_object = np.where(crop_object.numpy() >= 0.5, 1, 2)    # 1:crop object, 2:background
                    weed_image = weed_images[0]
                    weed_image = np.where(weed_image.numpy() >= 0.5, 1, 0)  # 1:weed, 0: background
                    predict_temp3 = weed_image
                    weed_object = weed_objects[0]
                    weed_object = np.where(weed_object.numpy() >= 0.5, 1, 2)    # 1: weed object, 2: background

                    cw_object_predict_axis = np.where(crop_weed_object==2)
                    predict_temp[cw_object_predict_axis] = 2    # 이 모델은 전 버전에서 잘 나온것을 확인했음
                    predict_temp = np.expand_dims(predict_temp, -1)

                    c_object_predict_axis = np.where(crop_object==2)
                    c_object_predict_axis2 = np.where(crop_object!=2)
                    predict_temp2[c_object_predict_axis] = 2
                    predict_temp2[c_object_predict_axis2] = 0
                    predict_temp2 = np.expand_dims(predict_temp2, -1)

                    w_object_predict_axis = np.where(weed_object==2)
                    w_object_predict_axis2 = np.where(weed_object!=2)
                    predict_temp3[w_object_predict_axis] = 2
                    predict_temp3[w_object_predict_axis2] = 1
                    predict_temp3 = np.expand_dims(predict_temp3, -1)

                    predict_temp4 = np.concatenate([predict_temp, predict_temp2, predict_temp3], -1)


                    pred_mask_color = np.where(predict_temp4 == np.array([0,0,0], dtype=np.uint8),
                                                np.array([255,0,0], dtype=np.uint8),
                                                predict_temp4)
                    pred_mask_color = np.where(predict_temp4 == np.array([1,1,1], dtype=np.uint8),
                                                np.array([0,0,255], dtype=np.uint8),
                                                pred_mask_color)
                    pred_mask_color = np.where(predict_temp4 == np.array([2,2,2], dtype=np.uint8),
                                                np.array([0,0,0], dtype=np.uint8),
                                                pred_mask_color)
                    func1 = lambda f:f != np.array([0,0,0], dtype=np.uint8)
                    func2 = lambda f:f != np.array([1,1,1], dtype=np.uint8)
                    pred_mask_color = np.where(func1(predict_temp4)&func2(predict_temp4),
                                                np.array([0,0,0], dtype=np.uint8),
                                                pred_mask_color)

                    func3 = lambda r: r[:, :, 0] == 255 
                    func4 = lambda g: g[:, :, 1] == 0 
                    func5 = lambda b: b[:, :, 2] == 0

                    func6 = lambda r: r[:, :, 0] == 0 
                    func7 = lambda g: g[:, :, 1] == 0 
                    func8 = lambda b: b[:, :, 2] == 255

                    metric_pred = np.zeros([FLAGS.img_size, FLAGS.img_size], dtype=np.uint8) + 2
                    metric_pred = np.where(func3(pred_mask_color) & func4(pred_mask_color) & func5(pred_mask_color), 0, 2)
                    metric_pred = np.where(func6(pred_mask_color) & func7(pred_mask_color) & func8(pred_mask_color), 1, metric_pred)

                    label = batch_labels[j]
                    label = tf.cast(label, tf.uint8).numpy()
                    label = np.where(label == FLAGS.ignore_label, 2, label)    # 2 is void
                    label = np.where(label == 255, 0, label)
                    label = np.where(label == 128, 1, label)
                    batch_label = label
                    label = np.concatenate((label, label, label), -1)
                    label_mask_color = np.zeros([FLAGS.img_size, FLAGS.img_size, 3], dtype=np.uint8)
                    label_mask_color = np.where(label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
                    label_mask_color = np.where(label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

                    miou_, crop_iou_, weed_iou_ = Measurement(predict=metric_pred,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=FLAGS.total_classes).MIOU()
                    f1_score_, recall_ = Measurement(predict=metric_pred,
                                            label=batch_label,
                                            shape=[FLAGS.img_size*FLAGS.img_size, ],
                                            total_classes=FLAGS.total_classes).F1_score_and_recall()
                    tdr_ = Measurement(predict=metric_pred,
                                            label=batch_label,
                                            shape=[FLAGS.img_size*FLAGS.img_size, ],
                                            total_classes=FLAGS.total_classes).TDR()

                    miou += miou_
                    f1_score += f1_score_
                    sensitivity += recall_
                    tdr += tdr_
                    crop_iou += crop_iou_
                    weed_iou += weed_iou_
            print("Epoch: %3d, test mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), test F1_score = %.4f, test sensitivity = %.4f, test TDR = %.4f" % (epoch, miou / len(test_img_dataset),
                                                                                                                                             crop_iou / len(test_img_dataset),
                                                                                                                                             weed_iou / len(test_img_dataset),
                                                                                                                                             f1_score / len(test_img_dataset),
                                                                                                                                             sensitivity / len(test_img_dataset),
                                                                                                                                             tdr / len(test_img_dataset)))
            print("=================================================================================================================================================")
            output_text.write("test mIoU: ")
            output_text.write("%.4f" % (miou / len(test_img_dataset)))
            output_text.write(", crop_iou: ")
            output_text.write("%.4f" % (crop_iou / len(test_img_dataset)))
            output_text.write(", weed_iou: ")
            output_text.write("%.4f" % (weed_iou / len(test_img_dataset)))
            output_text.write(", test F1_score: ")
            output_text.write("%.4f" % (f1_score / len(test_img_dataset)))
            output_text.write(", test sensitivity: ")
            output_text.write("%.4f" % (sensitivity / len(test_img_dataset)))
            output_text.write(", test TDR: ")
            output_text.write("%.4f" % (tdr / len(test_img_dataset)))
            output_text.write("\n")
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.flush()

            model_dir = "%s/%s" % (FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
               print("Make {} folder to store the weight!".format(epoch))
               os.makedirs(model_dir)
            ckpt = tf.train.Checkpoint(model=model, optim=optim)
            ckpt_dir = model_dir + "/Crop_weed_model_{}.ckpt".format(epoch)
            ckpt.save(ckpt_dir)
    else:
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        test_img_dataset = [FLAGS.image_path + data for data in test_list]
        test_lab_dataset = [FLAGS.label_path + data for data in test_list]

        test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
        test_ge = test_ge.map(test_func2)
        test_ge = test_ge.batch(1)
        test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

        test_iter = iter(test_ge)
        miou = 0.
        f1_score = 0.
        tdr = 0.
        sensitivity = 0.
        crop_iou = 0.
        weed_iou = 0.
        for i in range(len(test_img_dataset)):
            batch_images, nomral_img, batch_labels = next(test_iter)
            batch_labels = tf.squeeze(batch_labels, -1)
            for j in range(1):
                batch_image = tf.expand_dims(batch_images[j], 0)
                logits = run_model(model, batch_image, False) # type을 batch label과 같은 type으로 맞춰주어야함
                object_predict = tf.nn.sigmoid(logits[0, :, :, 1])
                predict = tf.nn.sigmoid(logits[0, :, :, 0:1])
                predict = np.where(predict.numpy() >= 0.5, 1, 0)
                predict_temp = predict
                object_predict_predict = np.where(object_predict.numpy() >= 0.5, 1, 2)
                onject_predict_axis = np.where(object_predict_predict==2)   # 2 배경성분이 있는 축만 가지고 옴
                predict_temp[onject_predict_axis] = 2

                #batch_image = tf.expand_dims(batch_images[j], 0)
                #predict = run_model(model, batch_image, False) # type을 batch label과 같은 type으로 맞춰주어야함
                #predict = tf.nn.sigmoid(predict[0, :, :, 0:1])
                #predict = np.where(predict.numpy() >= 0.5, 1, 0)

                batch_label = tf.cast(batch_labels[j], tf.uint8).numpy()
                batch_label = np.where(batch_label == FLAGS.ignore_label, 2, batch_label)    # 2 is void
                batch_label = np.where(batch_label == 255, 0, batch_label)
                batch_label = np.where(batch_label == 128, 1, batch_label)
                ignore_label_axis = np.where(batch_label==2)   # 출력은 x,y axis로 나옴!
                predict[ignore_label_axis] = 2

                miou_, crop_iou_, weed_iou_ = Measurement(predict=predict_temp,
                                    label=batch_label, 
                                    shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                    total_classes=FLAGS.total_classes).MIOU()
                f1_score_, recall_ = Measurement(predict=predict_temp,
                                        label=batch_label,
                                        shape=[FLAGS.img_size*FLAGS.img_size, ],
                                        total_classes=FLAGS.total_classes).F1_score_and_recall()
                tdr_ = Measurement(predict=predict_temp,
                                        label=batch_label,
                                        shape=[FLAGS.img_size*FLAGS.img_size, ],
                                        total_classes=FLAGS.total_classes).TDR()


                temp_img = predict

                pred_mask_color = color_map[predict_temp]  # 논문그림처럼 할것!
                pred_mask_color = np.squeeze(pred_mask_color, 2)
                batch_label = np.expand_dims(batch_label, -1)
                batch_label = np.concatenate((batch_label, batch_label, batch_label), -1)
                label_mask_color = np.zeros([FLAGS.img_size, FLAGS.img_size, 3], dtype=np.uint8)
                label_mask_color = np.where(batch_label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
                label_mask_color = np.where(batch_label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

                name = test_img_dataset[i].split("/")[-1].split(".")[0]
                plt.imsave(FLAGS.test_images + "/" + name + "_label.png", label_mask_color)
                plt.imsave(FLAGS.test_images + "/" + name + "_predict.png", pred_mask_color)

                miou += miou_
                f1_score += f1_score_
                sensitivity += recall_
                tdr += tdr_
                crop_iou += crop_iou_
                weed_iou += weed_iou_

        print("test mIoU = %.4f (crop_iou = %.4f, weed_iou = %.4f), test F1_score = %.4f, test sensitivity = %.4f, test TDR = %.4f" % (miou / len(test_img_dataset),
                                                                                                                                            crop_iou / len(test_img_dataset),
                                                                                                                                            weed_iou / len(test_img_dataset),
                                                                                                                                            f1_score / len(test_img_dataset),
                                                                                                                                            sensitivity / len(test_img_dataset),
                                                                                                                                            tdr / len(test_img_dataset)))


if __name__ == "__main__":
    main()
