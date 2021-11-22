# -*- coding:utf-8 -*-
#%%
from modified_deeplab_V3 import *
from PFB_measurement import Measurement
from random import shuffle, random

import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 512,

                           "train_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/CWFID/train_fix.txt",

                           "val_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/CWFID/val.txt",

                           "test_txt_path": "/yuwhan/yuwhan/Dataset/Segmentation/CWFID/test.txt",
                           
                           "label_path": "/yuwhan/yuwhan/Dataset/Segmentation/CWFID/aug_train_masks/",
                           
                           "image_path": "/yuwhan/yuwhan/Dataset/Segmentation/CWFID/aug_train_images/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "/yuhwan/yuhwan/checkpoint/Segmenation/V3_5th_paper/BoniRob/checkpoint/199",
                           
                           "lr": 0.0001,

                           "min_lr": 1e-7,
                           
                           "epochs": 200,

                           "total_classes": 3,

                           "ignore_label": 0,

                           "batch_size": 4,

                           "sample_images": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V3/CWFID/sample_images",

                           "save_checkpoint": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V3/CWFID/checkpoint",

                           "save_print": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V3/CWFID/train_out.txt",

                           "test_images": "D:/[1]DB/[5]4th_paper_DB/crop_weed/V2/test_images",

                           "train": True})


optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
optim2 = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
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

# @tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    # y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

@tf.function
def cal_loss(model, images, labels, objectiness, class_im_plain, ob_imbal_labels_buf, ignore_label):

    with tf.GradientTape() as tape:

        
        batch_labels = tf.reshape(labels, [-1,])
        indices = tf.squeeze(tf.where(tf.not_equal(batch_labels, ignore_label)),1)
        batch_labels = tf.cast(tf.gather(batch_labels, indices), tf.float32)

        logits = run_model(model, images, True)
        raw_logits = tf.reshape(logits, [-1, FLAGS.total_classes-1])
        #print(raw_logits.shape)
        predict = tf.gather(raw_logits, indices)
        #print(predict.shape)

        crop_indices = tf.squeeze(tf.where(tf.not_equal(batch_labels, 1)),1)
        weed_indices = tf.squeeze(tf.where(tf.not_equal(batch_labels, 0)),1)

        crop_labels = tf.cast(tf.gather(batch_labels, crop_indices), tf.float32)
        crop_predict = tf.gather(predict, crop_indices)
        weed_labels = tf.cast(tf.gather(batch_labels, weed_indices), tf.float32)
        weed_predict = tf.gather(predict, weed_indices)

        label_objectiness = tf.cast(tf.reshape(objectiness, [-1,]), tf.float32)
        logit_objectiness = raw_logits[:, -1]

        crop_dice_loss = dice_loss(1. - crop_labels, 1. - tf.nn.sigmoid(tf.squeeze(crop_predict[:, 0:1], -1)))
        weed_dice_loss = dice_loss(weed_labels, tf.nn.sigmoid(tf.squeeze(weed_predict[:, 0:1], -1)))

        no_obj_indices = tf.squeeze(tf.where(tf.equal(tf.reshape(objectiness, [-1,]), 0)),1)
        no_logit_objectiness = tf.gather(logit_objectiness, no_obj_indices)
        no_obj_labels = tf.cast(tf.gather(label_objectiness, no_obj_indices), tf.float32)
        no_obj_loss = -(1. - no_obj_labels) * tf.math.log(1 - tf.nn.sigmoid(no_logit_objectiness) + 1e-7) * ob_imbal_labels_buf[1] * ((tf.nn.sigmoid(no_logit_objectiness)**2))
        no_obj_loss = tf.reduce_mean(no_obj_loss)

        obj_indices = tf.squeeze(tf.where(tf.not_equal(tf.reshape(objectiness, [-1,]), 0)),1)
        yes_logit_objectiness = tf.gather(logit_objectiness, obj_indices)
        yes_obj_labels = tf.cast(tf.gather(label_objectiness, obj_indices), tf.float32)
        obj_loss = -yes_obj_labels * tf.math.log(tf.nn.sigmoid(yes_logit_objectiness) + 1e-7) * ob_imbal_labels_buf[0] * ((1-tf.nn.sigmoid(yes_logit_objectiness))**2)
        # obj_loss = (tf.reduce_sum(obj_loss + weed_dice_loss) / float(len(yes_obj_labels))) + (tf.reduce_sum(obj_loss + crop_dice_loss) / float(len(yes_obj_labels)))
        obj_loss = tf.reduce_mean(obj_loss)
        
        seg_loss = tf.reduce_sum(crop_dice_loss + class_im_plain[0]*((tf.nn.sigmoid(crop_predict[:, 0])**2))*tf.nn.sigmoid_cross_entropy_with_logits(crop_labels, tf.squeeze(crop_predict[:, 0:1], -1))) / float(len(crop_labels)) \
            + tf.reduce_sum(weed_dice_loss + class_im_plain[1]*((1-tf.nn.sigmoid(weed_predict[:, 0]))**2)*tf.nn.sigmoid_cross_entropy_with_logits(weed_labels, tf.squeeze(weed_predict[:, 0:1], -1))) / float(len(weed_labels))
        # seg_loss = dice_loss(batch_labels, tf.squeeze(predict[:, 0:1], -1)) \
        #     + tf.nn.sigmoid_cross_entropy_with_logits(batch_labels, tf.squeeze(predict[:, 0:1], -1))

        # seg_loss = tf.reduce_mean(seg_loss)
        
        loss = no_obj_loss + (seg_loss + obj_loss)
        #loss = seg_loss
    weights = model.trainable_variables
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# yilog(h(xi;θ))+(1−yi)log(1−h(xi;θ))
def main():
    tf.keras.backend.clear_session()
    # 마지막 plain은 objecttines에 대한 True or False값 즉 (mask값이고), 라벨은 annotation 이미지임 (crop/weed)
    # 학습이미지에 대해 online augmentation을 진행--> 전처리로서 필터링을 하던지 해서 , 피사체에 대한 high frequency 성분을
    # 가지고오자
    #model = PFB_model(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), OUTPUT_CHANNELS=FLAGS.total_classes-1)
    model = DeepLabV3Plus(FLAGS.img_size, FLAGS.img_size, 34)
    out = model.get_layer("activation_decoder_2_upsample").output
    out = tf.keras.layers.Conv2D(2, (1,1), name="output_layer")(out)
    model = tf.keras.Model(inputs=model.input, outputs=out)
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        #elif isinstance(layer, tf.keras.layers.Conv2D):
        #    layer.kernel_regularizer = tf.keras.regularizers.l2(0.0005)

    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim, optim2=optim2)
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
                batch_labels = np.squeeze(batch_labels, -1)

                
                class_imbal_labels = batch_labels
                class_imbal_labels_buf = 0.
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels[i]
                    class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=FLAGS.total_classes)
                    class_imbal_labels_buf += count_c_i_lab
                class_imbal_labels_buf /= FLAGS.batch_size
                ob_imbal_labels_buf = np.array([class_imbal_labels_buf[0] + class_imbal_labels_buf[1], class_imbal_labels_buf[2]], np.float32)
                ob_imbal_labels_buf = (np.max(ob_imbal_labels_buf / np.sum(ob_imbal_labels_buf)) + 1 - (ob_imbal_labels_buf / np.sum(ob_imbal_labels_buf)))
                ob_imbal_labels_buf = np.array(ob_imbal_labels_buf, np.float32)
                class_imbal_labels_buf = class_imbal_labels_buf[0:FLAGS.total_classes-1]
                class_imbal_labels_buf = (np.max(class_imbal_labels_buf / np.sum(class_imbal_labels_buf)) + 1 - (class_imbal_labels_buf / np.sum(class_imbal_labels_buf)))
                class_imbal_labels_buf = np.array(class_imbal_labels_buf, np.float32)
                # class_im_plain = np.where(batch_labels == 0, class_imbal_labels_buf[0], batch_labels)
                # class_im_plain = np.where(batch_labels == 1, class_imbal_labels_buf[1], batch_labels)
                class_im_plain = tf.cast(class_imbal_labels_buf, tf.float32).numpy()

                objectiness = np.where(batch_labels == 2, 0, 1)  # 피사체가 있는곳은 1 없는곳은 0으로 만들어준것

                loss = cal_loss(model, batch_images, batch_labels, objectiness, class_im_plain, ob_imbal_labels_buf, 2)
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] loss = {}".format(epoch, step+1, tr_idx, loss))

                if count % 100 == 0:

                    logits = run_model(model, batch_images, False)
                    object_images = tf.nn.sigmoid(logits[:, :, :, 1])
                    images = tf.nn.sigmoid(logits[:, :, :, 0:1])
                    for i in range(FLAGS.batch_size):
                        object_image = object_images[i]
                        image = images[i]
                        label = batch_labels[i]
                        image = np.where(image.numpy() >= 0.5, 1, 0)
                        temp_image = image
                        object_image_predict = np.where(object_image.numpy() >= 0.5, 1, 2)
                        object_image_axis = np.where(object_image_predict==2)
                        temp_image[object_image_axis] = 2


                        pred_mask_color = color_map[temp_image]  # 논문그림처럼 할것!
                        pred_mask_color = np.squeeze(pred_mask_color, 2)
                        label = np.expand_dims(label, -1)
                        label = np.concatenate((label, label, label), -1)
                        label_mask_color = np.zeros([FLAGS.img_size, FLAGS.img_size, 3], dtype=np.uint8)
                        label_mask_color = np.where(label == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), label_mask_color)
                        label_mask_color = np.where(label == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), label_mask_color)

                        temp_image = np.concatenate((temp_image, temp_image, temp_image), -1)
                        image = np.concatenate((image, image, image), -1)
                        pred_mask_warping = np.where(temp_image == np.array([2,2,2], dtype=np.uint8), print_images[i], image)
                        pred_mask_warping = np.where(temp_image == np.array([0,0,0], dtype=np.uint8), np.array([255, 0, 0], dtype=np.uint8), pred_mask_warping)
                        pred_mask_warping = np.where(temp_image == np.array([1,1,1], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8), pred_mask_warping)
                        pred_mask_warping /= 255.

                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_label.png", label_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_predict.png", pred_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_warping_predict.png", pred_mask_warping)
                    

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


if __name__ == "__main__":
    main()
# %%
