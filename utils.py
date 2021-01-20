import numpy as np
import cv2
import os
import scipy.io as sio
from sklearn.metrics import confusion_matrix
from glob import glob
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf


folder_path = './data/OBX/'
im_path = folder_path + 'img/'
gt_path = folder_path + 'gt/mask_' # for calculating scores
eps = 1e-14


def dataloaderOBX(trainval_set, test_set, patch_size=256, stride_size=64, an_type='line', an_id=1, nb_classes=3): #, noclutter=True, remove_null=True):

    # path of sparse label
    sparse_label_path = folder_path + an_type + '/an' + str(an_id) + '/' #mask_'
    print(sparse_label_path)
    image_path = folder_path+ 'img/'
    print(image_path)

    imagefiles = np.array(sorted(glob(image_path+ '*.jpg')))[trainval_set]
    labelfiles = np.array(sorted(glob(sparse_label_path+ '*.jpg')))[trainval_set]

    # crop images to patches
    counter = 0
    for fid, lfid in zip(imagefiles, labelfiles): #range(len(trainval_set)):
        X, y = img2patchOBX(fid, lfid, patch_size, stride_size, nb_classes) #, noclutter, remove_null)
        X_tra = np.concatenate([X_tra, X], axis=0) if counter>0 else X
        y_tra = np.concatenate([y_tra, y], axis=0) if counter>0 else y
        counter +=1
        del X, y

    imagefiles = np.array(sorted(glob(image_path+ '*.jpg')))[test_set]
    labelfiles = np.array(sorted(glob(sparse_label_path+ '*.jpg')))[test_set]

    counter = 0
    for fid, lfid in zip(imagefiles, labelfiles): #range(len(trainval_set)):
        X, y = img2patchOBX(fid, lfid, patch_size, stride_size, nb_classes) #, noclutter, remove_null)
        X_test = np.concatenate([X_test, X], axis=0) if counter>0 else X
        y_test = np.concatenate([y_test, y], axis=0) if counter>0 else y
        del X, y
        counter +=1


    X_tra = np.float32(X_tra)
    y_tra = np.uint8(y_tra)
    X_test = np.float32(X_test)
    y_test = np.uint8(y_test)
    print('the size of training data:', np.shape(X_tra))

    return X_tra, y_tra, X_test, y_test


def img2patchOBX(fid, lfid, patch_size=256, stride_size=256, nb_classes=3): #, noclutter=True, remove_null=True):

    im = cv2.imread(fid)
    label = cv2.imread(lfid)[:,:,0]

    # label[label>=255]=2
    # label[(label<255) & (label>2)]=0
    # label[im[:,:,0]<100] = 1
    #
    label = tf.one_hot(tf.cast(label, tf.uint8), nb_classes) #3) #, on_value=1, off_value=0)

    gt = tf.squeeze(label).numpy()
    del label

    # crop an image/mask to patches
    X, y = [], []
    im_row, im_col, _ = np.shape(im)
    steps_row = int(np.floor((im_row - (patch_size - stride_size)) / stride_size))
    steps_col = int(np.floor((im_col - (patch_size - stride_size)) / stride_size))

    for i in range(steps_row+1):
        for j in range(steps_col+1):
            if i == steps_row:
                if j == steps_col:
                    X_patch = im[-patch_size:im_row, -patch_size:im_col, :]
                    y_patch = gt[-patch_size:im_row, -patch_size:im_col, :]
                else:
                    X_patch = im[-patch_size:im_row, (j * stride_size):(j * stride_size + patch_size),:]
                    y_patch = gt[-patch_size:im_row, (j * stride_size):(j * stride_size + patch_size),:]
            else:
                if j == steps_col:
                    X_patch = im[(i * stride_size):(i * stride_size + patch_size), -patch_size:im_col, :]
                    y_patch = gt[(i * stride_size):(i * stride_size + patch_size), -patch_size:im_col, :]
                else:
                    X_patch = im[(i * stride_size):(i * stride_size + patch_size), (j * stride_size):(j * stride_size + patch_size), :]
                    y_patch = gt[(i * stride_size):(i * stride_size + patch_size), (j * stride_size):(j * stride_size + patch_size), :]

            # if remove_null and np.sum(y_patch) == 0:
            #     continue

            X.append(X_patch)
            y.append(y_patch)

    X = np.float32(X)
    y = np.uint8(y)
    return X, y


def eval_imageOBX(gt, pred, acc1, acc2, acc3, acc4, acc5, cal_classes = 3): #, noclutter=True):

    im_row, im_col = np.shape(pred)
    # cal_classes = 2 #4 #5 if noclutter else 6 # no. of classes to calculate scores
    #cal_classes = 3 #if noclutter else 2 # no. of classes to calculate scores

    # if noclutter:
    #     gt[gt == 5] = 6 # pixels in clutter are not considered (regarding them as boundary)

    #pred[gt == 6] = 6 # pixels on the boundary are not considered for calculating scores
    OA = np.float32(len(np.where((np.float32(pred) - np.float32(gt)) == 0)[0])-len(np.where(gt==cal_classes+1)[0]))/np.float32(im_col*im_row-len(np.where(gt==cal_classes+1)[0]))
    acc1 = acc1 + len(np.where((np.float32(pred) - np.float32(gt)) == 0)[0])-len(np.where(gt==cal_classes+1)[0])
    acc2 = acc2 + im_col*im_row-len(np.where(gt==cal_classes+1)[0])
    pred1 = np.reshape(pred, (-1, 1))
    gt1 = np.reshape(gt, (-1, 1))
    idx = np.where(gt1==cal_classes+1)[0]
    pred1 = np.delete(pred1, idx)
    gt1 = np.delete(gt1, idx)
    CM = confusion_matrix(pred1, gt1)
    for i in range(cal_classes):
        tp = np.float32(CM[i, i])
        acc3[i] = acc3[i] + tp
        fp = np.sum(CM[:, i])-tp
        acc4[i] = acc4[i] + fp
        fn = np.sum(CM[i, :])-tp
        acc5[i] = acc5[i] + fn
        P = tp/(tp+fp+eps)
        R = tp/(tp+fn+eps)
        f1 = 2*(P*R)/(P+R+eps)

    return acc1, acc2, acc3, acc4, acc5


def pred_imageOBX(image_filename, label_filename, model, patch_size, stride_size, nb_classes = 3):

    # croppping an image into patches for prediction
    X, _ = img2patchOBX(image_filename, label_filename, patch_size, stride_size, nb_classes) #True, False)
    pred_patches = model.predict(X)

    # rearranging patchess into an image
    # For pixels with multiple predictions, we take their averages
    im_row, im_col, _ = np.shape(cv2.imread(image_filename))
    steps_col = int(np.floor((im_col - (patch_size - stride_size)) / stride_size))
    steps_row = int(np.floor((im_row - (patch_size - stride_size)) / stride_size))
    im_out = np.zeros((im_row, im_col, np.shape(pred_patches)[-1]))
    im_index = np.zeros((im_row, im_col, np.shape(pred_patches)[-1])) # counting the number of predictions for each pixel

    patch_id = 0
    for i in range(steps_row+1):
        for j in range(steps_col+1):
            if i == steps_row:
                if j == steps_col:
                    im_out[-patch_size:im_row, -patch_size:im_col, :] += pred_patches[patch_id]
                    im_index[-patch_size:im_row, -patch_size:im_col, :] += np.ones((patch_size, patch_size, np.shape(pred_patches)[-1]))
                else:
                    im_out[-patch_size:im_row, (j * stride_size):(j * stride_size + patch_size), :] += pred_patches[patch_id]
                    im_index[-patch_size:im_row, (j * stride_size):(j * stride_size + patch_size), :] += np.ones((patch_size, patch_size, np.shape(pred_patches)[-1]))
            else:
                if j == steps_col:
                    im_out[(i * stride_size):(i * stride_size + patch_size), -patch_size:im_col, :] += pred_patches[patch_id]
                    im_index[(i * stride_size):(i * stride_size + patch_size), -patch_size:im_col, :] += np.ones((patch_size, patch_size, np.shape(pred_patches)[-1]))
                else:
                    im_out[(i * stride_size):(i * stride_size + patch_size), (j * stride_size):(j * stride_size + patch_size), :] += pred_patches[patch_id]
                    im_index[(i * stride_size):(i * stride_size + patch_size), (j * stride_size):(j * stride_size + patch_size), :] += np.ones((patch_size, patch_size, np.shape(pred_patches)[-1]))
            patch_id += 1

    return im_out/im_index

def TestModelOBX(test_set,model, out_folder='model', patch_size=256, stride_size=128, nb_classes = 3):#, noclutter=True):

    # path for saving output
    #output_path = folder_path + 'outputs/' + output_folder + '/'
    if not os.path.isdir(out_folder): #output_path):
        print('The target folder is created.')
        os.mkdir(out_folder) #output_path)

    #nb_classes = 3 #1 if noclutter else 2

    # nb_classes = 2 #4 #5 if noclutter else 6
    acc1 = 0.0 # accumulator for correctly classified pixels
    acc2 = 0.0 # accumulator for all valid pixels (not including label 0 and 6)
    acc3 = np.zeros((nb_classes, 1)) # accumulator for true positives
    acc4 = np.zeros((nb_classes, 1)) # accumulator for false positives
    acc5 = np.zeros((nb_classes, 1)) # accumulator for false negatives

    an_type = 'line'
    an_id = 1

    sparse_label_path = folder_path + 'gt' + '/' #mask_'
    print(sparse_label_path)
    image_path = folder_path+ 'img/'
    print(image_path)

    imagefiles = np.array(sorted(glob(image_path+ '*.jpg')))[test_set]
    labelfiles = np.array(sorted(glob(sparse_label_path+ '*.jpg')))[test_set]

    for image_filename, label_filename in zip(imagefiles, labelfiles):
        #im = cv2.imread(image_filename)
        bits = tf.io.read_file(label_filename.replace('./', os.getcwd()+os.sep))
        label = tf.image.decode_png(bits, channels=1)
        del bits
        gt = tf.cast(label, tf.uint8).numpy().squeeze()

        # predict one image
        pred = pred_imageOBX(image_filename, label_filename, model, patch_size, stride_size, nb_classes)
        pred = np.argmax(pred, -1)
        #gt = np.argmax(gt, -1)

        plt.subplot(122); plt.imshow(pred, cmap='bwr', vmin=0, vmax=2); plt.title('Pred', fontsize=7);  plt.axis('off')
        plt.subplot(121); plt.imshow(gt, cmap='bwr', vmin=0, vmax=2); plt.title('GT', fontsize=7); plt.axis('off')
        #plt.show()
        plt.savefig(out_folder+os.sep+image_filename.split(os.sep)[-1].replace('img/','outputs/'), dpi=300, bbox_inches='tight')
        plt.close()

        # evaluate one image
        acc1, acc2, acc3, acc4, acc5 = eval_imageOBX(gt, pred, acc1, acc2, acc3, acc4, acc5, nb_classes) #, noclutter)
        #cv2.imwrite(image_filename.replace('img/','outputs/'), pred) #index2bgr(pred, True))

    #
    # # predicting and measuring all images
    # for im_id in range(len(test_set)):
    #     filename = im_header + str(test_set[im_id]) + '.tif'
    #     print(im_id+1, '/', len(test_set), ': predicting ', filename)
    #     gt = bgr2index(cv2.imread(gt_path + filename), True)
    #
    #     # predict one image
    #     pred = pred_image(filename, model, patch_size, stride_size)
    #     pred = np.argmax(pred, -1)
    #     gt = np.argmax(gt, -1)
    #
    #     # evaluate one image
    #     acc1, acc2, acc3, acc4, acc5 = eval_image(gt, pred, acc1, acc2, acc3, acc4, acc5, noclutter)
    #     cv2.imwrite(output_path+filename, index2bgr(pred, True))
    #     print('Prediction is done. The output is saved in ', output_path)

    OA = acc1/acc2

    f1 = np.zeros((nb_classes, 1));
    iou = np.zeros((nb_classes, 1));
    #ca = np.zeros((nb_classes, 1));
    for i in range(nb_classes):
        P = acc3[i]/(acc3[i]+acc4[i])
        R = acc3[i]/(acc3[i]+acc5[i])
        f1[i] = 2*(P*R)/(P+R)
        iou[i] = acc3[i]/(acc3[i]+acc4[i]+acc5[i])
        #ca[i] =  acc3[i]/(acc3[i]+acc4[i])

    f1_mean = np.mean(f1)
    iou_mean = np.mean(iou)
    #ca_mean = np.mean(ca)
    print('mean f1:', f1_mean, '\nmean iou:', iou_mean, '\nOA:', OA)

    return 'All predicitions are done, and output images are saved.'
