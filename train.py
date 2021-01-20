# import tensorflow as tf
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#
import matplotlib.pyplot as plt
from fcn import *
from utils import *
from loss import *
from keras.optimizers import Nadam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import keras.backend as K

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

# ******************** image & label config ***********************
patch_size = 256 # 256 #size of each patch
stride_size = 64 #64 # stride of sliding window
an_type = 'polygon' #'line'# 'polygon' # type of sparse annotations
an_id = 1 # id of annotators: 1 and 2 are expert, 3 and 4 are non-expert


VAL_SPLIT = 0.25 # 0.05

nb_classes = 3

min_lr=0.5e-10
patience=5
patience_lr=0
factor=0.1

# ************************ training scheme *************************
batch_size = 2 # size of training batch
epochs = 20 #100 # number of training epochs
lr = 2e-4 # initial learning rate

alpha = 0.5 # weight of neighbour in the feature space
# beta = 2 #5 #1.5 # weight of neighbour in the image space
gamma = 1 # weight of far-away in the feature space
sample_ratio = 0.1 #0.01 # measure only sample_ratio % samples for computational efficiency


#lambda_festa = 0.1 #0.1 #0.5 #0.1 # lambda in Eq. 2, weight of festa

trainval_set = np.arange(20).tolist()
test_set = np.arange(20,30).tolist() #165

for beta in [1,2,3,4,5]:

    for lambda_festa in [0.0, 0.1, 0.2, 0.3]:

        # **************************** path ********************************
        weight_path = 'weights/fcn-obx_patch'+str(patch_size)+'_stride'+str(stride_size)+'_batch'+str(batch_size)+'_lambda'+str(lambda_festa)+'_alpha'+str(alpha)+'_beta'+str(beta)+'_gamma'+str(gamma)+'.h5'

        # **************************** losses ********************************
        # loss = [L_festa, 'categorical_crossentropy'] # final loss Eq. 2

        loss = [L_festa(alpha, beta, gamma, sample_ratio), 'categorical_crossentropy'] # final loss Eq. 2

        loss_weights = [lambda_festa, 1] # weight of each loss term in Eq. 2

        # ********************** loading data *****************************
        print('loading training data ...')
        X_tra, y_tra, _, _ = dataloaderOBX(trainval_set, test_set, patch_size, stride_size, an_type, an_id, nb_classes) #, noclutter, remove_null)
        print('training data is loaded.')

        print(X_tra.shape)
        print(y_tra.shape)


        X_tra, y_tra = unison_shuffled_copies(X_tra, y_tra)

        for k in range(1,10):
            plt.subplot(3,3,k)
            plt.imshow(X_tra[k].astype(np.uint8))
            plt.imshow(np.argmax(y_tra[k],-1), alpha=0.3, cmap='bwr', vmin=0, vmax=2)
            plt.axis('off')

        plt.savefig('examples'+str(np.random.randint(0,1000))+'.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ********************* initialize model ********************
        model = fcn_festa(patch_size, False, nb_classes) #, noclutter)
        optimizer = Nadam(lr=lr) # define yourself, e.g. sgd, adam

        # optimizer = Adam(lr=lr) # define yourself, e.g. sgd, adam
        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=['accuracy'])
        print('model is built')

        # ********************* train ***********************************
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=factor, cooldown=0, patience=patience_lr, min_lr=min_lr)

        earlystop = EarlyStopping(monitor="val_loss", mode="min", patience=patience)


        history = model.fit(X_tra, [y_tra, y_tra], batch_size=batch_size, shuffle = True, epochs=epochs, validation_split=VAL_SPLIT, callbacks=[lr_reducer, earlystop])
        model.save_weights(weight_path)

        del X_tra, y_tra
        K.clear_session()


        plt.subplot(231)
        plt.plot(history.history['loss'], 'k')
        plt.plot(history.history['val_loss'], 'r')
        plt.title('Categorical crossentropy loss')

        # plt.subplot(232)
        # plt.plot(history.history['final_feat_loss'], 'k')
        # plt.plot(history.history['val_final_feat_loss'], 'r')
        # plt.title('Feature loss')

        plt.subplot(222)
        plt.plot(history.history['final_out_loss'], 'k')
        plt.plot(history.history['val_final_out_loss'], 'r')
        plt.title('Combined loss')

        plt.subplot(223)
        plt.plot(history.history['final_feat_accuracy'], 'k')
        plt.plot(history.history['val_final_feat_accuracy'], 'r')
        plt.title('Feature accuracy')

        plt.subplot(224)
        plt.plot(history.history['final_out_accuracy'], 'k')
        plt.plot(history.history['val_final_out_accuracy'], 'r')
        plt.title('Combined accuracy')

        # plt.subplot(236)
        # plt.plot(history.history['lr'], 'k')
        # plt.title('Learning rate')

        # plt.show()
        plt.savefig(weight_path.replace('.h5','.png'), dpi=300, bbox_inches='tight')

        #
        # # out_folder = 'festa'
        # out_folder = 'data/OBX/outputs/fcn-obx_patch'+str(patch_size)+'_stride'+str(stride_size)+'_batch'+str(batch_size)+'_lambda'+str(lambda_festa)
        #
        # # ********************* initialize model ********************
        # model = fcn_festa(patch_size, True, nb_classes)#, noclutter)
        # model.load_weights(weight_path, by_name=True)
        #
        # # ********************* evaluate ****************************
        # TestModelOBX(test_set,model, out_folder, patch_size, stride_size, nb_classes) #, noclutter)
