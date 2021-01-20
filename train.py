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

# ******************** image & label config ***********************
patch_size = 128 # 256 #size of each patch
stride_size = 128 #64 # stride of sliding window
an_type = 'polygon' #'line'# 'polygon' # type of sparse annotations
an_id = 1 # id of annotators: 1 and 2 are expert, 3 and 4 are non-expert

# ************************ training scheme *************************
batch_size = 2 # size of training batch
epochs = 20 #100 # number of training epochs
lr = 2e-4 # initial learning rate
lambda_festa = 0.01 # 0.1 # lambda in Eq. 2, weight of festa

VAL_SPLIT = 0.5 # 0.05

nb_classes = 3

min_lr=0.5e-10
patience=5
patience_lr=0
factor=0.1

trainval_set = np.arange(20).tolist()
test_set = np.arange(20,165).tolist() #165

# **************************** path ********************************
weight_path = 'weights/fcn-obx_patch'+str(patch_size)+'_stride'+str(stride_size)+'_batch'+str(batch_size)+'_lambda'+str(lambda_festa)+'.h5'

# **************************** losses ********************************
loss = [L_festa, 'categorical_crossentropy'] # final loss Eq. 2
loss_weights = [lambda_festa, 1] # weight of each loss term in Eq. 2

# ********************** loading data *****************************
print('loading training data ...')
X_tra, y_tra, _, _ = dataloaderOBX(trainval_set, test_set, patch_size, stride_size, an_type, an_id, nb_classes) #, noclutter, remove_null)
print('training data is loaded.')

print(X_tra.shape)
print(y_tra.shape)

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


plt.subplot(231)
plt.plot(history.history['loss'], 'k')
plt.plot(history.history['val_loss'], 'r')
plt.title('Categorical crossentropy loss')

plt.subplot(232)
plt.plot(history.history['final_feat_loss'], 'k')
plt.plot(history.history['val_final_feat_loss'], 'r')
plt.title('Feature loss')

plt.subplot(233)
plt.plot(history.history['final_out_loss'], 'k')
plt.plot(history.history['val_final_out_loss'], 'r')
plt.title('Combined loss')

plt.subplot(234)
plt.plot(history.history['final_feat_accuracy'], 'k')
plt.plot(history.history['val_final_feat_accuracy'], 'r')
plt.title('Feature accuracy')

plt.subplot(235)
plt.plot(history.history['final_out_accuracy'], 'k')
plt.plot(history.history['val_final_out_accuracy'], 'r')
plt.title('Combined accuracy')

plt.subplot(236)
plt.plot(history.history['lr'], 'k')
plt.title('Learning rate')


# plt.show()
plt.savefig(weight_path.replace('.h5','.png'), dpi=300, bbox_inches='tight')


out_folder = 'festa'

# ********************* initialize model ********************
model = fcn_festa(patch_size, True, nb_classes)#, noclutter)
model.load_weights(weight_path, by_name=True)

# ********************* evaluate ****************************
TestModelOBX(test_set,model, out_folder, patch_size, stride_size, nb_classes) #, noclutter)
