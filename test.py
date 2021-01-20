
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from fcn import *
from utils import *

import keras.backend as K
K.clear_session()

patch_size = 256 # 256 #size of each patch
stride_size = 64 #64 # stride of sliding window
batch_size = 2 # size of training batch
lambda_festa = 0.1 #0.5 #0.1 # lambda in Eq. 2, weight of festa

VAL_SPLIT = 0.25 # 0.05

nb_classes = 3

test_set = np.arange(20,30).tolist() #165


weight_path = 'weights/fcn-obx_patch'+str(patch_size)+'_stride'+str(stride_size)+'_batch'+str(batch_size)+'_lambda'+str(lambda_festa)+'.h5'


# out_folder = 'festa'
out_folder = 'data/OBX/outputs/fcn-obx_patch'+str(patch_size)+'_stride'+str(stride_size)+'_batch'+str(batch_size)+'_lambda'+str(lambda_festa)

# ********************* initialize model ********************
model = fcn_festa(patch_size, True, nb_classes)#, noclutter)
model.load_weights(weight_path, by_name=True)

# ********************* evaluate ****************************
TestModelOBX(test_set,model, out_folder, patch_size, stride_size, nb_classes) #, noclutter)
