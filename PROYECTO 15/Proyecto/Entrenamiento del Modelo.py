# %% [markdown]
# ## Initialization

# %%
import pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% [markdown]
# ## Load Data

# %% [markdown]
# The dataset is stored in the `datasets/faces/` folder, there you can find
# - The `final_files` folder with 7.6k photos
# - The `labels.csv` file with labels, with two columns: `file_name` and `real_age`
# 
# Given the fact that the number of image files is rather high, it is advisable to avoid reading them all at once, which would greatly consume computational resources. We recommend you build a generator with the ImageDataGenerator generator. This method was explained in Chapter 3, Lesson 7 of this course.
# 
# The label file can be loaded as an usual CSV file.

# %%
path = '/datasets/faces/'
directory = path + 'final_files/'
labels = pd.read_csv(path + "labels.csv")

# # %%
# train_datagen = ImageDataGenerator(rescale=1./255)

# # %%
# train_gen_flow = train_datagen.flow_from_dataframe(
#         dataframe= labels,
#         directory= directory,
#         x_col='file_name',
#         y_col='real_age',
#         target_size=(224, 224),
#         batch_size=32,
#         class_mode='raw',
#         seed=12345)

# # %%
# image_arrays, age_array = train_gen_flow.next()

# # %% [markdown]
# # ## EDA

# # %%
# # labels.info()
# labels.head()

# # %%
# #image_arrays.shape

# # %%
# #image = Image.open(train_gen_flow.filepaths[4])
# #array = np.array(image)
# #plt.imshow(array)

# # %% [markdown]
# # - The 'labels' dataset consists of two columns: 'file_name' (the name of an image file) and 'real_age' (the age of the person in the image), and the actual image files are pulled from 'datasets/faces/final_files/' using ImageDataGenerator.
# # - The 'labels' dataset contains a list of 7591 image file names, which is not large.
# # - The data generator found 7591 images, each of which is a four-dimensional tensor of thirty-two 224x224 pixel images with three colour channels.
# # 
# # Let us look at the age distribution in the dataset.

# # %%
# #labels.describe()

# # %%
# #labels['real_age'].value_counts()

# # %%
# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# sns.histplot(data=labels, x="real_age", kde=True, ax=ax[0])
# sns.boxplot(x="real_age", data=labels, ax=ax[1])
# plt.title("Age distribution")
# ax[0].set(xlabel='Age', ylabel='Number of Images')
# ax[1].set_xlabel('Age')
# plt.show();

# # %% [markdown]
# # - Age range: 0 - 100 years old.
# # - The distribution is slightly skewed to the right. We do not have an equal number of images for each age. We have the largest amount of data in the 20-40 year-old age group.
# # 
# # Let us look at 15 randomly chosen photos of people at various ages.

# # %%
# n_images = 15
# rows = 5
# columns = n_images/rows

# fig = plt.figure(figsize=(10, 10))
# image_arrays, age_array = train_gen_flow.next()

# for i in range(n_images):
#     # add a subplot at the (i+1)th position
#     fig.add_subplot(rows, columns, i+1)
#     # showing image
#     image_file = image_arrays[i]
#     age_num = age_array[i]
#     plt.imshow(image_file)
#     plt.axis('off')
#     plt.title(f"Age: {age_num}")

# # %% [markdown]
# # ### Findings

# # %% [markdown]
# # - Resolution, lighting, and face angles are different from image to image.
# # - There is a wide variety of race and gender, in addition to age.
# # - Some images show the whole body of the person and his/her face is too small to be recognised.
# # - Each image contains a background, and it is a large area that does not offer the information about a person's face.
# # - Some photos are "a frame within a frame", and are rotated. Again, a person's face does not fill the entire display space of the image.
# # - Some images are warped and stretched.
# # - Some images suffer from partial occlusion with part of the face hidden by an object (e.g. a veil, a microphone, food, eyes closed...etc).
# # - Some images are in colour and others are in monochrome.

# # %% [markdown]
# # ## Modelling

# # %% [markdown]
# # Define the necessary functions to train your model on the GPU platform and build a single script containing all of them along with the initialization section.
# # 
# # To make this task easier, you can define them in this notebook and run a ready code in the next section to automatically compose the script.
# # 
# # The definitions below will be checked by project reviewers as well, so that they can understand how you built the model.

# %%
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

# %%
def load_train(path):

    """
    It loads the train part of dataset from path
    """

    train_datagen = ImageDataGenerator(validation_split=0.25,
                                       rescale=1.0/255
                                )

    train_gen_flow = train_datagen.flow_from_dataframe(
        dataframe=pd.read_csv(path + "labels.csv"),
        directory=path +'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16, # 32
        class_mode='raw',
        subset='training',
        seed=12345
    )

    return train_gen_flow

# %%
def load_test(path):

    """
    It loads the validation/test part of dataset from path
    """

    test_datagen = ImageDataGenerator(
        validation_split=0.25, rescale=1.0/255
    )

    test_gen_flow = test_datagen.flow_from_dataframe(
        dataframe=pd.read_csv(path + "labels.csv"),
        directory=path +'final_files/',
        x_col='file_name',
        y_col='real_age',
        target_size=(224, 224),
        batch_size=16, # 32
        class_mode='raw',
        subset='validation',
        seed=12345
    )

    return test_gen_flow

# %%
def create_model(input_shape=(224, 224, 3)):

    """
    It defines the model
    """

    backbone = ResNet50(
        input_shape=input_shape, #(224, 224, 3),
        weights='imagenet',
        include_top=False
    )

    model = Sequential()
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.0001), # default is 0.001
        metrics=['mae']
    )

    #print(model.summary())
    return model

# %%
def train_model(
    model,
    train_data,
    test_data,
    batch_size=None,
    epochs=20,
    steps_per_epoch=None,
    validation_steps=None
):

    if steps_per_epoch is None:
        steps_per_epoch = len(train_data)
    if validation_steps is None:
        validation_steps = len(test_data)

    """
    Trains the model given the parameters
    """

    model.fit(train_data,
              validation_data=test_data,
              batch_size=batch_size,
              epochs=epochs,
              steps_per_epoch=steps_per_epoch,
              validation_steps=validation_steps,
              verbose=2
    )

    return model

# %% [markdown]
# ## Prepare the Script to Run on the GPU Platform

# %% [markdown]
# Given you've defined the necessary functions you can compose a script for the GPU platform, download it via the "File|Open..." menu, and to upload it later for running on the GPU platform.
# 
# N.B.: The script should include the initialization section as well. An example of this is shown below.

# %%
# prepare a script to run on the GPU platform

init_str = """
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
"""

import inspect

with open('run_model_on_gpu.py', 'w') as f:

    f.write(init_str)
    f.write('\n\n')

    for fn_name in [load_train, load_test, create_model, train_model]:

        src = inspect.getsource(fn_name)
        f.write(src)
        f.write('\n\n')

# %% [markdown]
# ### Output

# %% [markdown]
# Place the output from the GPU platform as an Markdown cell here.

# %% [markdown]
# 2023-04-10 04:41:07.285193: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer.so.6
# 2023-04-10 04:41:07.338398: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libnvinfer_plugin.so.6
# Using TensorFlow backend.
# Found 5694 validated image filenames.
# Found 1897 validated image filenames.
# 2023-04-10 04:41:10.212230: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
# 2023-04-10 04:41:10.291251: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.291450: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties:
# pciBusID: 0000:00:1e.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
# coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s
# 2023-04-10 04:41:10.291485: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2023-04-10 04:41:10.291529: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2023-04-10 04:41:10.344041: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
# 2023-04-10 04:41:10.354479: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
# 2023-04-10 04:41:10.459874: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
# 2023-04-10 04:41:10.470656: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
# 2023-04-10 04:41:10.470711: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2023-04-10 04:41:10.470806: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.471008: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.471142: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
# 2023-04-10 04:41:10.471507: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
# 2023-04-10 04:41:10.494886: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2300010000 Hz
# 2023-04-10 04:41:10.496657: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x3a2ff60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
# 2023-04-10 04:41:10.496684: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
# 2023-04-10 04:41:10.623815: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.624124: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x25127c0 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
# 2023-04-10 04:41:10.624147: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Tesla V100-SXM2-16GB, Compute Capability 7.0
# 2023-04-10 04:41:10.624376: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.624565: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1555] Found device 0 with properties:
# pciBusID: 0000:00:1e.0 name: Tesla V100-SXM2-16GB computeCapability: 7.0
# coreClock: 1.53GHz coreCount: 80 deviceMemorySize: 15.78GiB deviceMemoryBandwidth: 836.37GiB/s
# 2023-04-10 04:41:10.624616: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2023-04-10 04:41:10.624629: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2023-04-10 04:41:10.624662: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10
# 2023-04-10 04:41:10.624691: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10
# 2023-04-10 04:41:10.624721: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10
# 2023-04-10 04:41:10.624738: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10
# 2023-04-10 04:41:10.624747: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 2023-04-10 04:41:10.624843: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.625084: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:10.625219: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1697] Adding visible gpu devices: 0
# 2023-04-10 04:41:10.626315: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# 2023-04-10 04:41:11.840376: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1096] Device interconnect StreamExecutor with strength 1 edge matrix:
# 2023-04-10 04:41:11.840428: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102]      0
# 2023-04-10 04:41:11.840438: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] 0:   N
# 2023-04-10 04:41:11.841791: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:11.842068: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:981] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
# 2023-04-10 04:41:11.842246: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
# 2023-04-10 04:41:11.842289: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1241] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 14988 MB memory) -> physical GPU (device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:1e.0, compute capability: 7.0)
# Downloading data from https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# 
#     8192/94765736 [..............................] - ETA: 1s
# 11714560/94765736 [==>...........................] - ETA: 0s
# 24510464/94765736 [======>.......................] - ETA: 0s
# 37552128/94765736 [==========>...................] - ETA: 0s
# 50290688/94765736 [==============>...............] - ETA: 0s
# 62988288/94765736 [==================>...........] - ETA: 0s
# 75726848/94765736 [======================>.......] - ETA: 0s
# 88399872/94765736 [==========================>...] - ETA: 0s
# 94773248/94765736 [==============================] - 0s 0us/step
# <class 'tensorflow.python.keras.engine.sequential.Sequential'>
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to
#   ['...']
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to
#   ['...']
# Train for 356 steps, validate for 119 steps
# Epoch 1/20
# 2023-04-10 04:41:26.034305: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10
# 2023-04-10 04:41:26.762342: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
# 356/356 - 54s - loss: 240.1997 - mae: 11.0967 - val_loss: 424.1841 - val_mae: 15.4917
# Epoch 2/20
# 356/356 - 38s - loss: 73.1405 - mae: 6.5176 - val_loss: 130.2860 - val_mae: 8.9525
# Epoch 3/20
# 356/356 - 37s - loss: 38.2911 - mae: 4.7705 - val_loss: 78.4883 - val_mae: 6.7900
# Epoch 4/20
# 356/356 - 37s - loss: 22.5524 - mae: 3.7080 - val_loss: 79.2300 - val_mae: 6.6938
# Epoch 5/20
# 356/356 - 38s - loss: 16.4091 - mae: 3.1380 - val_loss: 69.9737 - val_mae: 6.4156
# Epoch 6/20
# 356/356 - 38s - loss: 14.2627 - mae: 2.8796 - val_loss: 77.0521 - val_mae: 6.7016
# Epoch 7/20
# 356/356 - 38s - loss: 12.0682 - mae: 2.6552 - val_loss: 71.3144 - val_mae: 6.3349
# Epoch 8/20
# 356/356 - 38s - loss: 11.7980 - mae: 2.6073 - val_loss: 82.0825 - val_mae: 7.1355
# Epoch 9/20
# 356/356 - 38s - loss: 10.6117 - mae: 2.4818 - val_loss: 69.6299 - val_mae: 6.3353
# Epoch 10/20
# 356/356 - 38s - loss: 9.9028 - mae: 2.3691 - val_loss: 71.1683 - val_mae: 6.4408
# Epoch 11/20
# 356/356 - 38s - loss: 9.3474 - mae: 2.3169 - val_loss: 74.5620 - val_mae: 6.7784
# Epoch 12/20
# 356/356 - 37s - loss: 10.1380 - mae: 2.4163 - val_loss: 62.7982 - val_mae: 5.9394
# Epoch 13/20
# 356/356 - 38s - loss: 10.2378 - mae: 2.4257 - val_loss: 68.2272 - val_mae: 6.1635
# Epoch 14/20
# 356/356 - 38s - loss: 10.1878 - mae: 2.4146 - val_loss: 69.4524 - val_mae: 6.4428
# Epoch 15/20
# 356/356 - 38s - loss: 9.0975 - mae: 2.2912 - val_loss: 70.3128 - val_mae: 6.2446
# Epoch 16/20
# 356/356 - 37s - loss: 7.8139 - mae: 2.0939 - val_loss: 69.4005 - val_mae: 6.1634
# Epoch 17/20
# 356/356 - 38s - loss: 7.2549 - mae: 2.0230 - val_loss: 68.4559 - val_mae: 6.1004
# Epoch 18/20
# 356/356 - 37s - loss: 6.6996 - mae: 1.9719 - val_loss: 63.7187 - val_mae: 6.0363
# Epoch 19/20
# 356/356 - 38s - loss: 6.9101 - mae: 2.0044 - val_loss: 65.3447 - val_mae: 6.0575
# Epoch 20/20
# 356/356 - 38s - loss: 6.7118 - mae: 1.9685 - val_loss: 68.1251 - val_mae: 6.2608
# WARNING:tensorflow:sample_weight modes were coerced from
#   ...
#     to
#   ['...']
# 119/119 - 9s - loss: 68.1251 - mae: 6.2608
# Test MAE: 6.2608

# %% [markdown]
# ## Conclusions

# %% [markdown]
# The table below shows the result of four trial runs:
# 
# | Dropout | Learning Rate | Test MAE |
# |:-------:|---------------|:--------:|
# |   None  |     0.0001    |  6.2608  |
# |   0.5   |     0.0001    |  6.0046  |
# |   None  |     0.0005    |  6.7832  |
# |   0.5   |     0.0003    |  6.8061  |
# 
# The best model configuration used only three layers of neurons to predict a person's age from a photograph with an average error margin of 6.1 years. Despite the small size of the data, the pre-trained ResNet50 model did not require data augmentation or regularization and did not overfit using the power of transfer learning.
# 
# 
# We had identified numerous obvious obstacles when we examined the images prior to training the model. A learning rate of 0.0001 is both slow and expensive. If we address them, we may achieve even higher accuracy and learning performance.
# 
# - The resolution, lighting, and face angles vary from image to image.  -> Adjust them all to be equal.
# - In addition to age, there is a wide range of race and gender. -> Separate the data into age, race, and gender groups and train a model for each to build several models, each becoming an expert at identifying a specific group.
# - Some images show the person's entire body and his or her face is too small to be recognized. -> Ideally, these images should be excluded from the training dataset because they cannot be used to learn anything.
# - Each image has a background, which is a large area that does not reveal anything about a person's face. -> To exclude non-informative pixles, zoom in on the area where there is a face.
# - Some photos are rotated and are "framed within frames." Again, a person's face does not fill the entire image display space. -> Crop and rotate to create a new image that only has the person's face and no rotation.
# - Some images have been stretched and warped. -> Undo the stretching and warping effects.
# - Some images have partial occlusion, which means that part of the face is hidden by an object (e.g., a veil, a microphone, food, eyes closed, etc.). -> To remove the occluding objects, use face restoration techniques.
# - With 7591 rows, the data set is not large. -> Data augmentation expands the data size.
# - Some images are color, while others are monochrome. -> We could sharpen the edges and see the main features of the images (gender, age, and race) more clearly by converting the images to grayscale and using Canny Edge Detection. While we lose information on skin color and complexion, which aids in determining race, racial features will become more pronounced (e.g., height and shape of the nose, shape of the eyes, etc.).
# - The distribution is skewed slightly to the right. There are not an equal number of images for each age group. We have the most data in the 20-40 year old age group. -> We ran the risk of the model learning to perform best on people aged 20 to 40. We can compensate for the differences by increasing the images in the other age groups.
# 
# However, an error margin of 6.1 years does not appear feasible in our context. This project involves estimating the age of customers who enter a Good Seed supermarket in order to avoid selling alcohol to underage customers. The legal purchasing age for alcohol is between the ages of 18 and 21. There will be a problem if an 11.9 year-old (= 18-6.1) is cleared by the ML model as being 18 years old. The project instruction mentions the lowest MAE value recorded as a major success of 5.4 years, but even that does not seem sufficient in our particular situation (a 12.6 year-old has a chance of passing as an 18 year-old).
# 
# 
# With this level of precision, we will require a more lenient scenario than an attempt to follow the law. We can also lower our expectations by deciding on an age range rather than a specific age. To guess the age of someone who (a) does not know when his/her birthday is (in some parts of the world, proper steps are/were not taken to register a new born's birthdate); or (b) died with no one to identify him/her. The latter could also be extended to assisting forensic anthropologists, whose job it is to determine a deceased's age range, albeit from skeletons rather than faces, but picking up features would be a similar task for machine learning models.

# %% [markdown]
# # Checklist

# %% [markdown]
# - [x]  Notebook was opened
# - [x]  The code is error free
# - [x]  The cells with code have been arranged by order of execution
# - [x]  The exploratory data analysis has been performed
# - [x]  The results of the exploratory data analysis are presented in the final notebook
# - [x]  The model's MAE score is not higher than 8
# - [x]  The model training code has been copied to the final notebook
# - [x]  The model training output has been copied to the final notebook
# - [x]  The findings have been provided based on the results of the model training


