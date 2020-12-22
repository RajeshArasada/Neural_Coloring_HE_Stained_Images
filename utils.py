# Visualization

from matplotlib import pyplot as plt
#%matplotlib inline
import seaborn as sns

import os
from tqdm import tqdm
import zipfile


# Numpy
import numpy as np
from numpy import asarray
from numpy import save
from numpy import load

# Scikit Learn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Tensorflow 2.0
import tensorflow as tf
print(f"Tensorflow Version: {tf.__version__}")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Flatten, Dense, Dropout 
from tensorflow.keras.layers import Conv2DTranspose, Reshape, concatenate, RepeatVector
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# Scikit Learn image processing
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave




def load_pretrained_vgg19_embedding(pretrained_weights):
    
    # Construct VGG19 model without the classifer and weights trained on imagenet data
    # '''Takes (224, 224, 3) RGB and returns the embeddings(predicions) generated on the RGB image'''
    feature_extractor = VGG19(input_shape=(224, 224, 3),
                              include_top = False)
    
    x = feature_extractor.output
    flat = Flatten()(x)
    fc_1 = Dense(1024, activation='relu')(flat)
    do_1 = Dropout(0.2)(fc_1)
    fc_2 = Dense(512, activation='relu')(do_1)
    do_2 = Dropout(0.3)(fc_2)
    output = Dense(9, activation= 'softmax')(do_2)

    embed_model = Model(feature_extractor.inputs, output)
    # Compile model
    embed_model.compile(loss="categorical_crossentropy",
                        optimizer=tf.keras.optimizers.Adam(lr=0.0004),
                        metrics=["accuracy"])
    
    print("Model Compiled")
    
    embed_model.load_weights(pretrained_weights)
    print("Loaded Finetuned Weights")
    
    return embed_model


def block(x, n_convs, filters, strides, activation, block_name):
    '''
    This functions defines a convlution layer(s) in the encoder network.
    Couples a BatchNormalization layer after a convolution layer
    '''
    for i in range(n_convs):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3,3), 
                                   strides = strides,activation=activation, 
                                   padding='same', name=f"{block_name}_conv{i+1}")(x)
        
        x = tf.keras.layers.BatchNormalization(name=f"{block_name}_batch_normalization{i+1}")(x)
    return x


def Colorization_Model():
    
    
    embed_input = Input(shape=(9,), name="pretrained_model_output") # Prediction from the classifier
    visible = Input(shape=(224,224,1), name="input_image")          # Input image - L channel from 
                                                                    # Lab color space
    
    # Encoder
    ## Low Level Features Network
    
    x = block(visible, 1, 64, (2,2), 'relu', 'LL1')
    x = block(x, 1, 128, (1,1), 'relu', 'LL2')
    x = block(x, 1, 128, (2,2), 'relu', 'LL3')
    x = block(x, 1, 256, (1,1), 'relu', 'LL4')
    x = block(x, 1, 256, (2,2), 'relu', 'LL5')
    x = block(x, 1, 512, (1,1), 'relu', 'LL6')
    
    ## Mid-Level Features Network
    # The shape of mid-level network output is a (256, 28, 28) or 784 pixels for each filter
    # This layer shape determines the number of copies of embedding vectors to generate
    
    x = block(x, 1, 512, (1,1), 'relu', 'ML1')
    x = block(x, 1, 256, (1,1), 'relu', 'ML2')
    
    
    ## Bridge Network
    # bridge connecting the mid level features with the VGG19 classification output
    # Helps the encoder-decoder model to learn features of each class for better colorization
    
    fusion_output = RepeatVector(28*28)(embed_input) # generates 784 copies of the classifier embedding (9,)
    fusion_output = Reshape(([28, 28, 9]))(fusion_output) # Reshape 784 to (28,28,9)
    fusion_output = tf.keras.layers.concatenate([x, fusion_output], axis=-1) # Attach this (28,28,9) to (28,28,256)
    fusion_output = block(fusion_output, 1, 256, (1,1), 'relu', 'FuL')
    
    ## Decoder Network

    decoder_output = block(fusion_output, 1, 128, (1,1), 'relu', 'DL1')
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = block(decoder_output, 1, 64, (1,1), 'relu', 'DL2')
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = block(decoder_output, 1, 32, (1,1), 'relu', 'DL3')
    decoder_output = block(decoder_output, 1, 16, (1,1), 'relu', 'DL4')
    decoder_output = block(decoder_output, 1,  2, (1,1), 'tanh', 'DL5')
    decoder_output = UpSampling2D((2, 2), name='colorization')(decoder_output)
    
    # Define inputs, outputs and compile the model
    
    model = Model(inputs = [visible, embed_input], outputs = [decoder_output])
    optimizer = tf.keras.optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=optimizer, 
                loss = 'mse',
                metrics = tf.keras.metrics.RootMeanSquaredError())
    
    return model

def create_pretrained_embedding(embed_model, rgb):
    """
    This function uses the pretrained CRC classifier and 
    returns the embedding thta predicts the tissue class 
    for a given RGB image
    """
    return embed_model.predict(rgb)

def getImages(filelist, transform_size=(224, 224, 3)):
    """Reads image filelist from DATASET and returns float represtation of RGB [0.0, 1.0]"""
    img_list = []
    for filename in tqdm(filelist):
        # Loads JPEG image and converts it to numpy float array.
        image_in = img_to_array(load_img(filename))
        
        # [0.0, 255.0] => [0.0, 1.0]
        image_in = image_in/255
        
        if transform_size is not None:
            image_in = resize(image_in, transform_size, mode='reflect')

        img_list.append(image_in)
    img_list = np.array(img_list)
    
    return img_list

def preprocess_images(rgb, embed_model, input_size=(224,224,3), embed_size=(224,224,3)):
    # Resize for embed and convert to grayscale
    gray = gray2rgb(rgb2gray(rgb))
    gray = batch_apply(gray, resize, embed_size, mode='constant')
    # Zero-Center [-1,1]
    gray = gray * 2 - 1
    # generate embeddings
    embed = create_pretrained_embedding(embed_model, gray)

    # Resize to input size of model
    re_batch = batch_apply(rgb, resize, input_size, mode='constant')
    # RGB => L*a*b*
    re_batch = batch_apply(re_batch, rgb2lab)

    # Extract L* into X, zero-center and normalize
    X_batch = re_batch[:,:,:,0]
    X_batch = X_batch/50 - 1
    X_batch = X_batch.reshape(X_batch.shape+(1,))

    # Extract a*b* into Y and normalize. Already zero-centered.
    Y_batch = re_batch[:,:,:,1:]
    Y_batch = Y_batch/128

    return [X_batch, embed], Y_batch


def image_a_b_gen(images, batch_size, embed_model):
    for batch in datagen.flow(images, batch_size=batch_size):
        yield preprocess_images(batch, embed_model)
        

def train(model, embed_model, training_files, batch_size=20, epochs=500, steps_per_epoch=50):
    """Trains the model"""
    training_set = getImages(training_files)
    train_size = int(len(training_set)*0.85)
    train_images = training_set[:train_size]
    val_images = training_set[train_size:]
    val_steps = (len(val_images)//batch_size)
    print("Training samples:", train_size, "Validation samples:", len(val_images))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, verbose=1, min_delta=1e-5),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, cooldown=0, verbose=1, min_lr=1e-8),
        ModelCheckpoint(monitor='val_loss', filepath='model/colorize.hdf5', verbose=1,
                         save_best_only=True, save_weights_only=True, mode='auto'),
        # TensorBoard(log_dir='./logs', histogram_freq=10, batch_size=20, write_graph=True, write_grads=True,
        #             write_images=False, embeddings_freq=0)
    ]

    model.fit_generator(image_a_b_gen(train_images, batch_size, embed_model), epochs=epochs, steps_per_epoch=steps_per_epoch,
                        verbose=1, callbacks=callbacks, validation_data=preprocess_images(val_images, embed_model))


        
def batch_apply(ndarray, func, *args, **kwargs):
    """Calls func with samples, func should take ndarray as first positional argument"""

    batch = []
    for sample in ndarray:
        batch.append(func(sample, *args, **kwargs))
    return np.array(batch)

datagen = ImageDataGenerator(shear_range=0.2,
                             zoom_range=0.2,
                             rotation_range=20,
                             horizontal_flip=True)