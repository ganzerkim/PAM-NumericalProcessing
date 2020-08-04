# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:13:46 2019

@author: MIT-DGMIF
"""

import  os,shutil
import cv2
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from keras.models import Input, Model
from keras.layers import BatchNormalization, LeakyReLU, Conv2D, Dense, \
                         Flatten, Add, PReLU, Conv2DTranspose, Lambda, UpSampling2D                    
from keras.optimizers import Adam
from keras.applications import VGG19
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

import keras.backend as K

import efficientnet.keras as efn

save_path = 'D:\\H&E_dataset\\data\\003_Diffracted_data\\GAN_result'
#os.chdir('C:\\Users\\MG\\Desktop\\test_images\\SRGAN_result') #저장할 공간
train_path = 'D:\\H&E_dataset\\data\\003_Diffracted_data\\132um' #학습할 이미지

if not(os.path.exists(save_path)):
    os.mkdir(save_path)
'''    
def crop_center(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[starty:starty + cropy, startx:startx + cropx, :]

#@title Resizing the image
img_list, img_low_list = [], []
for i in os.listdir(train_path):
  if '.png' in i:
    img = cv2.imread(i)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img_crop = crop_center(img, 100,100)
    #img_list.append(img_crop)
    img_low_list.append(cv2.resize(img, (50, 50)))
'''

y_img_path = os.path.join('D:\\H&E_dataset\\data\\003_Diffracted_data\\000um')
y_img = [cv2.cvtColor(cv2.imread(y_img_path + '\\' + s, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for s in os.listdir(y_img_path)]
high_reso_imgs = np.array(y_img)
#low_reso_imgs = np.array(img_low_list)
x_img_path = os.path.join('D:\\H&E_dataset\\data\\003_Diffracted_data\\132um(50)')
x_img = [cv2.cvtColor(cv2.imread(x_img_path + '\\' + s, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for s in os.listdir(x_img_path)]
low_reso_imgs = np.array(x_img)  

os.chdir(save_path) #저장할 공간
plt.figure(figsize = (15,15))
plt.subplot(1,2,1)
plt.imshow(high_reso_imgs[3])
plt.grid('off')
plt.axis('off')
plt.title('High Resolution')
plt.subplot(1,2,2)
plt.imshow(cv2.resize(low_reso_imgs[3],(100,100),
                      interpolation = cv2.INTER_CUBIC))
plt.grid('off')
plt.axis('off')
_=plt.title('low Resolution X2 (bicubic)')
###############################################################################
###############################################################################
class SRGAN():
  # Implementation of SRGAN from paper:
  # https://arxiv.org/abs/1609.04802
    def __init__(self,lr_height = 50,lr_width = 50,channels = 3,
              upscale_factor = 2, generator_lr = 1e-4, discriminator_lr = 1e-4, gan_lr = 1e-4):
        self.height_low_reso = lr_height
        self.width_low_reso = lr_width

        if upscale_factor % 2 != 0:
            raise ValueError('Upscale factor is invalid, must be product of 2')

        self.upscale_factor = upscale_factor
        self.height_high_reso = self.height_low_reso * self.upscale_factor
        self.width_high_reso = self.width_low_reso * self.upscale_factor

        self.channels = channels
        self.shape_low_reso = (self.height_low_reso,self.width_low_reso,self.channels)
        self.shape_high_reso = (self.height_high_reso,self.width_high_reso,self.channels)

        self.samples = high_reso_imgs.shape[0]

        opti_generator = Adam(generator_lr,0.9)
        opti_discriminator = Adam(discriminator_lr,0.9)
        opti_gan = Adam(gan_lr,0.9) 

        self.eff = self.bulid_eff()

        self.discriminator = self.build_discriminator(opti_discriminator)
        self.discriminator.trainable = False
        self.generator = self.build_generator(opti_generator)
        self.srgan = self.build_srgan(opti_gan)

    def save_GAN_Model(self,epoch):
        self.srgan.save_weights('D:\\H&E_dataset\\data\\003_Diffracted_data\\GAN_result\\srgan_weights_epoch_%d.h5' % epoch)


    def plotLosses(self,dlosses,glosses,epo):
        fig, ax1 = plt.subplots(figsize = (10,12))
        color = 'tab:blue'
        ax1.plot(dlosses,color = color, label = 'Dis loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Dis loss', color = color)
        ax1.tick_params('y',color = color)
        color = 'tab:green'
        ax2 = ax1.twinx()
        ax2.plot(glosses, color = color, label = 'Gen loss')
        ax2.set_ylabel('Gen loss', color = color)
        ax2.tick_params('y', color = color)
        plt.title('Discriminator & Generator Losses')
        plt.savefig('Losses_%d.png' % epo)
        #plt.show()

    def gen_pipeline(self, batch_size = 16):
        while(1):
            indx_high = np.random.randint(0,high_reso_imgs.shape[0]-1,batch_size)
            
            indx_low = np.random.randint(0,low_reso_imgs.shape[0]-1,batch_size)
            
            real = np.ones((batch_size,) + self.discriminator.output_shape[1:])
            
            fake = np.zeros((batch_size,) + self.discriminator.output_shape[1:])
            
            norm_hr = high_reso_imgs[indx_high]/127.5-1
            norm_lr = low_reso_imgs[indx_low]/127.5 -1
            yield(norm_hr,real,norm_lr,fake)
            
    def vgg_pipeline(self, batch_size = 16):
      while(1):
        indx = np.random.randint(0,high_reso_imgs.shape[0]-1,batch_size)
        real = np.ones((batch_size,) + self.discriminator.output_shape[1:])
        norm_hr = high_reso_imgs[indx]/127.5-1
        norm_lr = low_reso_imgs[indx]/127.5 -1
        yield(norm_hr,norm_lr,real)
            
      
    def bulid_vgg(self):
        vgg = VGG19(weights = "imagenet")
#         vgg.summary()
        vgg.outputs = [vgg.layers[9].output]
        img = Input(shape = self.shape_high_reso)
        img_features = vgg(img)
        vgg_model = Model(img, img_features)
#         for layer in vgg_model.layers:
#             layer.trainable = False
        vgg_model.compile(loss = 'mse', optimizer = Adam(0.0002,0.5),
                         metrics =['acc'])
        return vgg_model



    def eff_pipeline(self, batch_size = 16):
      while(1):
        indx = np.random.randint(0,high_reso_imgs.shape[0]-1,batch_size)
        real = np.ones((batch_size,) + self.discriminator.output_shape[1:])
        norm_hr = high_reso_imgs[indx]/127.5-1
        norm_lr = low_reso_imgs[indx]/127.5 -1
        yield(norm_hr,norm_lr,real)

    def bulid_eff(self):
      
        eff = efn.EfficientNetB7(weights = "imagenet")
#         vgg.summary()
        eff.outputs = [eff.output]
        img = Input(shape = self.shape_high_reso)
        img_features = eff(img)
        eff_model = Model(img, img_features)
#         for layer in vgg_model.layers:
#             layer.trainable = False
        eff_model.compile(loss = 'mse', optimizer = Adam(0.0002,0.5),
                         metrics =['acc'])
        return eff_model


    def residual_block(self,input_layer):
        x = Conv2D(filters = 64, kernel_size = 3, padding = 'same')(input_layer)
        x = BatchNormalization(momentum=0.8)(x)
        x = PReLU()(x)
        x = Conv2D(filters = 64, kernel_size = 3, padding = 'same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        return Add()([input_layer,x])

    def disc_block(self,layer, n_filters, batch_norm = True):
        x = Conv2D(filters = n_filters, kernel_size = 3, padding = 'same')(layer)
        if batch_norm:
            x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters = n_filters, kernel_size = 3,
                   strides=2, padding = 'same')(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def Upsample_Block(self,x_in):
        x = Conv2D(filters = 256, kernel_size=3, padding='same')(x_in)
        #x = self.SubpixelConv2D(2)(x)
        x = UpSampling2D(size = (2, 2), interpolation = 'bilinear')(x)
        return PReLU()(x)
      
    def SubpixelConv2D(self,scale):
        return Lambda(lambda x: tf.depth_to_space(x, scale))
    
    
  
    def build_generator(self,opti_generator,n_blocks = 16):
        input_layer = Input(self.shape_low_reso)
        
        first_layer = Conv2D(filters = 64, kernel_size = 9,
                             padding = 'same')(input_layer)
        
        first_layer = PReLU()(first_layer)
        
        residual_blocks = self.residual_block(first_layer)
        
        for _ in range(n_blocks-1):
            residual_blocks = self.residual_block(residual_blocks)

        output_residual = Conv2D(filters = 64, kernel_size = 3,
                             padding = 'same')(residual_blocks)
        
        output_residual = BatchNormalization(momentum=0.8)(output_residual)
        
        output_residual = Add()([output_residual,first_layer])
               
        upsample_layer = self.Upsample_Block(output_residual)
        
        for _ in range(self.upscale_factor//2-1):
            upsample_layer =  self.Upsample_Block(upsample_layer)
            
        gen_output = Conv2D(filters = 3, kernel_size = 9,
                            padding = 'same', activation = 'tanh')(upsample_layer)

        gen_model = Model(inputs = input_layer, outputs = gen_output)
        gen_model.compile(loss = 'binary_crossentropy', optimizer = opti_generator)

        return gen_model

    def build_discriminator(self,opti_discriminator,n_blocks = 3, n_filters = 64):
        input_layer = Input(self.shape_high_reso)
        discriminator_blocks = self.disc_block(input_layer,n_filters,False)
        for i in range(n_blocks):
            discriminator_blocks = self.disc_block(discriminator_blocks, 
                                             n_filters = (i+1)*2*n_filters)
        
        #f_layer = GlobalAveragePooling2D()(discriminator_blocks)
        f_layer = Dense(units = 1024)(discriminator_blocks)
        f_layer = LeakyReLU(alpha=0.2)(f_layer)
        dis_output = Dense(units = 3, activation = 'sigmoid')(f_layer)
        disc_model = Model(inputs = input_layer, outputs = dis_output)
        disc_model.compile(loss = 'mse', optimizer = opti_discriminator,
                          metrics = ['accuracy'])

        return disc_model

    def build_srgan(self,optimizer):
        dis_input = Input(self.shape_high_reso)
        gen_input = Input(self.shape_low_reso)

        generated_high_reso = self.generator(gen_input)
        generated_features = self.eff(generated_high_reso)
        generator_valid = self.discriminator(generated_high_reso)


        gan_model = Model(inputs = [gen_input, dis_input], 
                          outputs = [generator_valid, generated_features])
        
        for l in gan_model.layers[-1].layers[-1].layers:
          l.trainable=False
        
        gan_model.compile(loss = ['binary_crossentropy','mse'], loss_weights = [1e-2,1], optimizer = 'adam')
        gan_model.summary()
        
        return gan_model



    def train(self, epochs, save_interval = 100, batch_size = 16):
        pipeline = self.gen_pipeline(batch_size)
        eff_pipeline = self.eff_pipeline(batch_size)

        batch_count = self.samples // batch_size
        dlosses = []
        glosses = []
        for epo in range(1,epochs+1):
            print ('-'*15,'Epoch %d' % epo, '-'*15)
            for _ in tqdm(range(batch_count)):

                ##########################
                # Train the Discriminator
                ##########################
                # Generate Batch
                hr_imgs, real, lr_imgs, fake = next(pipeline)

                # Generate high resolution photos from low resolution photos
                generated_hr_imags = self.generator.predict(lr_imgs)

                # Train the discriminator 
                real_dis_loss = self.discriminator.train_on_batch(hr_imgs,real)
                fake_dis_loss = self.discriminator.train_on_batch(generated_hr_imags,fake)
                dis_loss = (0.5*np.add(real_dis_loss,fake_dis_loss))

                ##########################
                # Train the Generator
                ##########################

                #Generate Batch
                hr_imgs, lr_imgs, real = next(eff_pipeline)

                # Extract ground truth using VGG model
                img_features = self.eff.predict(hr_imgs)
                gan_loss = self.srgan.train_on_batch([lr_imgs,hr_imgs], [real, img_features])

            if epo % save_interval == 0:
              self.save_GAN_Model(epo)
              self.plotLosses(dlosses,glosses,epo)
            dlosses.append(gan_loss[1])
            glosses.append(gan_loss[0])
            print('\n',dlosses[-1],glosses[-1])
            
#%%
from sewar.full_ref import mse, psnr, ssim
            
def plot_predict(low_reso_imgs,high_reso_imgs,srgan_model,idx, n_imgs):
    plt.figure(figsize = (12,12))
    plt.tight_layout()
    n_imgs = n_imgs
    for i in range(0,n_imgs*3,3):
        #idx = np.random.randint(0,low_reso_imgs.shape[0]-1)
        idx = idx
        plt.subplot(n_imgs,3,i+1)
        
        h_img = high_reso_imgs[idx]
        
        plt.imshow(h_img)
        plt.grid('off')
        plt.axis('off')
        #plt.title('Source')
        plt.title('Original')
        plt.subplot(n_imgs,3,i+2)
        plt.imshow(cv2.resize(low_reso_imgs[idx],(256,256), interpolation = cv2.INTER_LINEAR))
        plt.grid('off')
        plt.axis('off')
        #plt.title('X4 (bicubic)')
        plt.title('low resolution (bicubic)')
        img = srgan_model.generator.predict(np.expand_dims(low_reso_imgs[idx], axis = 0) / 127.5 - 1)
        img_unnorm = (img+1) * 127.5
        
        plt.subplot(n_imgs,3,i+3)
        plt.imshow(np.squeeze(img_unnorm, axis = 0).astype(np.uint8))
        
        prd_val_psnr = psnr(h_img, img_unnorm[0])
        prd_val_ssim = ssim(h_img, img_unnorm[0])
        
        plt.title('SRGAN_result')
        plt.ylabel('PSNR/SSIM')
        plt.xlabel(str(prd_val_psnr) + '/' + str(prd_val_ssim))
        
        img_rgb = cv2.cvtColor(np.squeeze(img_unnorm, axis = 0).astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imwrite('predict_img'+ str(idx) + '.png', img_rgb)
        #plt.grid('off')
        #plt.axis('off')
        
    plt.savefig('predicted_' + str(idx) + '.png')

#%%           
#Training start            
model_srgan = SRGAN()   
           
model_srgan.generator.summary()
model_srgan.discriminator.summary()
            
process_path = 'D:\\H&E_dataset\\data\\003_Diffracted_data\\GAN_result\\training_process'
if not(os.path.exists(process_path)):
    os.mkdir(process_path)

os.chdir('D:\\H&E_dataset\\data\\003_Diffracted_data\\GAN_result\\training_process')

model_srgan.train(100, save_interval=10 ,batch_size=16)          
#%%           
os.chdir('D:\\H&E_dataset\\data\\003_Diffracted_data\\GAN_result')
model_srgan.srgan.load_weights('srgan_weights_epoch_240.h5')
plot_predict(low_reso_imgs,high_reso_imgs,model_srgan, 6, 1)

#%%
shutil.move('C:\\Users\\MG\\Desktop\\H&E New dataset\\SuperResolutionData\\H&E_imageset\\20200213_SRGAN\\srgan_weights_epoch_160.h5', os.getcwd())    
model_srgan.srgan.load_weights('srgan_weights_epoch_240.h5')       
model_srgan.srgan.load_weights('srgan_weights_epoch_300.h5')
plot_predict(low_reso_imgs,high_reso_imgs,model_srgan, 6, 1)



            
            
            
            
            
            
            
            