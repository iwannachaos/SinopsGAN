from __future__ import print_function, division
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
import numpy as np

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
#from tensorflow.keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import sys
import os


class GAN():
    def __init__(self, filename):
        self.line_len = 0
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.FINAL_TEXT_LENGTH = 20
        self.predict_text = []
        #self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_shape = (self.FINAL_TEXT_LENGTH, 1)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)
        self.prepare_text_data(filename)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # Подготовка текстовых данных: удаление функциональных символов в тексте
    def prepare_text_data(self, string):
        fin = open(string, 'rb')
        self.lines = []

        for line in fin:
            line = line.strip().lower()
            line = line.decode("ascii","ignore")
            if len(line)==0:
                continue
            self.lines.append(line)
        fin.close()
        text = " ".join(self.lines)
        specs = "',]\"{?-+_;$*[&)~|=/#}%<:>`!.()1234567890"
       
        for sc in specs:
            text = text.replace(sc, "")
        
   # Подготовка текстовых данных: создание словаря слов уникальных слов
        words = text.split(' ')
    
        keywords = set([w for w in words])

        nb_words = len(keywords)
        self.words2index = dict((w,i) for i, w in enumerate(keywords))
        self.index2words = dict((i,w) for i, w in enumerate(keywords))
        self.words2index[""] = -1
        self.index2words[-1] = ""
        # self.words2index[""] = len(self.words2index)
        # self.index2words[len(self.index2words)] = ""
   # Подготовка текстовых данных: удаление функциональных символов в массивах строк
        self.lineswords = []
    
        self.clearlines = []
        for l in self.lines:
            for sc in specs:
                l = l.replace(sc, "")
            self.clearlines.append(l)
        
        for l in self.clearlines:
            ls = l.split(' ')
            self.lineswords.append(ls)
            
        #print(self.lineswords)
        #print(self.clearlines)
        
    # Подготовка текстовых данных: удаление функциональных символов в массивах строк    
        self.lineswordsdigit = self.lineswords.copy()
        self.linesDigit = self.clearlines.copy()
    # Подготовка текстовых данных: создание числовых массивов-слов   
#         print(self.clearlines[547])
#         print(self.lines[547])
        for l in range (len(self.clearlines)):
            for w in range(len(self.lineswords[l])):
#                 print((l, w))
                #                 if(self.lineswords[l][w] == 'years,'):
                #                     print('lineswords[l][w]');

                wdigit = self.words2index[self.lineswords[l][w]]
                self.lineswordsdigit[l][w] = wdigit
        
    # Подготовка текстовых данных: заполнение массивов чисел до максимальной длины строки
        # maxl = 0
        EMPTY_LINE_SYMBOL = -1
        # for l in range(len(self.lineswordsdigit)):
        #     if len(self.lineswordsdigit[l])>maxl:
        #         maxl = len(self.lineswordsdigit[l])
        
        # print("------------------------------------------------------------------------------------" + str(maxl))
        # self.line_len = maxl       
        # self.img_shape = (self.line_len, 1)
        # for l in range(len(self.lineswordsdigit)):
        #     while len(self.lineswordsdigit[l])<maxl:
        #         self.lineswordsdigit[l].append(EMPTY_LINE_SYMBOL)

        self.text_cuts = []
        for l in range(len(self.lineswordsdigit)):
            if len(self.lineswordsdigit[l])>=self.FINAL_TEXT_LENGTH:
                 for i in range (len(self.lineswordsdigit[l]) - self.FINAL_TEXT_LENGTH):
                    self.text_cuts.append(self.lineswordsdigit[l][i:i+self.FINAL_TEXT_LENGTH])
            else:
                while len(self.lineswordsdigit[l])<self.FINAL_TEXT_LENGTH:
                    self.lineswordsdigit[l].append(EMPTY_LINE_SYMBOL)
                self.text_cuts.append(self.lineswordsdigit[l])

                    
        
  # Создание блока-генератора

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    # Создание блока-дешифратора

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

   
    def train(self, epochs, batch_size=128, sample_interval=1):

        # Load the dataset

        X_train = np.array(self.text_cuts).astype('int32') # linesworddigit ???

        # Rescale -1 to 1
        X_train = ((X_train + 1) / (len(self.words2index)+1)-0.5)*2
        X_train = np.expand_dims(X_train, axis=2)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.uniform(-1, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
#             if epoch % sample_interval == 0:
#                 self.sample_text(epoch)
    


    def sample_text(self, filename):
        l = 10
        noise = np.random.uniform (-1, 1, (l, self.latent_dim))
        gen_txt = self.generator.predict(noise)

        gen_txt = (gen_txt+1)*len(self.words2index)*0.5-1 
        gen_txt=np.rint(gen_txt)
        gen_txt=np.array(gen_txt).astype('int32')
        
        t_text = gen_txt.copy()

        new_text = t_text.copy()
        #   np.reshape(t_text, (len(t_text), len(t_text[0]),1)) # 10, 3000, 1 -> 1 ^ 10
        new_text_2d = t_text.reshape([new_text.shape[0], new_text.shape[1]])
        
        self.predict_text.append("")

        for wi in range (len(new_text_2d)):
            for ei in range (len(new_text_2d[wi])):
                w = self.index2words[new_text_2d[wi][ei]]
                # self.predict_text[wi].append(w)                
                self.predict_text[wi] += " " + w
            self.predict_text.append("")


        file = open("generics_texts/"+filename, "w")
        for l_2 in self.predict_text:
            file.write(str(l_2) + '\n')
        file.close()


models = []
def mass_text_data():
    for root, dirs, files in os.walk("Texts by genres"):
        for f in files:
            models.append(GAN("Texts by genres/"+f))
            models[len(models)-1].train(2000)
            models[len(models)-1].sample_text("Texts by genres/"+f)



mass_text_data()        

