### 개고양이 이미지 분류

# 함수, 모듈 준비
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from glob import glob

import cv2, os, random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


##########################################################
### seed값 설정
# seed 값은 random 함수에서 랜덤 값을 계산할 때 사용하며 매 번 바뀝니다.
# 초기 seed 값을 설정하지 않으면 랜덤 값을 생성하는 순서가 매 번 달라집니다.
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

##########################################################
### dataset 경로 및 사이즈
ROW, COL = 96, 96
path = 'dataset/dogs-vs-cats2/'

##########################################################
### 데이터 불러오기(dog_train)
# os - 환경 변수나 디렉토리, 파일 등의 OS자원을 제어할 수 있게 해주는 모듈
# 함수	                설명
# os.mkdir(디렉터리)	    디렉터리를 생성한다.
# os.rmdir(디렉터리)	    디렉터리를 삭제한다.단, 디렉터리가 비어있어야 삭제가 가능하다.
# os.unlink(파일)	    파일을 지운다.
# os.rename(src, dst)	src라는 이름의 파일을 dst라는 이름으로 바꾼다.

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
dog_path = os.path.join(path, 'training_set/dogs/dog.*')
len(glob(dog_path))

dogs = []
for dog_image in glob(dog_path):
    dog = cv2.imread(dog_image)                           # 이미지 데이터를 읽기
    dog = cv2.cvtColor(dog, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    dog = cv2.resize(dog, (ROW, COL))                     # 가로,세로 사이즈 설정
    dog = image.img_to_array(dog)                         # 이미지를 array로 변환
    dogs.append(dog)
len(dogs)

img = image.array_to_img(random.choice(dogs))
plt.imshow(img, cmap=plt.get_cmap('gray'))
##########################################################
# 데이터 불러오기(cat_train)
cat_path = os.path.join(path, 'training_set/cats/cat.*')
len(glob(cat_path))

cats = []
for cat_image in glob(cat_path):
    cat = cv2.imread(cat_image)
    cat = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)
    cat = cv2.resize(cat, (ROW, COL))
    cat = image.img_to_array(cat)
    cats.append(cat)
len(cats)

##########################################################
# enumerate - 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환
# dog=0, cat=1
y_dog, y_cat = [], []

y_dog = [0 for item in enumerate(dogs)]
y_cat = [1 for item in enumerate(cats)]

len(y_dog)
len(y_cat)


##########################################################
# 리스트의 형태를 ndarray로 바꿔줌
dogs = np.asarray(dogs).astype('float32')
cats = np.asarray(cats).astype('float32')

y_dog = np.asarray(y_dog).astype('int32')
y_cat = np.asarray(y_cat).astype('int32')

##########################################################
# 표준화를 하면 학습 속도가 더 빨라지고, 지역 최적의 상태에 빠지게 될 가능성을 줄여준다
# local optima 요즘 trend에 의하면, 중요한 문제가 아니다(실제 딥러닝에서 local optima에 빠질 확률이 거의 없음)
# values값을 0과 1 사이로 맞춰줌
dogs /= 255                                                # 이미지는 숫자로 0~255의 8비트 부호없는 정수로 저장
cats /= 255

dogs.shape
##########################################################
# concatenate 함수를 이용해서 배열 결합
X = np.concatenate((dogs, cats), axis=0)
Y = np.concatenate((y_dog, y_cat), axis=0)

len(X)
len(Y)

##########################################################
# 학습셋, 테스트셋 구분
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size = 0.2,    # 테스트 셋 크기 설정
                                                    random_state = 0)   # 난수 발생 시드

len(X_train)
len(X_test)
len(Y_train)
len(Y_test)

##########################################################
# Local response normalization(LRN)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Lambda
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Layer

class LocalResponseNormalization(Layer):

    def __init__(self, n=5, alpha=1e-4, beta=0.75, k=2, **kwargs):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.k = k
        super(LocalResponseNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.shape = input_shape
        super(LocalResponseNormalization, self).build(input_shape)

    def call(self, x):
        _, r, c, f = self.shape
        squared = K.square(x)
        pooled = K.pool2d(squared, (self.n, self.n), strides=(1, 1), padding="same", pool_mode='avg')
        summed = K.sum(pooled, axis=3, keepdims=True)
        averaged = self.alpha * K.repeat_elements(summed, f, axis=3)
        denom = K.pow(self.k + averaged, self.beta)
        return x / denom

    def compute_output_shape(self, input_shape):
        return input_shape

##########################################################
# 모델 설계(alexnet)
input_shape = (96, 96, 1)

model = Sequential()
model.add(Conv2D(96, (11, 11), strides=4, padding='same', input_shape=input_shape))

model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
model.add(LocalResponseNormalization(input_shape=model.output_shape[1:]))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(LocalResponseNormalization(input_shape=model.output_shape[1:]))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))



optimizer = 'adam'
model.compile(loss = 'binary_crossentropy',
              optimizer = optimizer,
              metrics = ['accuracy'])
model.summary()


## 모델 저장 폴더 설정
Model_dir = 'ImageForCNN/model1/'
if not os.path.exists(Model_dir):
    os.mkdir(Model_dir)

## 모델 저장 조건 설정
modelpath = 'ImageForCNN/model1/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,                   # 해당 함수의 진행사항을 출력
                               save_best_only = True)         # 성능이 좋아졌을때만 기록

## 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 10)         # 성능이 좋아 질때까지 10 epoch 기다리기


## 모델 실행
history = model.fit(X_train, Y_train,
                    validation_data = (X_test, Y_test),
                    epochs = 30,
                    batch_size = 200,
                    verbose = 1,
                    callbacks = [early_stopping_callback, checkpointer])


## 테스트 정확도 출력
print('\n Test Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))

# 테스트셋의 오착
y_vloss = history.history['val_loss']

# 학습셋의 오착
y_loss = history.history['loss']

# loss 그래프 그리기
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = '.', c = 'red', label = 'Testset_loss')
plt.plot(x_len, y_loss, marker = '.', c = 'blue', label = 'Trainset_loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()