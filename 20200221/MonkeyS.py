### Monkey Species

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
# Label,  Latin Name             , Common Name                   , Train Images , Validation Images
# n0   , alouatta_palliata	     , mantled_howler                , 131          , 26
# n1   , erythrocebus_patas	     , patas_monkey                  , 139          , 28
# n2   , cacajao_calvus	         , bald_uakari                   , 137          , 27
# n3   , macaca_fuscata	         , japanese_macaque              , 152          , 30
# n4   , cebuella_pygmea	     , pygmy_marmoset                , 131          , 26
# n5   , cebus_capucinus	     , white_headed_capuchin         , 141          , 28
# n6   , mico_argentatus	     , silvery_marmoset              , 132          , 26
# n7   , saimiri_sciureus	     , common_squirrel_monkey        , 142          , 28
# n8   , aotus_nigriceps	     , black_headed_night_monkey     , 133          , 27
# n9   , trachypithecus_johnii , nilgiri_langur                , 132          , 26
##########################################################

##########################################################
### seed값 설정
# seed 값은 randdom 함수에서 랜덤 값을 계산할 때 사용하며 매 번 바뀝니다.
# 초기 seed 값을 설정하지 않으면 랜덤 값을 생성하는 순서가 매 번 달라집니다.
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

##########################################################
### dataset 경로 및 사이즈
ROW, COL = 96, 96
path = 'dataset/MonkeySpecies/training/'

##########################################################   학습셋
### 데이터 불러오기(mantled_howler)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
train_mantled_howler_path = os.path.join(path, 'n0/*.jpg')
len(glob(train_mantled_howler_path))

train_mantled_howler = []
for mantled_howler_image in glob(train_mantled_howler_path):
    mantled_howler = cv2.imread(mantled_howler_image)                           # 이미지 데이터를 읽기
    mantled_howler = cv2.cvtColor(mantled_howler, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    mantled_howler = cv2.resize(mantled_howler, (ROW, COL))                     # 가로,세로 사이즈 설정
    mantled_howler = image.img_to_array(mantled_howler)                         # 이미지를 array로 변환
    train_mantled_howler.append(mantled_howler)
len(train_mantled_howler)

##########################################################
### 데이터 불러오기(patas_monkey)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
train_patas_monkey_path = os.path.join(path, 'n1/*.jpg')
len(glob(train_patas_monkey_path))

train_patas_monkey = []
for patas_monkey_image in glob(train_patas_monkey_path):
    patas_monkey = cv2.imread(patas_monkey_image)                           # 이미지 데이터를 읽기
    patas_monkey = cv2.cvtColor(patas_monkey, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    patas_monkey = cv2.resize(patas_monkey, (ROW, COL))                     # 가로,세로 사이즈 설정
    patas_monkey = image.img_to_array(patas_monkey)                         # 이미지를 array로 변환
    train_patas_monkey.append(patas_monkey)
len(train_patas_monkey)

##########################################################
### 데이터 불러오기(uakari_image)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
train_bald_uakari_path = os.path.join(path, 'n2/*.jpg')
len(glob(train_bald_uakari_path))

train_bald_uakari = []
for bald_uakari_image in glob(train_bald_uakari_path):
    uakari_image = cv2.imread(bald_uakari_image)                           # 이미지 데이터를 읽기
    uakari_image = cv2.cvtColor(uakari_image, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    uakari_image = cv2.resize(uakari_image, (ROW, COL))                     # 가로,세로 사이즈 설정
    uakari_image = image.img_to_array(uakari_image)                         # 이미지를 array로 변환
    train_bald_uakari.append(uakari_image)
len(train_bald_uakari)

##########################################################
### 데이터 불러오기(japanese_macaque)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
train_japanese_macaque_path = os.path.join(path, 'n3/*.jpg')
len(glob(train_japanese_macaque_path))

train_japanese_macaque = []
for japanese_macaque_image in glob(train_japanese_macaque_path):
    japanese_macaque = cv2.imread(japanese_macaque_image)                           # 이미지 데이터를 읽기
    japanese_macaque = cv2.cvtColor(japanese_macaque, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    japanese_macaque = cv2.resize(japanese_macaque, (ROW, COL))                     # 가로,세로 사이즈 설정
    japanese_macaque = image.img_to_array(japanese_macaque)                         # 이미지를 array로 변환
    train_japanese_macaque.append(japanese_macaque)
len(train_japanese_macaque)

##########################################################
### 데이터 불러오기(pygmy_marmoset)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
train_pygmy_marmoset_path = os.path.join(path, 'n4/*.jpg')
len(glob(train_pygmy_marmoset_path))

train_pygmy_marmoset = []
for pygmy_marmoset_image in glob(train_pygmy_marmoset_path):
    pygmy_marmoset = cv2.imread(pygmy_marmoset_image)                           # 이미지 데이터를 읽기
    pygmy_marmoset = cv2.cvtColor(pygmy_marmoset, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    pygmy_marmoset = cv2.resize(pygmy_marmoset, (ROW, COL))                     # 가로,세로 사이즈 설정
    pygmy_marmoset = image.img_to_array(pygmy_marmoset)                         # 이미지를 array로 변환
    train_pygmy_marmoset.append(pygmy_marmoset)
len(train_pygmy_marmoset)

##########################################################
### 데이터 불러오기(white_headed_capuchin)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
train_white_headed_capuchin_path = os.path.join(path, 'n5/*.jpg')
len(glob(train_white_headed_capuchin_path))

train_white_headed_capuchin = []
for white_headed_capuchin_image in glob(train_white_headed_capuchin_path):
    white_headed_capuchin = cv2.imread(white_headed_capuchin_image)                           # 이미지 데이터를 읽기
    white_headed_capuchin = cv2.cvtColor(white_headed_capuchin, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    white_headed_capuchin = cv2.resize(white_headed_capuchin, (ROW, COL))                     # 가로,세로 사이즈 설정
    white_headed_capuchin = image.img_to_array(white_headed_capuchin)                         # 이미지를 array로 변환
    train_white_headed_capuchin.append(white_headed_capuchin)
len(train_white_headed_capuchin)

##########################################################
### 데이터 불러오기(silvery_marmoset)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
train_silvery_marmoset_path = os.path.join(path, 'n6/*.jpg')
len(glob(train_silvery_marmoset_path))

train_silvery_marmoset = []
for silvery_marmoset_image in glob(train_silvery_marmoset_path):
    silvery_marmoset = cv2.imread(silvery_marmoset_image)                           # 이미지 데이터를 읽기
    silvery_marmoset = cv2.cvtColor(silvery_marmoset, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    silvery_marmoset = cv2.resize(silvery_marmoset, (ROW, COL))                     # 가로,세로 사이즈 설정
    silvery_marmoset = image.img_to_array(silvery_marmoset)                         # 이미지를 array로 변환
    train_silvery_marmoset.append(silvery_marmoset)
len(train_silvery_marmoset)

##########################################################
### 데이터 불러오기(common_squirrel_monkey)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
train_common_squirrel_monkey_path = os.path.join(path, 'n7/*.jpg')
len(glob(train_common_squirrel_monkey_path))

train_common_squirrel_monkey = []
for common_squirrel_monkey_image in glob(train_common_squirrel_monkey_path):
    common_squirrel_monkey = cv2.imread(common_squirrel_monkey_image)                           # 이미지 데이터를 읽기
    common_squirrel_monkey = cv2.cvtColor(common_squirrel_monkey, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    common_squirrel_monkey = cv2.resize(common_squirrel_monkey, (ROW, COL))                     # 가로,세로 사이즈 설정
    common_squirrel_monkey = image.img_to_array(common_squirrel_monkey)                         # 이미지를 array로 변환
    train_common_squirrel_monkey.append(common_squirrel_monkey)
len(train_common_squirrel_monkey)

##########################################################
### 데이터 불러오기(black_headed_night_monkey)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
train_black_headed_night_monkey_path = os.path.join(path, 'n8/*.jpg')
len(glob(train_black_headed_night_monkey_path))

train_black_headed_night_monkey = []
for black_headed_night_monkey_image in glob(train_black_headed_night_monkey_path):
    black_headed_night_monkey = cv2.imread(black_headed_night_monkey_image)                           # 이미지 데이터를 읽기
    black_headed_night_monkey = cv2.cvtColor(black_headed_night_monkey, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    black_headed_night_monkey = cv2.resize(black_headed_night_monkey, (ROW, COL))                     # 가로,세로 사이즈 설정
    black_headed_night_monkey = image.img_to_array(black_headed_night_monkey)                         # 이미지를 array로 변환
    train_black_headed_night_monkey.append(black_headed_night_monkey)
len(train_black_headed_night_monkey)

##########################################################
### 데이터 불러오기(nilgiri_langur)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
train_nilgiri_langur_path = os.path.join(path, 'n9/*')
len(glob(train_nilgiri_langur_path))

train_nilgiri_langur = []
for nilgiri_langur_image in glob(train_nilgiri_langur_path):
    nilgiri_langur = cv2.imread(nilgiri_langur_image)                           # 이미지 데이터를 읽기
    nilgiri_langur = cv2.cvtColor(nilgiri_langur, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    nilgiri_langur = cv2.resize(nilgiri_langur, (ROW, COL))                     # 가로,세로 사이즈 설정
    nilgiri_langur = image.img_to_array(nilgiri_langur)                         # 이미지를 array로 변환
    train_nilgiri_langur.append(nilgiri_langur)
len(train_nilgiri_langur)
##########################################################

##########################################################
# enumerate - 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환
#
y_train_mantled_howler, y_train_patas_monkey, y_train_bald_uakari, y_train_japanese_macaque, y_train_pygmy_marmoset = [], [], [], [], []
y_train_white_headed_capuchin, y_train_silvery_marmoset, y_train_common_squirrel_monkey, y_train_black_headed_night_monkey, y_train_nilgiri_langur = [], [], [], [], []

y_train_mantled_howler            = [0 for item in enumerate(train_mantled_howler)]
y_train_patas_monkey              = [1 for item in enumerate(train_patas_monkey)]
y_train_bald_uakari               = [2 for item in enumerate(train_bald_uakari)]
y_train_japanese_macaque          = [3 for item in enumerate(train_japanese_macaque)]
y_train_pygmy_marmoset            = [4 for item in enumerate(train_pygmy_marmoset)]
y_train_white_headed_capuchin     = [5 for item in enumerate(train_white_headed_capuchin)]
y_train_silvery_marmoset          = [6 for item in enumerate(train_silvery_marmoset)]
y_train_common_squirrel_monkey    = [7 for item in enumerate(train_common_squirrel_monkey)]
y_train_black_headed_night_monkey = [8 for item in enumerate(train_black_headed_night_monkey)]
y_train_nilgiri_langur            = [9 for item in enumerate(train_nilgiri_langur)]

##########################################################
# 리스트의 형태를 ndarray로 바꿔줌
train_mantled_howler            = np.asarray(train_mantled_howler).astype('float32')
train_patas_monkey              = np.asarray(train_patas_monkey).astype('float32')
train_bald_uakari               = np.asarray(train_bald_uakari).astype('float32')
train_japanese_macaque          = np.asarray(train_japanese_macaque).astype('float32')
train_pygmy_marmoset            = np.asarray(train_pygmy_marmoset).astype('float32')
train_white_headed_capuchin     = np.asarray(train_white_headed_capuchin).astype('float32')
train_silvery_marmoset          = np.asarray(train_silvery_marmoset).astype('float32')
train_common_squirrel_monkey    = np.asarray(train_common_squirrel_monkey).astype('float32')
train_black_headed_night_monkey = np.asarray(train_black_headed_night_monkey).astype('float32')
train_nilgiri_langur            = np.asarray(train_nilgiri_langur).astype('float32')
len(train_mantled_howler)

y_train_mantled_howler            = np.asarray(y_train_mantled_howler).astype('int32')
y_train_patas_monkey              = np.asarray(y_train_patas_monkey).astype('int32')
y_train_bald_uakari               = np.asarray(y_train_bald_uakari).astype('int32')
y_train_japanese_macaque          = np.asarray(y_train_japanese_macaque).astype('int32')
y_train_pygmy_marmoset            = np.asarray(y_train_pygmy_marmoset).astype('int32')
y_train_white_headed_capuchin     = np.asarray(y_train_white_headed_capuchin).astype('int32')
y_train_silvery_marmoset          = np.asarray(y_train_silvery_marmoset).astype('int32')
y_train_common_squirrel_monkey    = np.asarray(y_train_common_squirrel_monkey).astype('int32')
y_train_black_headed_night_monkey = np.asarray(y_train_black_headed_night_monkey).astype('int32')
y_train_nilgiri_langur            = np.asarray(y_train_nilgiri_langur).astype('int32')



##########################################################
# values값을 0과 1 사이로 맞춰줌
train_mantled_howler /= 255
train_patas_monkey /= 255
train_bald_uakari /= 255
train_japanese_macaque /= 255
train_pygmy_marmoset /= 255
train_white_headed_capuchin /= 255
train_silvery_marmoset /= 255
train_common_squirrel_monkey /= 255
train_black_headed_night_monkey /= 255
train_nilgiri_langur /= 255

##########################################################
# concatenate 함수를 이용해서 배열 결합
X_train = np.concatenate((train_mantled_howler,
                          train_patas_monkey,
                          train_bald_uakari,
                          train_japanese_macaque,
                          train_pygmy_marmoset,
                          train_white_headed_capuchin,
                          train_silvery_marmoset,
                          train_common_squirrel_monkey,
                          train_black_headed_night_monkey,
                          train_nilgiri_langur), axis=0)

y_train = np.concatenate((y_train_mantled_howler,
                          y_train_patas_monkey,
                          y_train_bald_uakari,
                          y_train_japanese_macaque,
                          y_train_pygmy_marmoset,
                          y_train_white_headed_capuchin,
                          y_train_silvery_marmoset,
                          y_train_common_squirrel_monkey,
                          y_train_black_headed_night_monkey,
                          y_train_nilgiri_langur), axis=0)

# One-Hot Encoding
y_train = np_utils.to_categorical(y_train, 10)

len(X_train)
len(y_train)
##########################################################
##########################################################
##########################################################   테스트셋

### dataset 경로 및 사이즈
ROW, COL = 96, 96
path = 'D:/HeechulFromGithub/dataset/MonkeySpecies/validation/'

### 데이터 불러오기(mantled_howler)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
test_mantled_howler_path = os.path.join(path, 'n0/*')
len(glob(test_mantled_howler_path))

test_mantled_howler = []
for mantled_howler_image in glob(test_mantled_howler_path):
    mantled_howler = cv2.imread(mantled_howler_image)                           # 이미지 데이터를 읽기
    mantled_howler = cv2.cvtColor(mantled_howler, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    mantled_howler = cv2.resize(mantled_howler, (ROW, COL))                     # 가로,세로 사이즈 설정
    mantled_howler = image.img_to_array(mantled_howler)                         # 이미지를 array로 변환
    test_mantled_howler.append(mantled_howler)
len(test_mantled_howler)

##########################################################
### 데이터 불러오기(patas_monkey)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
test_patas_monkey_path = os.path.join(path, 'n1/*')
len(glob(test_patas_monkey_path))

test_patas_monkey = []
for patas_monkey_image in glob(test_patas_monkey_path):
    patas_monkey = cv2.imread(patas_monkey_image)                           # 이미지 데이터를 읽기
    patas_monkey = cv2.cvtColor(patas_monkey, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    patas_monkey = cv2.resize(patas_monkey, (ROW, COL))                     # 가로,세로 사이즈 설정
    patas_monkey = image.img_to_array(patas_monkey)                         # 이미지를 array로 변환
    test_patas_monkey.append(patas_monkey)
len(test_patas_monkey)

##########################################################
### 데이터 불러오기(uakari_image)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
test_bald_uakari_path = os.path.join(path, 'n2/*')
len(glob(test_bald_uakari_path))

test_bald_uakari = []
for bald_uakari_image in glob(test_bald_uakari_path):
    uakari_image = cv2.imread(bald_uakari_image)                           # 이미지 데이터를 읽기
    uakari_image = cv2.cvtColor(uakari_image, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    uakari_image = cv2.resize(uakari_image, (ROW, COL))                     # 가로,세로 사이즈 설정
    uakari_image = image.img_to_array(uakari_image)                         # 이미지를 array로 변환
    test_bald_uakari.append(uakari_image)
len(test_bald_uakari)

##########################################################
### 데이터 불러오기(japanese_macaque)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
test_japanese_macaque_path = os.path.join(path, 'n3/*.jpg')
len(glob(test_japanese_macaque_path))

test_japanese_macaque = []
for japanese_macaque_image in glob(test_japanese_macaque_path):
    japanese_macaque = cv2.imread(japanese_macaque_image)                           # 이미지 데이터를 읽기
    japanese_macaque = cv2.cvtColor(japanese_macaque, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    japanese_macaque = cv2.resize(japanese_macaque, (ROW, COL))                     # 가로,세로 사이즈 설정
    japanese_macaque = image.img_to_array(japanese_macaque)                         # 이미지를 array로 변환
    test_japanese_macaque.append(japanese_macaque)
len(test_japanese_macaque)

##########################################################
### 데이터 불러오기(pygmy_marmoset)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
test_pygmy_marmoset_path = os.path.join(path, 'n4/*')
len(glob(test_pygmy_marmoset_path))

test_pygmy_marmoset = []
for pygmy_marmoset_image in glob(test_pygmy_marmoset_path):
    pygmy_marmoset = cv2.imread(pygmy_marmoset_image)                           # 이미지 데이터를 읽기
    pygmy_marmoset = cv2.cvtColor(pygmy_marmoset, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    pygmy_marmoset = cv2.resize(pygmy_marmoset, (ROW, COL))                     # 가로,세로 사이즈 설정
    pygmy_marmoset = image.img_to_array(pygmy_marmoset)                         # 이미지를 array로 변환
    test_pygmy_marmoset.append(pygmy_marmoset)
len(test_pygmy_marmoset)

##########################################################
### 데이터 불러오기(white_headed_capuchin)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
test_white_headed_capuchin_path = os.path.join(path, 'n5/*')
len(glob(test_white_headed_capuchin_path))

test_white_headed_capuchin = []
for white_headed_capuchin_image in glob(test_white_headed_capuchin_path):
    white_headed_capuchin = cv2.imread(white_headed_capuchin_image)                           # 이미지 데이터를 읽기
    white_headed_capuchin = cv2.cvtColor(white_headed_capuchin, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    white_headed_capuchin = cv2.resize(white_headed_capuchin, (ROW, COL))                     # 가로,세로 사이즈 설정
    white_headed_capuchin = image.img_to_array(white_headed_capuchin)                         # 이미지를 array로 변환
    test_white_headed_capuchin.append(white_headed_capuchin)
len(test_white_headed_capuchin)

##########################################################
### 데이터 불러오기(silvery_marmoset)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
test_silvery_marmoset_path = os.path.join(path, 'n6/*')
len(glob(test_silvery_marmoset_path))

test_silvery_marmoset = []
for silvery_marmoset_image in glob(test_silvery_marmoset_path):
    silvery_marmoset = cv2.imread(silvery_marmoset_image)                           # 이미지 데이터를 읽기
    silvery_marmoset = cv2.cvtColor(silvery_marmoset, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    silvery_marmoset = cv2.resize(silvery_marmoset, (ROW, COL))                     # 가로,세로 사이즈 설정
    silvery_marmoset = image.img_to_array(silvery_marmoset)                         # 이미지를 array로 변환
    test_silvery_marmoset.append(silvery_marmoset)
len(test_silvery_marmoset)

##########################################################
### 데이터 불러오기(common_squirrel_monkey)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
test_common_squirrel_monkey_path = os.path.join(path, 'n7/*')
len(glob(test_common_squirrel_monkey_path))

test_common_squirrel_monkey = []
for common_squirrel_monkey_image in glob(test_common_squirrel_monkey_path):
    common_squirrel_monkey = cv2.imread(common_squirrel_monkey_image)                           # 이미지 데이터를 읽기
    common_squirrel_monkey = cv2.cvtColor(common_squirrel_monkey, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    common_squirrel_monkey = cv2.resize(common_squirrel_monkey, (ROW, COL))                     # 가로,세로 사이즈 설정
    common_squirrel_monkey = image.img_to_array(common_squirrel_monkey)                         # 이미지를 array로 변환
    test_common_squirrel_monkey.append(common_squirrel_monkey)
len(test_common_squirrel_monkey)

##########################################################
### 데이터 불러오기(black_headed_night_monkey)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
test_black_headed_night_monkey_path = os.path.join(path, 'n8/*')
len(glob(test_black_headed_night_monkey_path))

test_black_headed_night_monkey = []
for black_headed_night_monkey_image in glob(test_black_headed_night_monkey_path):
    black_headed_night_monkey = cv2.imread(black_headed_night_monkey_image)                           # 이미지 데이터를 읽기
    black_headed_night_monkey = cv2.cvtColor(black_headed_night_monkey, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    black_headed_night_monkey = cv2.resize(black_headed_night_monkey, (ROW, COL))                     # 가로,세로 사이즈 설정
    black_headed_night_monkey = image.img_to_array(black_headed_night_monkey)                         # 이미지를 array로 변환
    test_black_headed_night_monkey.append(black_headed_night_monkey)
len(test_black_headed_night_monkey)

##########################################################
### 데이터 불러오기(nilgiri_langur)

# glob - 디렉토리에 있는 파일들을 리스트로 만들어 주는 모듈
test_nilgiri_langur_path = os.path.join(path, 'n9/*')
len(glob(test_nilgiri_langur_path))

test_nilgiri_langur = []
for nilgiri_langur_image in glob(test_nilgiri_langur_path):
    nilgiri_langur = cv2.imread(nilgiri_langur_image)                           # 이미지 데이터를 읽기
    nilgiri_langur = cv2.cvtColor(nilgiri_langur, cv2.COLOR_BGR2GRAY)           # 흑백사진으로 변환
    nilgiri_langur = cv2.resize(nilgiri_langur, (ROW, COL))                     # 가로,세로 사이즈 설정
    nilgiri_langur = image.img_to_array(nilgiri_langur)                         # 이미지를 array로 변환
    test_nilgiri_langur.append(nilgiri_langur)
len(test_nilgiri_langur)
##########################################################

##########################################################
# enumerate - 인덱스 번호와 컬렉션의 원소를 tuple형태로 반환
#
y_test_mantled_howler, y_test_patas_monkey, y_test_bald_uakari, y_test_japanese_macaque, y_test_pygmy_marmoset = [], [], [], [], []
y_test_white_headed_capuchin, y_test_silvery_marmoset, y_test_common_squirrel_monkey, y_test_black_headed_night_monkey, y_test_nilgiri_langur = [], [], [], [], []

y_test_mantled_howler            = [0 for item in enumerate(test_mantled_howler)]
y_test_patas_monkey              = [1 for item in enumerate(test_patas_monkey)]
y_test_bald_uakari               = [2 for item in enumerate(test_bald_uakari)]
y_test_japanese_macaque          = [3 for item in enumerate(test_japanese_macaque)]
y_test_pygmy_marmoset            = [4 for item in enumerate(test_pygmy_marmoset)]
y_test_white_headed_capuchin     = [5 for item in enumerate(test_white_headed_capuchin)]
y_test_silvery_marmoset          = [6 for item in enumerate(test_silvery_marmoset)]
y_test_common_squirrel_monkey    = [7 for item in enumerate(test_common_squirrel_monkey)]
y_test_black_headed_night_monkey = [8 for item in enumerate(test_black_headed_night_monkey)]
y_test_nilgiri_langur            = [9 for item in enumerate(test_nilgiri_langur)]

##########################################################
# 리스트의 형태를 ndarray로 바꿔줌
test_mantled_howler            = np.asarray(test_mantled_howler).astype('float32')
test_patas_monkey              = np.asarray(test_patas_monkey).astype('float32')
test_bald_uakari               = np.asarray(test_bald_uakari).astype('float32')
test_japanese_macaque          = np.asarray(test_japanese_macaque).astype('float32')
test_pygmy_marmoset            = np.asarray(test_pygmy_marmoset).astype('float32')
test_white_headed_capuchin     = np.asarray(test_white_headed_capuchin).astype('float32')
test_silvery_marmoset          = np.asarray(test_silvery_marmoset).astype('float32')
test_common_squirrel_monkey    = np.asarray(test_common_squirrel_monkey).astype('float32')
test_black_headed_night_monkey = np.asarray(test_black_headed_night_monkey).astype('float32')
test_nilgiri_langur            = np.asarray(test_nilgiri_langur).astype('float32')

y_test_mantled_howler            = np.asarray(y_test_mantled_howler).astype('float32')
y_test_patas_monkey              = np.asarray(y_test_patas_monkey).astype('float32')
y_test_bald_uakari               = np.asarray(y_test_bald_uakari).astype('float32')
y_test_japanese_macaque          = np.asarray(y_test_japanese_macaque).astype('float32')
y_test_pygmy_marmoset            = np.asarray(y_test_pygmy_marmoset).astype('float32')
y_test_white_headed_capuchin     = np.asarray(y_test_white_headed_capuchin).astype('float32')
y_test_silvery_marmoset          = np.asarray(y_test_silvery_marmoset).astype('float32')
y_test_common_squirrel_monkey    = np.asarray(y_test_common_squirrel_monkey).astype('float32')
y_test_black_headed_night_monkey = np.asarray(y_test_black_headed_night_monkey).astype('float32')
y_test_nilgiri_langur            = np.asarray(y_test_nilgiri_langur).astype('float32')

##########################################################
# values값을 0과 1 사이로 맞춰줌
test_mantled_howler /= 255
test_patas_monkey /= 255
test_bald_uakari /= 255
test_japanese_macaque /= 255
test_pygmy_marmoset /= 255
test_white_headed_capuchin /= 255
test_silvery_marmoset /= 255
test_common_squirrel_monkey /= 255
test_black_headed_night_monkey /= 255
test_nilgiri_langur /= 255

##########################################################

# concatenate 함수를 이용해서 배열 결합
X_test = np.concatenate((test_mantled_howler,
                         test_patas_monkey,
                         test_bald_uakari,
                         test_japanese_macaque,
                         test_pygmy_marmoset,
                         test_white_headed_capuchin,
                         test_silvery_marmoset,
                         test_common_squirrel_monkey,
                         test_black_headed_night_monkey,
                         test_nilgiri_langur), axis=0)

y_test = np.concatenate((y_test_mantled_howler,
                         y_test_patas_monkey,
                         y_test_bald_uakari,
                         y_test_japanese_macaque,
                         y_test_pygmy_marmoset,
                         y_test_white_headed_capuchin,
                         y_test_silvery_marmoset,
                         y_test_common_squirrel_monkey,
                         y_test_black_headed_night_monkey,
                         y_test_nilgiri_langur), axis=0)
# One-Hot Encoding
y_test = np_utils.to_categorical(y_test, 10)

len(X_test)
len(y_test)

##########################################################

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

model.add(Dense(10, activation='softmax'))



optimizer = 'adam'
model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizer,
              metrics = ['accuracy'])
model.summary()


## 모델 저장 폴더 설정
Model_dir = 'ImageForCNN/model2/'
if not os.path.exists(Model_dir):
    os.mkdir(Model_dir)

## 모델 저장 조건 설정
modelpath = 'ImageForCNN/model2/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,
                               save_best_only = True)

## 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 3)


## 모델 실행
history = model.fit(X_train, y_train,
                    validation_data = (X_test, y_test),
                    epochs = 10,
                    batch_size = 200,
                    verbose = 1,
                    callbacks = [early_stopping_callback, checkpointer])


## 테스트 정확도 출력
print('\n Test Accuracy: %.4f' % (model.evaluate(X_test, y_test)[1]))

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