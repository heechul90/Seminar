### 전력량 예측하기

### 함수
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

### seed 설정
# 랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
# 하이퍼파라미터 튜닝하기 위한 용도(흔들리면 무엇때문에 좋아졌는지 알기 어려움)
tf.set_random_seed(777)

### Standardization (정규분포를 따르는 데이터의 표준정규분포로의 표준화: 평균과 표준편차 이용)
# 각 observation이 평균을 기준으로 어느 정도 떨어져 있는지를 나타낼때 사용한다
# 서로 다른 통계 데이터들을 비교하기 용이하게 하기 위해
# 어떤 변수를 어떤 표본에 대해 통계를 구하였는가에 따라 평균과 분산값은 제각각이기 때문에, 서로 비교하기가 불편하다
# 표준화를 하면 평균은 0, 분산과 표준편차는 1이 되므로 비교하기가 용이하다
# 각 값에서 평균을 빼고, 표준편차로 나눠줌
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x.np.std()

# a = [80, 90, 100, 110, 120]
# a = pd.DataFrame(a)
# a.mean()
# a.std()
# print(20 / a.std())
# print(10 / a.std())
# print(0 / a.std())
# print(-10 / a.std())
# print(-20 / a.std())


### Min-Max soaling (이상치/특이값이 포함되어 있는 데이터의 표준화: 중앙값과 IQR(InterQuarile Range) 이용)
# 사이킷런에서 sklearn.preprocessing.MinMaxScaler(), sklearn.preprocessing.minmax_scale()로 제공
# StandardScaler(X): 평균이 0과 표준편차가 1이 되도록 변환.
# RobustScaler(X)  : 중앙값(median)이 0, IQR(interquartile range)이 1이 되도록 변환.
# MinMaxScaler(X)  : 최대값이 각각 1, 최소값이 0이 되도록 변환
# MaxAbsScaler(X)  : 0을 기준으로 절대값이 가장 큰 수가 1또는 -1이 되도록 변환
# 너무 작거나 너무 큰 값이 학습을 방해하는 것을 방지하고자 정규화한다.
# x가 양수라는 가정하에 최소값과 최대값을 이용하여 0-1사이의 값으로 변환
# (값 - 최소값) / (최대값 - 최소값)

def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7) # 1e-7 0으로 나누는 오류 예방차원

######
# from sklearn.preprocessing import MinMaxScaler
# a = [80, 90, 100, 110, 120]
# a = pd.DataFrame(a)
# a.min()
# a.max()
# min_max_scaler = MinMaxScaler()
# trainA = min_max_scaler.fit_transform(a)
#
# from sklearn.preprocessing import minmax_scale
# a_MinMax_scaled = minmax_scale(a, axis=0, copy=True)
# a_MinMax_scaled


### 정규화된 값을 원래의 값으로 되돌린다
# 정규화하기 이전의 org_x값과 되돌리고 싶은 x를 입력하면 역정규화된 값을 리턴한다
def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()


### 하이퍼파라미터
input_data_column_cnt = 2     # 입력데이터의 컬럼 개수
output_data_column_cnt = 1    # 결과데이터의 컬럼 개수

seq_length = 60               # 1개 시퀀스의 길이)시계열데이터 입력 개수)
rnn_cell_hidden_dim = 20      # 각 셀의 (hidden)출력 크기
forget_bias = 1.0             # 망각편향 (기본값 1.0) - 셀 스테이트에서 어떤 정보를 버릴지 선택하는 과정
num_stacked_layers = 1        # stacked LSTM layers 개수
keep_prob = 1.0               # dropout 할 때 keep 할 비율

epoch_num = 1000              # 에폭 횟수
learning_rate = 0.01          # 학습률

# 데이터 불러오기
raw_data = pd.read_csv('dataset/통신구/water_level.csv',
                       encoding = 'euc-kr')
raw_data.info()
# data_des = raw_data.describe()
# data_des.to_csv('Data/정보.csv',
#                 encoding = 'euc-kr')

raw_data['time'] = raw_data['time'].astype(str)
raw_data['time'] = pd.to_datetime(raw_data['time'])
raw_data.set_index('time', inplace = True)
# del raw_data['date']
raw_data.value.plot()
raw_data.label.plot()

#
# import seaborn as sns
# sns.heatmap(data = raw_data.corr(), annot=True, fmt = '.2f', linewidths = 0.5)


## 데이터를 array로 바꿈
water = raw_data.values.astype(np.float)
print('watt_info.shape: ', water.shape)
print('watt_info[0]: ', water[0])


### 데이터 전처리

# water value 컬럼 정규화
water_value = water[:, :-1]
norm_water_value = min_max_scaling(water_value)  # 전력량 데이터 정규화 처리
print('water_value.shape: ', norm_water_value.shape)
print('water_value[0]: ', norm_water_value[0])
print('norm_water_value.shape: ', norm_water_value.shape)
print('norm_water_value[0]: ', norm_water_value[0])
print('-'*100)   # 화면상 구분용

# from sklearn.preprocessing import MinMaxScaler
# min_max_scaler = MinMaxScaler()
# trainA = min_max_scaler.fit_transform(weather)
# trainA.min()
# trainA.max()

# water level 컬럼 정규화
water_level = water[:, -1:]
# elec_norm = min_max_scaling(elec)
# print('elec.shape: ', elec.shape)
# print('elec[0]: ', elec[0])
# print('elec_norm.shape: ', elec_norm.shape)
# print('elec_norm[0]: ', elec_norm[0])
# print('-'*100)   # 화면상 구분용

# from sklearn.preprocessing import MinMaxScaler
# min_max_scaler = MinMaxScaler()
# trainB = min_max_scaler.fit_transform(elec)
# trainB.min()
# trainB.max()

# 배열 결합
x = np.concatenate((norm_water_value, water_level), axis = 1)
print('x.shape: ', x.shape)
print('x[0]: ', x[0])
print('x[-1]: ', x[-1])
print('-'*100)   # 화면상 구분용

# AB = np.concatenate((trainA, trainB), axis = 1)
# print('AB.shape: ', AB.shape)
# print('AB[0]: ', AB[0])
# print('AB[-1]: ', AB[-1])
# print('-'*100)   # 화면상 구분용


### 왜 날씨데이터와 전력량데이터 정규화를 따로하는지 의문이다
### 하지만 따로 정규화를 하고 결합을 했을 때와 값이랑 통채로 정규화를 했을 때의 값은 달랐다
### sklearn.preprocessing을 이용하면 정규화할때 각 컬럼별로 정규화를 하지만
### 위에서 만든 min_max_scaling은 전체 데이터에서 최소값, 최대값으로 계산을 하기 때문에 값이 달라진다
# a = min_max_scaling(watt)
# a.min()
# a.max()
# print('a.shape: ', a.shape)
# print('a[0]: ', a[0])
# print('a[-1]: ', a[-1])

# from sklearn.preprocessing import MinMaxScaler
# min_max_scaler = MinMaxScaler()
# c = min_max_scaler.fit_transform(watt)
# c.min()
# c.max()

# d = watt[:, 0:1]
# from sklearn.preprocessing import MinMaxScaler
# min_max_scaler = MinMaxScaler()
# d = min_max_scaler.fit_transform(d)
# d.min()
# d.max()




y = x[:, [-1]]       # 타겟은 전력량
print('y[0]: ', y[0])
print('y[-1]: ', y[-1])


dataX = []  # 입력으로 사용될 Sequence Data
dataY = []  # 출력(타켓)으로 사용

for i in range(0, len(y) - (seq_length)):          # 10741
    _x = x[i: i + seq_length]                      # [0 : 0 + 60]
    _y = y[i + seq_length]                         # [0 + 60] 다음에 나타날 전력량(정답)
    if i is 0:
        print(_x, "->", _y)        # 첫번째 행만 출력해 봄
    dataX.append(_x)               # dataX 리스트에 추가
    dataY.append(_y)               # dataY 리스트에 추가

len(dataX)
### 학습용/테스트용 데이터 생성
# 전체 70%를 학습용 데이터로 사용
train_size = int(len(dataY) * 0.7)

# 나머지(30%)를 테스트용 데이터로 사용
test_size = len(dataY) - train_size

# 데이터를 잘라 학습용 데이터 생성
trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

# 데이터를 잘라 테스트용 데이터 생성
testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])

########################################################################################################################

### 텐서플로우 플레이스홀더 생성
# tf.placeholder(dtype, [shape], name)
# dtype : 데이터 타입을 의미하며 반드시 적어주어야 한다.
# shape : 입력 데이터의 형태를 의미한다. 상수 값이 될 수도 있고 다차원 배열의 정보가 들어올 수도 있다. ( 디폴트 파라미터로 None 지정 )
# name  : 해당 placeholder의 이름을 부여하는 것으로 적지 않아도 된다.  ( 디폴트 파라미터로 None 지정 )
# 입력 X, 출력 Y를 생성한다
X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
print("X: ", X)
Y = tf.placeholder(tf.float32, [None, 1])
print("Y: ", Y)

# 검증용 측정지표를 산출하기 위한 targets, predictions를 생성한다
targets = tf.placeholder(tf.float32, [None, 1])
print("targets: ", targets)

predictions = tf.placeholder(tf.float32, [None, 1])
print("predictions: ", predictions)


# 모델(LSTM 네트워크) 생성
def lstm_cell():
    # LSTM셀을 생성
    # num_units: 각 Cell 출력 크기
    # forget_bias:  셀 스테이트에서 어떤 정보를 버릴지 선택하는 과정
    #               이 결정은 'forget gate laye'라고 불리는 시그모이드 레이어로 만들어짐
    #               이 게이트에서는 0과 1의 입력값을 받는다(0: 완전히 이 값을 버려라, 1: 완전히 이 값을 유지해라)
    # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
    # state_is_tuple: False ==> they are concatenated along the column axis.
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_cell_hidden_dim,
                                        forget_bias=forget_bias, state_is_tuple=True, activation=tf.nn.softsign)
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


# Multi Layer Perceptron RNN: layer 연결
# 입력층과 출력층 사이에 하나 이상의 중간층(hidden layer: 은닉층)이 존재하는 신경망
# 네트워크는 입력층, 은닉층, 출력층 방향으로 연결
# 각 층내의 연결과 출력층에서 입력층으로의 직접적인 연결은 존재하지 않은 전방향 (Feedforward) 네트워크
# num_stacked_layers개의 층으로 쌓인 Stacked RNNs 생성
stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

# RNN Cell(여기서는 LSTM셀임)들을 연결
hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("hypothesis: ", hypothesis)

# [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
# 과거 여러일수의 전력량을 이용해서 다음날의 전력량 1개를 에측하기때문에 many-to-one 형태이다
# one to one: Vanilla Neural Networks
# one to many: Image Captioning - 한장의 이미지에 대해 여러개의 문장으로 해석하는 형태, '소년이 사과를 고르고 있다'
# many to one: Sentiment Classification - 여러개의 문장으로 구성된 글을 해석하여, 감정상태를 나타내는 형태, '긍정', '부정'
# many to many: Machine Translation - 여러개의 문장에서 각각의 문장들을 다른 언어로 해석해주는 형태, 'Hello' -> '안녕'
# many to many: Video classification - 여러개의 이미지에 대해 여러개의 설병, 번역을 하는 형태

hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.identity)

# 손실함수로 평균제곱오차를 사용한다
loss = tf.reduce_sum(tf.square(hypothesis - Y))

# 최적화함수로 AdamOptimizer를 사용한다
# optimizer = tf.train.RMSPropOptimizer(learning_rate) # LSTM과 궁합 별로임
optimizer = tf.train.AdamOptimizer(learning_rate)

train = optimizer.minimize(loss)

# RMSE(Root Mean Square Error)
# 제곱오차의 평균을 구하고 다시 제곱근을 구하면 평균 오차가 나온다
# rmse = tf.sqrt(tf.reduce_mean(tf.square(targets-predictions))) # 아래 코드와 같다
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

train_error_summary = []  # 학습용 데이터의 오류를 중간 중간 기록한다
test_error_summary = []  # 테스트용 데이터의 오류를 중간 중간 기록한다
test_predict = ''  # 테스트용데이터로 예측한 결과

### 세션 정의
# 세션 생성: Session 객체 생성. 분산 환경에서는 계산 노드와의 연결을 만든다.
# 세션 사용: run 메서드에 그래프를 입력하면 출력 값을 계산하여 반환한다. 분산 환경에서는 계산 노드로 그래프를 보내 계산을 수행한다.
# 세션 종료: close 메서드. with 문을 사용하면 명시적으로 호출하지 않아도 된다.
sess = tf.Session()
sess.run(tf.global_variables_initializer())


def build_accuracy(predictions, labels_):
    """
    Create accuracy
    """
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)

    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

# # 학습
start_time = datetime.datetime.now()  # 시작시간을 기록한다
print('학습을 시작합니다...')

for epoch in range(epoch_num):
    _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    if ((epoch + 1) % 100 == 0) or (epoch == epoch_num - 1):  # 100번째마다 또는 마지막 epoch인 경우
        # 학습용데이터로 rmse오차를 구한다
        train_predict = sess.run(hypothesis, feed_dict={X: trainX})
        train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
        train_error_summary.append(train_error)

        # 테스트용데이터로 rmse오차를 구한다
        test_predict = sess.run(hypothesis, feed_dict={X: testX})
        test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
        test_error_summary.append(test_error)

        # 현재 오류를 출력한다
        print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch + 1,
                                                                                 train_error,
                                                                                 test_error,
                                                                                 test_error - train_error))



print(build_accuracy(test_predict, testY))

# 하이퍼파라미터 출력
print('input_data_column_cnt:', input_data_column_cnt, end='')
print(',output_data_column_cnt:', output_data_column_cnt, end='')

print(',seq_length:', seq_length, end='')
print(',rnn_cell_hidden_dim:', rnn_cell_hidden_dim, end='')
print(',forget_bias:', forget_bias, end='')
print(',num_stacked_layers:', num_stacked_layers, end='')
print(',keep_prob:', keep_prob, end='')

print(',epoch_num:', epoch_num, end='')
print(',learning_rate:', learning_rate, end='')

print(',train_error:', train_error_summary[-1], end='')
print(',test_error:', test_error_summary[-1], end='')
print(',min_test_error:', np.min(test_error_summary))

# 결과 그래프 출력
plt.figure(1)
plt.plot(train_error_summary, 'red')
plt.plot(test_error_summary, 'blue')
plt.xlabel('Epoch(x1000)')
plt.ylabel('Root Mean Square Error')

plt.figure(2)
plt.plot(testY, 'red')
plt.plot(test_predict, 'blue')
plt.xlabel('Time Period')
plt.ylabel('elec')
plt.show()




# sequence length만큼의 가장 최근 데이터를 슬라이싱한다
recent_data = np.array([x[len(x) - seq_length-1:-1]])
print("recent_data.shape: ", recent_data.shape)
print("recent_data: ", recent_data)

# 내일 전력량을 예측해본다
test_predict = sess.run(hypothesis, feed_dict={X: recent_data})

print("test_predict: ", test_predict[0])
# test_predict = reverse_min_max_scaling(water_level, test_predict)  # 전력량 데이터를 다시 역정규화
# print("Tomorrow's elec price: ", test_predict[0])             # 내일 전력량 데이터

# test_predict:  [0.5484068]
# Tomorrow's elec price:  [788970.2]

# test_predict [0.5380935]
# Tomorrow's elec price [787583.9]

