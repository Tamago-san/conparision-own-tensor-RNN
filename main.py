"""
node/own/rnn
x - y
 10/6.6254194286819487E-002/0.0644
 50/5.5374950569890453E-002/0.0073
 100/5.0790885290730531E-002/0.0028
x - x+d10
 10/8.2902834389943128E-002/0.0148
 50/
 100/
x - x+d40
 10/0.17848630340696628/0.2123
 50/0.12698598924414800/0.337634631625661
 100/8.1799997049907619E-002/0.261384203642178
 
x - z
 
"""

#gfortran -shared -o rnn_tanh.so rnn_tanh.f90 -llapack -lblas -fPIC
#gfortran
import pandas as pd
import numpy as np
import math
import random
from keras import initializers
from keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras import losses
import ctypes
import datetime
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import zscore
#from sklearn.metrics import mean_squared_error




GOD_step=1 #CSVファイルより固定
inout_node = 1
hidden_node = 100 #学習epoch数
epoch=500
traning_rato=0.8
SAMPLE_step =1000
timesteps= 1
inout_node= 1
target_columns=1
DT    = 0.01
GUSAI = 1.0
CI =  0.
BETA =1.
SIGMA = 1.
OUT_NODE=1
SPECTRAL_RADIUS = 1
ITRMAX=5000000
R0= 1.
GOD_STEP=50
steps_of_history=1
steps_in_future=1
#EPOCH=5000
update=1#SDM=0 ADAM=1
EPSILON=0.01


GOD_step=10
in_out_neurons = 1
hidden_neurons =hidden_node
n_prev = 1000
#epoch=100
test_size=0.2



#def _load_data(data, n_prev,fbatch ):
#    """
#    data should be pd.DataFrame()
#    """
#
#    X, Y = [], []
#    for i in range(0,fbatch*n_prev,n_prev):
#        X.append(data.iloc[i:i+n_prev].as_matrix())
##        Y.append(data.iloc[i+n_prev+GOD_step-1].as_matrix())
#        Y.append(data.iloc[i+GOD_step:i+n_prev+GOD_step].as_matrix())
#    print(np.array(X).shape)
#    reX = np.array(X).reshape(fbatch, n_prev, 1)
#    reY = np.array(Y).reshape(fbatch, n_prev, 1)
#
#    return reX, reY
#
#def train_test_split(df, test_size, n_prev):
#    """
#    This just splits data to training and testing parts
#    """
##    allsize_batch=int((len(df)-GOD_step)/n_prev)
#    train_batch=1
##    train_batch=int(allsize_batch*(1 - test_size))
#    test_batch =5
#    print(len(df))
#    print(allsize_batch)
#    print(train_batch)
#    print(test_batch)
#    X_train, y_train = _load_data(df.iloc[0:int(n_prev*train_batch)+GOD_step], n_prev,train_batch)
#    X_test, y_test = _load_data(df.iloc[int(n_prev*train_batch):int(n_prev*train_batch)+1000], n_prev,test_batch)
#
#    return (X_train, y_train), (X_test, y_test) ,train_batch,test_batch


def _load_data2(data, n_prev,fbatch ):
    """
    data should be pd.DataFrame()
    """
    
    X, Y = [], []
    for i in range(0,fbatch*n_prev,n_prev):
        X.append(data.iloc[i:i+n_prev,0].as_matrix())
#        Y.append(data.iloc[i+n_prev+GOD_step-1].as_matrix())
        Y.append(data.iloc[i:i+n_prev,1].as_matrix())
    print(np.array(X).shape)
    reX = np.array(X).reshape(fbatch, n_prev, 1)
    reY = np.array(Y).reshape(fbatch, n_prev, 1)

    return reX, reY

def train_test_split2(df, test_size, n_prev):
    """
    This just splits data to training and testing parts
    """
    allsize_batch=int(len(df)/n_prev)
    train_batch =5
    test_batch =1
#    train_batch=int(allsize_batch*(1 - test_size))
#    test_batch =allsize_batch-train_batch
    print(len(df))
    print(allsize_batch)
    print(train_batch)
    print(test_batch)
    X_train, y_train = _load_data2(df.iloc[0:int(n_prev*train_batch),:], n_prev,train_batch)
    X_test, y_test = _load_data2(df.iloc[int(n_prev*train_batch):,:], n_prev,test_batch)

    return (X_train, y_train), (X_test, y_test) ,train_batch,test_batch
    
def mean_squared_error(y_true,y_pre):
    loss = 0
    for istep in range(1000):
        loss +=(y_true[istep] - y_pre[istep])**2
    loss = loss/1000
    loss = np.sqrt(loss)
    
    return loss


def create_dataset(df, ntrn):
    """
    This just splits data to training and testing parts
    """
#    ntrn = round(len(df) * (1 - test_size))
#    ntrn = int(ntrn)
    X_train = df.iloc[0:ntrn,0].values.reshape(ntrn, 1)
    y_train = df.iloc[0:ntrn,1].values.reshape(ntrn, 1)
    X_test  = df.iloc[ntrn:,0].values.reshape(len(df)-ntrn, 1)
    y_test  = df.iloc[ntrn:,1].values.reshape(len(df)-ntrn, 1)
    print(X_train.shape)
    print(y_train.shape)
    return X_train, y_train, X_test, y_test

def call_fortran_rnn_traning_own(_in_node,_out_node,_rnn_node,_traning_step,_rnn_step,
                        _sample_num,_epoch,_epsilon,_g,
                        U_in,S_out,U_rc,S_rc,W_out,W_rnn,W_in,_Tre_CH,_update):
    f = np.ctypeslib.load_library("rnn_tanh.so", ".")
    f. rnn_traning_own_fortran_.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        np.ctypeslib.ndpointer(dtype=np.float64),
        ctypes.POINTER(ctypes.c_int32),
        ]
    f. rnn_traning_own_fortran_.restype = ctypes.c_void_p

    f_in_node = ctypes.byref(ctypes.c_int32(_in_node))
    f_out_node = ctypes.byref(ctypes.c_int32(_out_node))
    f_rnn_node = ctypes.byref(ctypes.c_int32(_rnn_node))
    f_traning_step = ctypes.byref(ctypes.c_int32(_traning_step))
    f_rnn_step = ctypes.byref(ctypes.c_int32(_rnn_step))
    f_sample_num = ctypes.byref(ctypes.c_int32(_sample_num))
    f_epoch = ctypes.byref(ctypes.c_int32(_epoch))
    f_epsilon = ctypes.byref(ctypes.c_double(_epsilon))
    f_g = ctypes.byref(ctypes.c_double(_g))
    f_update = ctypes.byref(ctypes.c_int32(_update))
    f.rnn_traning_own_fortran_(f_in_node,f_out_node,f_rnn_node,f_traning_step,f_rnn_step,
                            f_sample_num,f_epoch,f_epsilon,f_g,
                            U_in,S_out,U_rc,S_rc,W_out,W_rnn,W_in,Tre_CH,f_update)

#def my_init(shape,dtype=None):
#    return initializers.RandomNormal(mean=0.0, stddev=1/hidden_neurons**0.5 , seed=None)

#ファイルをpd形式で取ってきて成型
df1 = pd.read_csv('./data/Lorenz_xy.csv',
                    engine='python',
                )


df = pd.read_csv('./data/output_Runge_Lorenz.csv',
#                    usecols=[1],
                    engine='python',
                    header =None
                )

df = df.rename(columns={0: 'X'})
df = df.rename(columns={1: 'Y'})
df = df.rename(columns={2: 'Z'})
#df[["sin_t"]].head(steps_per_cycle * 2).plot()
#(X_train, y_train), (X_test, y_test),train_batch,test_batch = train_test_split(df[["X"]], test_size,n_prev )
(X_train, y_train), (X_test, y_test),train_batch,test_batch = train_test_split2(df[["X","Y"]], test_size,n_prev )
#dataframe=pd.Dataframe(X_train,columns("x"))

length_of_sequence = X_train.shape[1]
in_out_neurons = 1
ibatch_size = 5
#initial_state=keras.initializers.RandomNormal(mean=0.0, stddev=1/hidden_neurons**0.5 , seed=None)
model = Sequential()
#model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequence, in_out_neurons), return_sequences=False))
model.add(SimpleRNN(hidden_neurons,
                    batch_input_shape=(1, length_of_sequence, in_out_neurons),
                    return_sequences=True,
                    kernel_initializer=initializers.random_normal(stddev=1/hidden_neurons**0.5),
                    stateful=True,
                    activation='tanh'
                    ))
#model.add(stateful(True))
#model.add(SimpleRNN(hidden_neurons,batch_input_shape=(None, length_of_sequence, in_out_neurons)
#                    ,return_sequences=True
#                    ,kernel_initializer='random_normal'))
model.add(Dense(in_out_neurons,kernel_initializer='random_normal'))
model.add(Activation("linear"))
#sgd=optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss="mean_squared_error", optimizer="sgd")
#early_stoppingをcallbacksで定義　→　validationの誤差値(val_loss)の変化が収束したと判定された場合に自動で終了
#modeをauto　→　収束の判定を自動で行う．
#patience　→　判定値からpatienceの値の分だけのepoch学習. 変化がなければ終了
#patience=0　→　val_lossが上昇した瞬間終了
#early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=20)
csv_logger=CSVLogger("./data_out/RNN_history.dat", separator='\t')
print(X_test.shape)
hist=model.fit(X_train, y_train,
                    batch_size=1,
                    nb_epoch=epoch,
                    validation_split=0.,
                    verbose=1,
                    shuffle=False,
                    validation_data=(X_test,y_test),
                    callbacks=[csv_logger])
                    

predicted = model.predict(X_test)


#score = model.evaluate(X_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#print(history)

#+++++++++++++++++++++++++++++++++++++++++++++++++++
#===================================================
#===================================================
df1 = df1.rename(columns={0: 't'})
df1 = df1.rename(columns={1: 't+1'})
#################
datalen0=len(df1)
SAMPLE_num=int(traning_rato*datalen0/SAMPLE_step)
traning_step=SAMPLE_step*SAMPLE_num
RNN_STEP = datalen0 - traning_step #トレーニングとRCとを分ける
#RC_STEP = datalen0                #RCの出力にトレーニング時間も含める。
#################
X_train1, y_train1, X_test1, y_test1 = create_dataset(df1, traning_step )


X_train1.reshape(traning_step,-1)
X_test1.reshape(RNN_STEP,-1)
y_rnn = np.empty((RNN_STEP,inout_node))
E=np.eye(hidden_node)
#W_IN= W_IN/(float(IN_NODE))**0.5
W_in = np.empty((hidden_node,inout_node))
W_rnn = np.empty((hidden_node,hidden_node))
W_out = np.empty((inout_node,hidden_node))
Tre_CH = np.empty((epoch))
r_befor = np.zeros((hidden_node))
#S_rc = np.zeros((RNN_STEP,inout_node))
#+++++++++++++++++++++++++++++++++++++++++++++++++++
#===================================================
#===================================================
#U_in = zscore(U_in,axis=0)
#S_out = zscore(S_out,axis=0)

#print(X_train1.shape)
#print(X_test1.shape)

#call_fortran_rnn_traning_own(inout_node,inout_node,hidden_node,SAMPLE_step,RNN_STEP,
#            SAMPLE_num,epoch,EPSILON,G
#            ,X_train1,y_train1,X_test1,y_rnn
#            ,W_out,W_rnn,W_in,Tre_CH,update)
#

#グラフ表示
dataf =  pd.DataFrame(predicted.flatten())
dataf.columns = ["OUTPUT_RNN"]
dataf["OUTPUT_ORI"] = y_test.flatten()
dataf.to_csv("./data_out/Keras.csv")

loss_sk=mean_squared_error(dataf["OUTPUT_RNN"].values,dataf["OUTPUT_ORI"].values)
print(loss_sk)
#plt.figure()
#dataf.plot()
#plt.show()
#loss = pd.read_table('./data_out/RNN_history.dat'
##                    usecols=[1],
##                   engine='python'
#                )
val_loss=np.array(hist.history['val_loss']).reshape(-1,1)
#val_loss = np.sqrt(val_loss)
loss = pd.DataFrame(val_loss)
loss[1] = np.sqrt(2*loss[0])
print(loss)


df201=pd.read_csv("./data_out/lyapnov_end_trstep.1000",
                     usecols=[3],
                     header =None)
print(df201)
df202=pd.read_csv("./data_renban2/rc_out.0500",
                    usecols=[0,1],
                    header =None)
df202.columns = ["OUTPUT_ORI","OUTPUT_RNN"]
print(df202)

#print(dataf)
#plt.figure()
#dataf.plot()
#plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.title("test (Keras.ver)")
p1 = plt.plot(dataf["OUTPUT_RNN"], label="label-A")
p2 = plt.plot(dataf["OUTPUT_ORI"], label="label-B")

plt.xlabel("step")
#plt.ylabel("predo")
plt.legend(["OUTPUT_RNN", "OUTPUT_ORI"],
#           ["wa~i!", "sugo~i!"],
           fontsize=20,
           loc=1,
 #          title="LABEL NAME",
           prop={'size':6})


plt.subplot(132)
plt.title("loss (Keras.ver, Kohashi.ver)")
p1 = plt.plot(loss[1], label="label-A")
p2 = plt.plot(df201[3], label="label-A")

plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["loss", "loss_own"],
#           ["wa~i!", "sugo~i!"],
           fontsize=20,
           loc=1,
#           title="LABEL NAME",
           prop={'size':6})

plt.subplot(133)
plt.title("test (Khashi.ver)")
p1 = plt.plot(df202["OUTPUT_RNN"], label="label-A")
p2 = plt.plot(df202["OUTPUT_ORI"], label="label-B")

plt.xlabel("step")
#plt.ylabel("predo")
plt.legend(["OUTPUT_RNN", "OUTPUT_ORI"],
#           ["wa~i!", "sugo~i!"],
           fontsize=20,
           loc=1,
 #          title="LABEL NAME",
           prop={'size':6})

plt.show()

