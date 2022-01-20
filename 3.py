from this import d
import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import math

NUMBER_OF_PICTURES = 60000
CELL_OF_X = 28
CELL_OF_Y = 28
CELL_OF_AROW = CELL_OF_X * CELL_OF_Y
WEIGHT_M = 5
WEIGHT_d = CELL_OF_AROW
Node_1 = 5
Node_2 = 10
EPOCH = 10
LR = 0.01
BATCH_SIZE = 200
global outputE
global X
global Y
global outputy

def sigmoid(a):
    s = 1 / (1 + np.exp(-a))
    return s

def softmax(input_a):
  max_a = np.max(input_a)
  sum_exp_a = 0.0
  a = np.empty(0)
  for num in range(Node_2): 
    sum_exp_a = sum_exp_a + np.exp(input_a[num,0] - max_a)
  for num in range(Node_2):
    a = np.append(a,np.exp(input_a[num,0]-max_a)/sum_exp_a)
  return a

def randommaker(seed):
    np.random.seed(seed)
    b1 = np.random.normal(0, 1/CELL_OF_AROW, Node_1)
    b1 = np.reshape(b1,[Node_1,1])
    b2 = np.random.normal(0, 1/Node_1, Node_2)
    b2 = np.reshape(b2,[Node_2,1])
    w1 = np.reshape((np.random.normal(0, 1/CELL_OF_AROW, Node_1*CELL_OF_AROW)),([Node_1,CELL_OF_AROW]))
    w2 = np.reshape((np.random.normal(0, 1/Node_1, Node_1*Node_2)),([Node_2,Node_1]))
    return w1,b1,w2,b2

def input_layer(x_1,w1,b1):
  y1 = np.empty(0)
  y1 = np.dot(w1,x_1) + b1
  return y1

def middle_layer(y_1,w2,b2):
  a_k = np.empty(0)
  a_k = np.dot(w2,y_1) + b2
  return a_k 

def output_layer(a_k): 
  return softmax(a_k)

def get_minibatch(X,Y,num,w1,b1,w2,b2):
  x = np.random.choice(NUMBER_OF_PICTURES, BATCH_SIZE)#0~60000の中から200個の数字をランダムに選ぶ  
  result = 0
  sigmoid_input = np.empty(0)
  minibatch_input_x = np.empty(0)
  middle_input = np.empty(0)
  result12 = np.empty(0)
  row12 = np.empty(0)
  for num in range(BATCH_SIZE):  
    x1 = np.reshape(X[x[num]], (CELL_OF_AROW,1)) #画像を取り出し786次元ベクトルにする。
    minibatch_input_x = np.append(minibatch_input_x,x1)
    y1 = Y[x[num]]#答えの数
    b=input_layer(x1,w1,b1)#入力層
    sigmoid_input = np.append(sigmoid_input,b)
    sigmoided_output = sigmoid(b)
    result_sigmoided = sigmoided_output
    middle_input = np.append(middle_input,result_sigmoided)
    c=middle_layer(result_sigmoided,w2,b2)#中間層
    d=output_layer(c)#出力層
    e=-math.log(d[y1])#クロスエントロピー誤差を求める
    row12 = np.reshape((d - np.eye(10)[y1]),[1,Node_2])
    result += e
    result12 = np.append(result12,row12)
  result = result/BATCH_SIZE 
  outputy = result 
  result12 = np.transpose(np.reshape(result12,[BATCH_SIZE,Node_2]))/BATCH_SIZE 
  minibatch_input_x = np.transpose(np.reshape(minibatch_input_x,[BATCH_SIZE,CELL_OF_AROW]))
  middle_input = np.transpose(np.reshape(middle_input,[BATCH_SIZE,Node_1]))
  return result,result12,minibatch_input_x,middle_input

def caluculate_16(w,ey):
  return np.dot(np.transpose(w),ey)

def caluculate_17(ey,x):
  return np.dot(ey,np.transpose(x))

def caluculate_18(input):
  return np.sum(input, axis=1)

def caluculate_19(input):
  sigmoided_input = sigmoid(input)
  return (1-sigmoided_input)*sigmoided_input

def caluculate_20(y,ey):
    result = np.empty(0)
    oneliney = np.reshape(y,[Node_1*BATCH_SIZE,1])
    onelineey = np.reshape(ey,[Node_1*BATCH_SIZE,1])
    ones =np.reshape((np.ones(Node_1*BATCH_SIZE)),[Node_1*BATCH_SIZE,1])
    result = [x * y for (x, y) in zip((ones - oneliney), [x * y for (x, y) in zip(onelineey, oneliney)])]
    return np.reshape(result,[Node_1,BATCH_SIZE])
def renew_parameter2(EnW2,Enb2,w2,b2):
  renewed_w2 = w2 - LR*EnW2 #22
  renewed_b2 = b2 - np.reshape(LR*Enb2,[Node_2,1]) #24
  return renewed_w2,renewed_b2
def renew_parameter1(EnW1,Enb1,w1,b1):
  renewed_w1 = w1 - LR*EnW1 #21
  renewed_b1 = b1 - np.reshape(LR*Enb1,[Node_1,1]) #23
  return renewed_w1, renewed_b1

w1_new,b1_new,w2_new,b2_new  = randommaker(1)#初期の重みを乱数で生成
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
for i in range(EPOCH):
    for num in range(int(NUMBER_OF_PICTURES/BATCH_SIZE)):    
        w1_old,b1_old,w2_old,b2_old = w1_new,b1_new,w2_new,b2_new 
        result_cross_entropy_loss,result12,minibatch_input_array,middle_input_array = get_minibatch(X,Y,num,w1_old,b1_old,w2_old,b2_old)
        result16_2=caluculate_16(w2_old,result12)
        result17_2=caluculate_17(result12,middle_input_array)
        result18_2=caluculate_18(result12)
        w2_new,b2_new = renew_parameter2(result17_2,result18_2,w2_old,b2_old)
        result20 = caluculate_20(middle_input_array,result16_2)
        result17_1=caluculate_17(result20,minibatch_input_array)
        result18_1=caluculate_18(result20)
        w1_new,b1_new = renew_parameter1(result17_1,result18_1,w1_old,b1_old)
        print(i,"エポック クロスエントロピー誤差 "+str(result_cross_entropy_loss),num)    
print("end")    
np.savez('/Users/t_kogo/Desktop/gazou_task/learningdata.npz',w1=w1_old,b1=b1_old,w2=w2_old,b2=b2_old)
print("saved")