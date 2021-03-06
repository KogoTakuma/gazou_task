import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import math

NUMBER_OF_PICTURES = 10000
CELL_OF_X = 28
CELL_OF_Y = 28
CELL_OF_AROW = CELL_OF_X * CELL_OF_Y
WEIGHT_M = 5
WEIGHT_d = CELL_OF_AROW
Node_1 = 5
Node_2 = 10
epoch = 20
LR = 0.01
BATCH_SIZE = 30
global outputE
global X
global Y
global outputy
global minibatch_input_x,sigmoid_input_x,middle_input
global w1,w2,b1,b2
global result12

def sigmoid(a):
    e = math.e
    s = 1 / (1 + e**-a)
    return s
def numpy_frompyfunc(l, f):
    return np.frompyfunc(f, 1, 1)(l)

def softmax(input_a):
  max_a = np.max(input_a)
  sum_exp_a = 0.0
  a = np.empty(0)
  for num in range(Node_2):
    sum_exp_a = sum_exp_a + np.exp(input_a[num] - max_a)
  for num in range(Node_2):
    a = np.append(a,np.exp(input_a[num]-max_a)/sum_exp_a)
  return a

def randommaker(seed):
    global b1,b2,w1,w2
    np.random.seed(seed)
    b1 = np.random.normal(0, 1/CELL_OF_AROW, Node_1)
    b1 = np.reshape(b1,[Node_1,1])
    b2 = np.random.normal(0, 1/Node_1, Node_2)
    b2 = np.reshape(b2,[Node_2,1])
    w1 = np.reshape((np.random.normal(0, 1/CELL_OF_AROW, Node_1*CELL_OF_AROW)),([Node_1,CELL_OF_AROW]))
    w2 = np.reshape((np.random.normal(0, 1/Node_1, Node_1*Node_2)),([Node_2,Node_1]))

def input_layer(x_1):
  global b1,w1,sigmoid_input_x
  y1 = np.empty(0)
  result = np.empty(0)
  sigmoid_input_x = np.empty(0)  
  y1 = np.dot(w1,np.transpose(x_1)) + b1
  sigmoid_input_x = np.append(sigmoid_input_x,y1) 
  sigmoided_y = numpy_frompyfunc(y1,sigmoid)
  result_sigmoided = np.reshape(sigmoided_y,[Node_1,1])
  return result_sigmoided

def middle_layer(y_1):
  global b2,w2
  a_k = np.empty(0)
  a_k = np.dot(w2,y_1) + b2
  a_k = np.reshape(a_k,10)
  return a_k 

def output_layer(a_k): 
  return softmax(a_k)

def get_minibatch(num):
  global X, Y, outputy,result12,minibatch_input_x,middle_input
  x = np.random.choice(NUMBER_OF_PICTURES, BATCH_SIZE)#0~60000の中から100個の数字をランダムに選ぶ  
  result = 0
  minibatch_input_x = np.empty(0)
  middle_input = np.empty(0)
  result12 = np.empty(0)
  for num in range(BATCH_SIZE):  
    x1 = np.reshape(X[x[num]], (1,CELL_OF_AROW )) #画像を取り出し786次元ベクトルにする。
    minibatch_input_x = np.append(minibatch_input_x,x1)
    y1 = Y[x[num]]#答えの数
    b=input_layer(x1)#入力層
    middle_input = np.append(middle_input,b)
    c=middle_layer(b)#中間層
    d=output_layer(c)#出力層
    e=-math.log(d[y1])#クロスエントロピー誤差を求める
    row12 = d - np.eye(10)[y1]
    result += e
    result12 = np.append(result12,row12)
  result = result/BATCH_SIZE 
  outputy = result 
  result12 = np.reshape(result12,(Node_2,BATCH_SIZE))
  minibatch_input_x = np.reshape(minibatch_input_x,[CELL_OF_AROW,BATCH_SIZE])
  print("クロスエントロピー誤差 "+str(result))
  return result

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
    #result = np.dot(ones-oneliney,np.dot(oneliney,onelineey))
    result = [x * y for (x, y) in zip((ones - oneliney), [x * y for (x, y) in zip(onelineey, oneliney)])]
    return np.reshape(result,[Node_1,BATCH_SIZE])
def renew_parameter2(EnW2,Enb2):
  global w1,w2,b1,b2
  w2 = w2 - LR*EnW2 #22
  b2 = b2 - np.reshape(LR*Enb2,[Node_2,1])  #24
def renew_parameter1(EnW1,Enb1,):
  global w1,w2,b1,b2
  w1 = w1 - LR*EnW1 #21
  b1 = b1 - np.reshape(LR*Enb1,[Node_1,1]) #23


randommaker(1)#初期の重みを乱数で生成
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
for num in range(epoch):  
    get_minibatch(num)
    a=caluculate_16(w2,result12)
    b=caluculate_17(result12,np.reshape(middle_input,[Node_1,BATCH_SIZE]))
    c=caluculate_18(result12)
    result16=caluculate_16(w2,result12)
    renew_parameter2(b,c)
    result20 = caluculate_20(middle_input,result16)
    e=caluculate_17(result20,minibatch_input_x)
    f=caluculate_18(result20)
    renew_parameter1(e,f)
print("end")    

