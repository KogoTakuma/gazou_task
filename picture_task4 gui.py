from os import error
import numpy as np
import mnist
import matplotlib.pyplot as plt
from pylab import cm
import math
import tkinter
import tkinter.ttk as ttk
from tkinter import filedialog
import os


NUMBER_OF_PICTURES = 10000
CELL_OF_X = 28
CELL_OF_Y = 28
CELL_OF_AROW = CELL_OF_X * CELL_OF_Y
WEIGHT_M = 5
WEIGHT_d = CELL_OF_AROW
Node_1 = 5
Node_2 = 10
BATCH_SIZE = 100
LR = 0.01

def sigmoid(a):
    s = 1 / (1 + np.exp(-a))
    return s

def softmax(input_a):
  max_a = np.max(input_a)
  sum_exp_a = 0.0
  a = np.empty(0)
  for num in range(Node_2):
    sum_exp_a = sum_exp_a + np.exp(input_a[num] - max_a)
  for num in range(Node_2):
    a = np.append(a,np.exp(input_a[num]-max_a)/sum_exp_a)
  return a

def preprocessing():
  print('Type Number!')
  val_1 = input('picture number :')
  if val_1.isdecimal():
    int_val_1 = int(val_1)
    if 0<= int_val_1 and int_val_1 < 10000:

      X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
      Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz");      
      print ('正しい値 '+ str(Y[int_val_1]))
      x1 = np.reshape(X[int_val_1],(1,CELL_OF_AROW ))
      plt.imshow(X[int_val_1], cmap=cm.gray)
      #plt.show()
      return x1
    else:
      print('invalid value')
      exit()
  else:
    print('invalid value')
    exit()

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
    row12 = np.reshape((d - np.eye(Node_2)[y1]),[1,Node_2])
    result += e
    result12 = np.append(result12,row12)
  result = result/BATCH_SIZE 
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

def caluculate_20(y,ey):
    result = np.empty(0)
    oneliney = np.reshape(y,[Node_1*BATCH_SIZE,1])
    onelineey = np.reshape(ey,[Node_1*BATCH_SIZE,1])
    ones =np.reshape((np.ones(Node_1*BATCH_SIZE)),[Node_1*BATCH_SIZE,1])
    #result = [x * y for (x, y) in zip((ones - oneliney), [x * y for (x, y) in zip(onelineey, oneliney)])]
    subbedy = [x - y for (x, y) in zip(ones,oneliney)]
    ey1y = [x * y for (x, y) in zip(onelineey,subbedy)]
    result = [x * y for (x, y) in zip(ey1y,oneliney)]
    result_reshaped = np.reshape(result,[Node_1,BATCH_SIZE])
    return result_reshaped
def renew_parameter2(EnW2,Enb2,w2,b2):
  renewed_w2 = w2 - LR*EnW2 #22
  renewed_b2 = b2 - np.reshape(LR*Enb2,[Node_2,1]) #24
  return renewed_w2,renewed_b2
def renew_parameter1(EnW1,Enb1,w1,b1):
  renewed_w1 = w1 - LR*EnW1 #21
  renewed_b1 = b1 - np.reshape(LR*Enb1,[Node_1,1]) #23
  return renewed_w1, renewed_b1

def learning():
  w1_new,b1_new,w2_new,b2_new  = randommaker(1)#初期の重みを乱数で生成
  X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
  Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")
  Node_1 = num_area_middlenode.get()
  BATCH_SIZE = num_area_batchsize.get()
  EPOCH = num_area_epoch.get()
  for i in range(EPOCH):
    lossaverage = 0.0
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
        lossaverage = lossaverage + result_cross_entropy_loss
        print(i,"エポック クロスエントロピー誤差 "+str(result_cross_entropy_loss),num)    
    print(i,"エポック 平均クロスエントロピー誤差 ",lossaverage/int(NUMBER_OF_PICTURES/BATCH_SIZE))
  print("end")    
  np.savez('/Users/t_kogo/Desktop/gazou_task/learningdata.npz',w1=w1_old,b1=b1_old,w2=w2_old,b2=b2_old,node1=Node_1)
  print("saved")

def testing(npz):
  w1=npz['w1']
  w2=npz['w2']
  b1=npz['b1']
  b2=npz['b2']
  Node_1=npz['Node_1']
  a=preprocessing()
  b=input_layer(a,w1,b1)
  c=sigmoid(b)
  d=middle_layer(c,w2,b2)
  e=output_layer(d)
  print('出力 '+str(np.argmax(e)))
  return np.argmax(e)

def select_file():
  idir = 'C:\\python_test'
  file_path = tkinter.filedialog.askdirectory(initialdir = idir)  
  epoch_label = tkinter.Label(master=frame1, text=file_path, font=20)
  epoch_label.place(x=60, y=80)


np.random.seed(1)
npz = np.load('/Users/t_kogo/Desktop/gazou_task/learningdata.npz')
frame1 = tkinter.Tk() 
frame1.geometry("800x500")
frame1.title('計算機科学実験4 画像認識 課題4')
button_learning = tkinter.Button(frame1, text='学習',command=learning)
button_learning.place(x=700, y=100)
num_area_epoch = tkinter.Entry(master=frame1, width=5, font=20)
num_area_epoch.place(x=60, y=60)
num_area_epoch.insert(tkinter.END,"10")
num_area_output_name = tkinter.Entry(master=frame1, width=5, font=20)
num_area_output_name.place(x=420, y=60)
num_area_batchsize = tkinter.Entry(master=frame1, width=5, font=20)
num_area_batchsize.place(x=180, y=60)
num_area_batchsize.insert(tkinter.END,"200")
num_area_middlenode = tkinter.Entry(master=frame1, width=5, font=20)
num_area_middlenode.place(x=300, y=60)
num_area_middlenode.insert(tkinter.END,"100")
epoch_label = tkinter.Label(master=frame1, text='エポック数', font=20)
epoch_label.place(x=60, y=40)
batch_label = tkinter.Label(master=frame1, text='バッチサイズ', font=20)
batch_label.place(x=180, y=40)
middle_label = tkinter.Label(master=frame1, text='中間層数', font=20)
middle_label.place(x=300, y=40)
middle_label = tkinter.Label(master=frame1, text='出力ファイル名', font=20)
middle_label.place(x=420, y=40)
middle_label = tkinter.Label(master=frame1, text='機械学習をする条件を設定してください。', font=20)
middle_label.place(x=0, y=40)
outputname_label = tkinter.Label(master=frame1, text='出力ファイル名')
module = ('tkinter', 'math', 'os', 'pyinstaller', 'pathlib', 'sys')
combobox = ttk.Combobox(frame1,width=10, values=module, state="readonly")
combobox.place(x=600, y=60)
button_learning = tkinter.Button(frame1, text='出力先',command=select_file)
button_learning.place(x=300, y=80)
frame1.mainloop() 
