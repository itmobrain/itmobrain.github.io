---
layout: post
title: "TensorFlow Lite(1)"
date: 2018-11-09 03:54:51
image: '/assets/img/'
description: "TensorFlow Lite 모델 생성부터 .tflite까지 + Google Colab Files"
tags:
- TFLite
- Tensorflow
- Colab
categories:
- DeepLearning
- Tutorial
twitter_text:
---

이번 학기 프로젝트 중에 2개나 모바일에 딥러닝 모델을 사용해야할 필요가 생겨서(사실은 지난 학기에 텐서플로우 라이트를 맛본게 화근이었다...왠지 모를 자신감 상승...) 튜토리얼 정도 수준이 아닌 능동적인 수준의 실력이 필요하게 됐다.
TensorFlow-for-poet같은 구글 코드랩의 예제 앱들을 보면서 어떻게 이렇게 되는걸까 항상 궁금했는데, 한 번 기초부터(Linear Regression 모델) 적용해보고자 한다.  

이 글을 준비하기 위해 여러 블로그 글과 유투브 강의들을 참고하였다.  

  
    
      
      
## 당신이 이 글을 통해 배울 수 있는 것
1. 내가 만든 그래프를 freeze하기
2. 약간의 Google Colab  
  
  
  
  
## 모델을 freeze한다?
모델을 freeze한다는 것은 weight나 bias값을 variable에서 constant로 만들어준다는 의미이다.(말그대로 출렁이던 모델을 꽁꽁 얼려버린다는 의미이다)  
  
    
    
## 모델을 왜 freeze하는건데?
이유는 별거 없다.
1. 트레이닝 과정을 더이상 거치지 않고, 필요하지도 않다고 판단되어서.
2. 텐서플로우는 gradient값이나 meta data등을 만들어내게 되는데, 이러한 것들이 실제로 결과값을 inference하는 단계에서는 더이상 필요하지 않기 때문.
3. 모델의 파라미터들을 export하기위한 준비를 하려고.  
  
## 어떻게 freeze하는데?
우선 예제 코드 [링크](https://colab.research.google.com/drive/1pHT172kXrhLCPBv-7YaVfoa47p-DLO73)이다. (예제라고 부를 정도로 기초탄탄 코드는 아니다. 죄송하다...)

  
  
## 학습된 모델, 그래프, 체크포인트 구하기
```python
from google.colab import files # mounting google drive
import tensorflow as tf
import numpy as np

W = tf.Variable(initial_value=tf.random_normal([1]), name='weight', trainable=True)
b = tf.Variable(initial_value=0.001, name='bias', trainable=True)

x = tf.placeholder(dtype=tf.float32, shape=[1], name='x')
y = tf.add(tf.multiply(W, x), b, name='output')

init = tf.global_variables_initializer()

saver = tf.train.Saver()
save_path = "data/"
model_save = save_path + "model.ckpt"

with tf.Session() as sess:
    sess.run(init)
    op = sess.run(y, feed_dict={x: np.reshape(1.5, [1])})
    saver.save(sess, model_save)
    tf.train.write_graph(sess.graph_def, save_path, 'savegraph.pbtxt')

# 다운로드 받기(Colab + Google Drive)
files.download("data/savegraph.pbtxt")
files.download("data/model.ckpt.meta")
```

## 모델 freeze하기
```python
from tensorflow.python.tools import freeze_graph

# Freeze the graph
save_path = "data/"
MODEL_NAME = 'Sample_model'
input_graph_path = save_path + 'savegraph.pbtxt'
checkpoint_path = save_path + 'model.ckpt'
input_saver_def_path = ""
input_binary = False
output_node_names = "output"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = save_path + 'frozen_model_' + MODEL_NAME + '.pb'
clear_devices = True

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                         input_binary, checkpoint_path, output_node_names,
                         restore_op_name, filename_tensor_name,
                         output_frozen_graph_name, clear_devices, "")
```

## frozen 모델 import해오기, Input & Output 노드 정의하기
```python
graph_def_file = 'data/frozen_model_Sample_model.pb' # our pb file

input_arrays = ['x'] # input node, 내가 그래프 만들 때 사용한 input의 이름으로 설정해야됨. output도 동일!
output_arrays = ['output'] # output node

# DEPRECATED : tf.contrib.lite.TocoConverter.from_frozen_graph
converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)

tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

files.download("converted_model.tflite") # tflite 파일 다운로드
```