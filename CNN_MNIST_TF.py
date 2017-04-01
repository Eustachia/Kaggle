
# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)



# In[2]:

train_file='J:\\Eustachia\\KAGGLE\\digits\\train.csv'
test_file='J:\\Eustachia\\KAGGLE\\digits\\test.csv'
train_data=pd.read_csv(train_file)
test_data=pd.read_csv(test_file)
labels=train_data['label'].values
train=train_data.drop('label', axis=1).values  #train_data.ix[:, 1:]
X_std=StandardScaler().fit_transform(train)
Y_std=StandardScaler().fit_transform(test_data.values)

new_labels=np.zeros([len(labels), len(np.unique(labels))])
for i in range(0, len(labels)):
    new_labels[i, labels[i]]=1

X = X_std.astype(np.float32)
Y = Y_std.astype(np.float32)
train_num = 41000
val_num = 1000
X_train, y_train = X[:train_num],labels[:train_num].astype(np.int32) #new_labels[:train_num]
X_val, y_val = X[train_num:],labels[train_num:].astype(np.int32) #new_labels[train_num:]

print(X_train.shape, y_train.shape, X_train.dtype, y_train.dtype)
print(X_val.shape, y_val.shape, X_val.dtype, y_val.dtype)
N_f=X_train.shape[1]
N_s=X_train.shape[0]
N_c=len(np.unique(labels))



# In[2]:

mnist = learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
print (train_data.shape, train_labels.shape)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)


# In[3]:

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  input_layer = tf.reshape(features, [-1, 28, 28, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  loss = None
  train_op = None

  # Calculate Loss (for both TRAIN and EVAL modes)
  if mode != learn.ModeKeys.INFER:
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10) #labels
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == learn.ModeKeys.TRAIN:
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        learning_rate=0.001,
        optimizer="SGD")

  # Generate Predictions
  predictions = {
      "classes": tf.argmax(
              input=logits, axis=1)} #,
 #     "probabilities": tf.nn.softmax(
 #        logits, name="softmax_tensor")
 # }

  # Return a ModelFnOps object
  return model_fn_lib.ModelFnOps(
      mode=mode, predictions=predictions, loss=loss, train_op=train_op)


# In[4]:


#Create the Estimator
mnist_classifier = learn.Estimator(
      model_fn=cnn_model_fn, model_dir="J:\\Eustachia\\KAGGLE\\digits\\")


# In[5]:

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=20000)


# In[6]:

# Train the model
mnist_classifier.fit(
    x=X_train, #train_data, #X_train,
    y=y_train, #train_labels, #y_train,
    batch_size=100,
    steps=60000,
    monitors=None) #[logging_hook])


# In[8]:

# Configure the accuracy metric for evaluation
metrics = {
    "accuracy":
        learn.MetricSpec(
            metric_fn=tf.metrics.accuracy, prediction_key="classes"),
}


# In[9]:

# Evaluate the model and print results
eval_results = mnist_classifier.evaluate(
    x=X_val, y=y_val, metrics=metrics) #eval_data, eval_labels
print(eval_results)


# In[10]:
my_prediction=[]
batch=100
for i in range(0, len(Y), 100):
    if i % 1000 ==0:
        print (i)
    test_prediction = mnist_classifier.predict(x=Y[i:i+batch], as_iterable=True)
    my_prediction.append([c["classes"] for i, c in enumerate(test_prediction)])
#print (len(my_prediction))

my_prediction = np.asarray(my_prediction).ravel()
#print (my_prediction.shape())
#print ("done")
print (my_prediction.shape)
# In[ ]:

#a= enumerate(test_prediction)
#for i, c in a:
#    print (i, c)
#    my_prediction.append(c["classes"])
#print (my_prediction[0:10])


# In[ ]:


my_solution = pd.DataFrame(my_prediction, columns = ["Label"])
my_solution.index += 1 
my_solution.head()
my_solution.to_csv("my_solution_cnntf.csv", index_label = ["ImageId"])

