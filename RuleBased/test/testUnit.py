import numpy as np
from sklearn.model_selection import train_test_split


a = np.arange(0,10)
b = np.zeros_like(a)
train_x,test_x,train_y,test_y = train_test_split(a,b,test_size=5)
print(train_x)
print(test_x)
print(train_y)
print(test_y)
