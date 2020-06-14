# -*- coding: utf-8 -*-
"""plot of filters of our model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pgZvCUlXuBfXM3XG0WXghno9r1ne7fn9
"""

from matplotlib import pyplot
from tensorflow.keras.models import load_model

model = load_model('my_model1.h5')

for layer in model.layers:

	if 'conv' not in layer.name:
		continue
	
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)

filters, biases = model.layers[2].get_weights()

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters, ix = 6, 1
for i in range(n_filters):

	f = filters[:, :, :, i]
	
	for j in range(3):
		
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
	
		pyplot.imshow(f[:, :, j], cmap='magma')
		ix += 1

pyplot.show()

