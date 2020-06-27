import numpy as np
from matplotlib import pyplot as plt
import pickle

outC = open('C','wb')
re = []


infile = open('FrameHistory_file','rb')
new_dict = pickle.load(infile)
infile.close()

print(new_dict)


for x in range(len(new_dict)):
	print(new_dict[x])
	







	