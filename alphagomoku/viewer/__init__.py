from . import Config as Config
#from . import nativeUI
from .UI import UI
from .dataviewer import Dataviewer

#import numpy as np 
#import sys
#import Config

# import matplotlib.pyplot as plt 

# class Dataviewer:
#     def __init__(self,gamedata):
#         self.gamedata = gamedata

#     def plot_index(self,index):
#         chessboard = self.gamedata.getdata([index])[0]
#         width, height = self.gamedata.getdatashape()
#         blackx, blacky, whitex, whitey = list(), list(), list(), list()
#         for i in range(width):
#             for j in range(height):
#                 if chessboard[i,j] != 0:
#                     if chessboard[i,j] == 1:
#                         whitex.append(i)
#                         whitey.append(j)
#                     else:
#                         blackx.append(i)
#                         blacky.append(j)
#         plot_white = plt.plot(whitex, whitey, 'ro')
#         plt.setp(plot_white, markersize=30)

#         plot_black = plt.plot(blackx, blacky, 'bo')
#         plt.setp(plot_black, markersize=30)

#         plt.show()