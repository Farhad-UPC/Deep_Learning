import numpy as np
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.plotting import *

col_separator = ','
url = "https://raw.githubusercontent.com/neogi/machine-learning/master/regression/data/Week02/kc_house_train_data.csv"
house_date = pd.read_csv (url, sep = col_separator, header = 0, index_col = False )
print (house_date.head (5))

#Regression Coefficients
x = house_date ['sqft_living']
y = house_date ['price']
regression = np.polyfit(x,y,1)  #Last argument is degree of polynomial

#print ("[line slope] , [intercept] =")
#print (regression)

#Bokeh is an interactive visualization library that targets modern web browsers for presentation. 
t = figure (title =" price/sqtf", plot_width = 600, plot_height = 700)
t.xaxis.axis_label = 'Sqtf'
t.xaxis.axis_label_standoff = 20
t.xaxis.axis_line_width = 2
t.xaxis.axis_line_color = "blue"
t.xaxis.major_label_text_color = "red"
t.yaxis.axis_label = "Price-$"
t.yaxis.axis_line_width = 2
t.yaxis.axis_line_color = "red"
t.yaxis.major_label_orientation = "vertical"
t.yaxis.major_label_text_color = "blue"
t.yaxis.axis_label_standoff = 20
t.circle(house_date["sqft_living"], house_date["price"], fill_alpha = 0.1, size = 20, color ="orange")
t.line(x,[i*regression[0] + regression[1] for i in x], color = "black")

t.axis.minor_tick_in = -3
t.axis.minor_tick_out = 6
t.ygrid.minor_grid_line_color = "navy"
t.ygrid.minor_grid_line_alpha = 0.1
 
show (t)
