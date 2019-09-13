import numpy as np 
#import matplotlib.pyplot as plt 
from bokeh.plotting import figure, show
from bokeh.plotting import *


current_x= 3 #Starts at x=3
learning_rate = 0.01
precision = 0.000001
prev_step = 1
maximum_iteration = 10000
iteration_counter = 0
# xn = (xn-1) - (Learning Rate) * (dy/dx) 
# e.g. dy/dx = d/dx (x+5)**2 ->  2 * (x+5) --->
# --->  x = 3 ->  x1 = 3 - (0.01) * (2 *(3+5)) = 2.84
df = lambda x : 2 * (x+5)   # anonymous function  ---> Gradient

while prev_step > precision and iteration_counter < maximum_iteration:
	prev_x = current_x
	current_x = current_x - learning_rate * df(prev_x)
	prev_step = abs (current_x - prev_x)
	iteration_counter = iteration_counter + 1
	print ("Iteration", iteration_counter,"\n")
	print ("X =", current_x)


print (" Local Minimum : ", current_x)

x = np.linspace(-20,10,100)
y = (x+5)**2

#plt.plot(x,y)
#plt.grid()
#plt.show()

c = str (current_x)

t = figure (title ="Local minimum of y = (x+5)**2 ---> \t" + c, plot_width = 600, plot_height = 700)
t.xaxis.axis_label = 'X'
t.xaxis.axis_label_standoff = 20
t.xaxis.axis_line_width = 2
t.xaxis.axis_line_color = "blue"
t.xaxis.major_label_text_color = "red"
t.yaxis.axis_label = "Y"
t.yaxis.axis_line_width = 2
t.yaxis.axis_line_color = "red"
t.yaxis.major_label_orientation = "vertical"
t.yaxis.major_label_text_color = "blue"
t.yaxis.axis_label_standoff = 20
t.line(x,y , color = "black")
t.axis.minor_tick_in = -3
t.axis.minor_tick_out = 6
t.ygrid.minor_grid_line_color = "navy"
t.ygrid.minor_grid_line_alpha = 0.1
show (t) #  x=-5 ->  y=0  -> Minimum
