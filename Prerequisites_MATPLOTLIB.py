
#Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms.

import numpy as np
import matplotlib.pyplot as plt

#line plot
year = [2011, 2012, 2013, 2014, 2015, 2016]
cars_produced = [59.8, 63.0, 65.7, 67.7, 68.5, 72.1]
plt.plot(year, cars_produced)
#print (plt.show())
plt.show()
print ('-------------------------------------------------')
plt.plot(year, cars_produced)
plt.xlabel('Year')
plt.ylabel('Cars_produced')
plt.show()
print ('-------------------------------------------------')
plt.figure(figsize=(10, 9), dpi=90)
plt.plot(year, cars_produced)
plt.title('Cars produced In The World')
plt.xlabel('Year')
plt.ylabel('Cars_produced')
plt.xticks([2008, 2010, 2011, 2013, 2014, 2016, 2020] )
plt.yticks([59.8, 63.0, 65.7, 67.7, 68.5, 72.1],['60M','63M','66M','68M','69M','73M'])
plt.show()
print ('-------------------------------------------------')

#scatter plot
plt.scatter(year, cars_produced)
plt.show()
print ('-------------------------------------------------')
size = np.array(cars_produced)*4
plt.scatter(year, cars_produced, s=size)
plt.show()
print ('-------------------------------------------------')
plt.scatter(year, cars_produced)
size = np.array(cars_produced)*4
Colors = ['black', 'violet', 'orange', 'green', 'blue', 'yellow']
#alpha : scalar, optional, default: None   => The alpha blending value, between 0 (transparent) and 1 (opaque)
plt.scatter(year, cars_produced, s=size, c=Colors, alpha=1)
plt.title('Cars produced In The World', rotation=0 ,fontsize='large')
plt.xlabel('Year')
plt.ylabel('Population')
#plt.text(0,60,'iran Capital')
plt.show()


plt.scatter(np.arange(6),cars_produced)
plt.show()


#histogram plot
plt.hist(cars_produced, bins='auto')
plt.show()


plt.hist(cars_produced, bins=1)
plt.show()

#The "square root rule" is a commonly-used rule of thumb for choosing number of bins
#The bins parameter tells you the number of bins that your data will be divided into. set bins=1 to understand it

print ('-------------------------------------------------')
plt.pie(cars_produced,labels=year )
plt.show()
print ('-------------------------------------------------')
plt.pie(cars_produced,labels=['A', 'B', 'C', 'D', 'E', 'F'] )
plt.show()

plt.pie(cars_produced,labels=cars_produced )
plt.show()



#Multiple Plot vs Subplot
cars_produced_spain = [0.8, 0.90, 1.1, 1.99, 2.5, 3.72]
plt.plot(year,cars_produced,ls='-',lw=2,marker='+',mew=8)
plt.plot(year,cars_produced_spain,ls='--',lw=1)
plt.title('Spain vs world_cars_produced')
plt.xlabel('Year')
plt.ylabel('cars_produced')
plt.yticks([20,30,40,50,60,70,80],['20M', '30M', '40M', '50M' ,'60M', '70M', '80M',])
plt.legend(['Spain','world'],loc='best')
#plt.text(1980,40,'Iran–Iraq War',fontsize=15)
plt.grid()
plt.annotate("comment",xy=(2012,1.67) ,xytext=(2013,15.7), arrowprops=dict(facecolor='silver',width=4),fontsize=14)
plt.show()


plt.subplot(1,2,1)
plt.plot(year, cars_produced)
plt.title('Cars produced In The World')
plt.subplot(1,2,2)
plt.plot(year,cars_produced_spain)
plt.title('Cars produced In The Spain')
plt.savefig('myplot.jpg')
%pwd
#import os
#print (os.getcwd())  # Prints the current working directory
plt.show()


x = np.linspace(0,15,150)
y = np.sin(x)
plt.plot(x, y)
plt.xlabel("Time")
plt.ylabel("Function of Time")
plt.title("Line Chart")
plt.show()
