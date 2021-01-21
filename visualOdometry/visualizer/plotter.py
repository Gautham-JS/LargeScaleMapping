import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#plt.style.use('fivethirtyeight')

x_values = []
y_values = []

xgt_values = []
ygt_values = []

data = pd.read_csv("/home/gautham/Documents/Projects/LargeScaleMapping/trajectory.csv")

index = count()
print(data.iloc[:5,2], data.iloc[:5,5])

xBound = np.max( [np.max(data.iloc[:,0]), np.max(data.iloc[:,3])] )
ybound = np.max( [np.max(data.iloc[:,2]), np.max(data.iloc[:,5])] )


def animate(i):
    # x_values = data.iloc[:,1]
    # y_values = data.iloc[:,3]
    x_values.append(data.iloc[i,0])
    y_values.append(-1*data.iloc[i,2])

    xgt_values.append(data.iloc[i,3])
    ygt_values.append(data.iloc[i,5])

    plt.cla()
    plt.scatter(x_values, y_values, label="Predicted Trajectory")
    plt.scatter(xgt_values, ygt_values, label="True Trajectory")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.legend()
    plt.axis("scaled")
    plt.grid(True)


ani = FuncAnimation(plt.gcf(), animate, 4500, interval=10)

plt.show()