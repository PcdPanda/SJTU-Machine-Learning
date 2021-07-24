import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame(pd.read_csv(r"D:\PANDA\Study\VG441\Homework\Midterm 1\demands.csv"))
T=df['time']
D=df['demands']
plt.title('Demands against Time')
plt.scatter(T,D)
plt.show()