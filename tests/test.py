import matplotlib.pyplot as plt
import csv
import torch
import numpy as np

error = []
x = []
y = []
x0 = []
y0 = []

# sqrt((x-x0)^2+(y-y0)^2)/sqrt(x0^2+y0^2) * 100
# (sqrt(x^2+y^2)-sqrt(x0^2+y0^2)) / sqrt(x0^2+y0^2) * 100
with open('testt.txt','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    for row in plots:
        r0 = row[0]
        r1 = row[1]
        r2 = row[2]
        r3 = row[3]
        # error.append(np.sqrt(float(row[0])**2 + float(row[1])**2) - np.sqrt(float(row[2])**2 + float(row[3])**2))
        e = np.sqrt((float(row[0]) - float(row[2]))**2 + (float(row[1]) - float(row[3]))**2)
        e /= np.sqrt(float(row[2])**2 + float(row[3])**2)
        e *= 100

        # e = (np.sqrt(float(row[0])**2 + float(row[1])**2) - np.sqrt(float(row[2])**2 + float(row[3])**2))
        # e /= np.sqrt(float(row[2])**2 + float(row[3])**2)
        # e *= 100

        error.append(e)

        x.append(float(row[0]))
        y.append(float(row[1]))
        x0.append(float(row[2]))
        y0.append(float(row[3]))

idx = [i for i in range(len(x0))]

print(np.average(error))

# plt.scatter(x0, y0, s=100, c=error)

plt.scatter(idx, error, label='Error')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.scatter(error, label='Real')
# plt.xlabel('x0')
# plt.ylabel('y0')
plt.title('Error for 4-pixel range\n')
plt.grid(True)
plt.legend()
plt.show()