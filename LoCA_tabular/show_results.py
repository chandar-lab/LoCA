################################################
#  Author: Harm van Seijen
#  Copyright 2020 Microsoft
################################################

import numpy as np
import math
import json
import matplotlib.pyplot as plt

filenames = {}; i = 0

filenames[i] = 'sarsa_lambda_LoCA'; i+=1
filenames[i] = 'sarsa_lambda_noLoCA'; i+=1


results = {}
time = {}
labels = {}
avg_regret = {}
std_error = {}
num_results = i


for i in range(num_results):
    labels[i]=filenames[i]
    performance = np.load('data/' + filenames[i] + '_results.npy')
    results[i] = np.mean(performance,axis=0)

    with open('data/' + filenames[i] + '_settings.txt') as f:
        settings = json.load(f)
    if 'avg_regret' in settings:
        avg_regret[i] = settings['avg_regret']
        std_error[i] = settings['std_error']
    else:
        avg_regret[i] = 0
        std_error[i] = 0
    num_steps = settings['num_steps']
    num_datapoints = settings['num_datapoints']
    window_size = num_steps // num_datapoints
    time[i] = np.arange(1,num_datapoints+1)*window_size


### show  regrets ##############

print("REGRET (x1000) :")
for i in range(num_results):
    print(labels[i], ": {:3.2f}".format(avg_regret[i]/1000), ", std error: {:3.2f}".format(std_error[i]/1000))






##########

plt.figure(figsize=(8,5))

font_size = 20
font_size_legend = 20
font_size_title = 20


plt.rc('font', size=font_size)  # controls default text sizes
plt.rc('axes', titlesize=font_size_title)  # fontsize of the axes title
plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
plt.rc('ytick', labelsize=font_size)  # fontsize of the lt.rc('legend', fontsize=font_size_legend)  # legend fontsize
plt.rc('figure', titlesize=font_size)  # fontsize of the figure title

color = 'krgbykrgby'


for i in range(num_results):
    plt.plot(time[i]/1000,results[i],color[i],label=labels[i])


plt.ylim(0,1.1)
plt.ylabel('top-terminal fraction')
plt.xlabel('time steps (x 1000)')
plt.legend(loc=7)
plt.show()