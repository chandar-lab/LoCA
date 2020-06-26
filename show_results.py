import numpy as np
import math
import json
import matplotlib.pyplot as plt
from numpy import trapz


filenames = {}; i = 0

filenames[i] = 'sarsa_lambda'; i+=1
# filenames[i] = 'sarsa_lambda_no_pre_training'; i+=1
filenames[i] = 'sarsa_lambda_flipped_actions'; i+=1

filenames[i] = 'MuZero'; i+=1
filenames[i] = 'MuZero_with_flipped_actions'; i+=1
#
# filenames[i] = 'sarsa_lambda_flipped_terminal'; i+=1
# filenames[i] = 'sarsa_lambda_flipped_terminal_flipped_actions'; i+=1
# # filenames[i] = 'MuZero'; i+=1
# filenames[i] = 'MuZero_no_pre_training_flipped'; i+=1



results = {}
results_std = {}
time = {}
labels = {}
area = {}
num_results = i
MuZero_steps = np.load('data/' + 'MuZero' + '_eval_steps.npy')
# MuZero2_steps = np.load('data/' + 'MuZero_no_pre_training' + '_eval_steps.npy')
# MuZero3_steps = np.load('data/' + 'MuZero_no_pre_training_flipped' + '_eval_steps.npy')

for i in range(num_results):
    results[i] = np.load('data/' + filenames[i] + '_avg_results.npy')
    results_std[i] = np.load('data/' + filenames[i] + '_std_results.npy')
    with open('data/' + filenames[i] + '_settings.txt') as f:
        settings = json.load(f)
    num_steps = settings['num_steps']
    num_datapoints = settings['num_datapoints']
    window_size = num_steps // num_datapoints
    time[i] = np.arange(1,num_datapoints+1)*window_size

    if filenames[i] == 'MuZero':
        time[i] = MuZero_steps
    if filenames[i] == 'MuZero_with_no_pre_training':
        time[i] = MuZero2_steps
    if filenames[i] == 'MuZero_no_pre_training_flipped':
        time[i] = MuZero3_steps

labels = ['sarsa($\lambda$)', 'sarsa($\lambda$) with shuffled actions', 'MuZero', 'MuZero with shuffled actions']
          # 'sarsa($\lambda$)', 'sarsa($\lambda$) with flipped actions'] # ,'MuZero with no pre-training'
font_size = 14
font_size_legend = 14
font_size_title = 14

plt.rc('font', size=font_size)  # controls default text sizes
plt.rc('axes', titlesize=font_size_title)  # fontsize of the axes title
plt.rc('axes', labelsize=font_size)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_size)  # fontsize of the tick labels
plt.rc('ytick', labelsize=font_size)  # fontsize of the tick labels
plt.rc('legend', fontsize=font_size_legend)  # legend fontsize
plt.rc('figure', titlesize=font_size_title)  # fontsize of the figure title

color = 'krgbykrgby'
titles = ['Original', 'Flipped']
mv = [4, 6, 1, 2, 2, 2, 2, 1]
std_sc = [0.3, 0.3, 1, 0.5]
# plt.figure()
# fig, axs = plt.subplots(1)
# fig.suptitle('MountainCar-v0')

for i in [0, 1, 2, 3]:

    plt.plot(time[i], np.convolve(results[i][:], np.ones((mv[i],)) / mv[i], mode='same'), color[i],
                label=labels[i])
    plt.fill_between(time[i], np.convolve(results[i] - std_sc[i] * results_std[i], np.ones((mv[i],)) / mv[i], mode='same'),
                        np.convolve(results[i] + std_sc[i] * results_std[i], np.ones((mv[i],)) / mv[i], mode='same'), facecolor=color[i], alpha=0.4)
count = 0
# for i in [2,3]:
#     axs[1].plot(time[i], np.convolve(results[i][:], np.ones((mv[i],)) / mv[i], mode='same'), color[count],
#                     label=labels[i])
#     count += 1
area[0] = trapz(results[0][:80], time[0][:80])
area[1] = trapz(results[1][:80], time[1][:80])
area[2] = trapz(results[2], time[2])
area[3] = trapz(results[3], time[3])

print("area =", area)



# for i, ax in enumerate(axs.flat):
#     ax.set(xlabel='time steps', ylabel='top-terminal fraction')
#     ax.title.set_text(titles[i])
#     ax.legend(loc='best')
#     ax.set_xlim([0, 35000])
#     ax.set_ylim([0,1.1])
#     ax.grid()
plt.legend(loc='best')
plt.xlim([0, 45000])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ylim([0,1.1])
plt.grid()
plt.ylabel('top-terminal fraction')
plt.xlabel('time steps')
# plt.title('MountainCar-v0')
plt.show()