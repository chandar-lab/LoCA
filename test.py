import matplotlib.pyplot as plt
import numpy as np
import json
#
perf1 = np.asarray([[771, 5534, 6500, 9600, 11240, 19720, 23120, 26530, 30000, 35000, 40000],
                    [0, 0.2, 0.8, 0.95, 1, 1, 1, 1, 1, 1, 1]])
perf2 = np.asarray([[775, 3579, 6500, 9600, 11240 , 19370, 22000, 24360, 30000, 35000, 40000],
                    [0, 0,   0.7,  0.95, 1, 1, 1, 1, 1, 1, 1]])
perf3 = np.asarray([[390, 3542, 6500, 9600, 11240, 14660, 21620, 23190, 30000, 35000, 40000],
                    [0, 0,   0.6, 0.89, 1, 1, 1, 1, 1, 1, 1]])
# perf1 = np.asarray([[57, 3677, 6528, 9044, 11203, 13389, 17781, 20174, 22509, 24607, 26665, 28806], [0.0, 0, 0, 0.0, 0.5, 0.7, 0.55, 0.9, 1, 1.0, 1.0, 1.0]])
# perf2 = np.asarray([[35, 4000, 6400, 9044, 11000, 14830, 17490, 22174, 22509, 24607, 26665, 28806], [0.0, 0, 0, 0.1, 0.6, 0.45, 0.95, 0.93, 1, 1.0, 1.0, 1.0]])
# perf3 = np.asarray([[688, 2600, 7400, 9044, 12000, 14200, 17000, 21074, 22509, 24607, 26665, 28806], [0.0, 0, 0.9, 1, 0.1, 0.8, 0.75, 1, 1, 1.0, 1.0, 1.0]])
#
perf = np.zeros((3, len(perf1[1])))
perf[0, :] = perf1[1]
perf[1, :] = perf2[1]
perf[2, :] = perf3[1]
plt.plot((perf1[0] + perf2[0] + perf3[0])/3, (perf1[1] + perf2[1] + perf2[1])/3)
std_performance = np.std(perf, axis=0)
avg_performance = (perf1[1] + perf2[1] + perf2[1])/3
eval_steps =(perf1[0] + perf2[0] + perf3[0])/3
settings = {}
settings['method_name'] = 'MuZero'
settings['num_datapoints'] = len(eval_steps)
settings['num_steps'] = 40000
settings['num_runs'] = 3

with open('data/' + settings['method_name'] + '_settings.txt', 'w') as json_file:
    json.dump(settings, json_file)
np.save('data/' + settings['method_name'] + '_avg_results.npy', avg_performance)
np.save('data/' + settings['method_name'] + '_std_results.npy', std_performance)
np.save('data/' + settings['method_name'] + '_eval_steps.npy', eval_steps)
#
plt.show()