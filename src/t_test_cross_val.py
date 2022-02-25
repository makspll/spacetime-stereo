from math import ceil, floor
from scipy import stats 
import argparse 
import csv 
import numpy as np
import glob 
import os 

ARGS = argparse.ArgumentParser()
ARGS.add_argument('column',type=int)
ARGS.add_argument('-per_sample',action="store_const", const=True,default=False, help="if true will calculate over granuality of each sample as opposed to whole split")
ARGS.add_argument('-system1',nargs="+",required=True,help="list of paths to each consecutive cross validation result rile")
ARGS.add_argument('-system2',nargs="+",required=True,help="list of paths to each consecutive cross validation result rile")


if __name__ == "__main__":
    ARGS = ARGS.parse_args()

    system1_paths = ARGS.system1
    system2_paths = ARGS.system2
    labels = []
    average_readings = [[],[]] # each list is a list of paired measurements for each system (2 systems)
    for i,system_paths in enumerate([system1_paths,system2_paths]):
        for s in system_paths:
            readings = []
            file = s if os.path.isfile(s) else glob.glob(f'{s}/**/*results.csv',recursive=True)[0] # can be incomplete path
            with open(file ,'r') as f:
                reader = csv.reader(f)
                next(iter(reader))
                for r in reader:
                    # for each sample, store value of picked column
                    if r[0] == 'AVG':
                        continue 
                    
                    readings.append(float(r[ARGS.column]))
                    if ARGS.per_sample:
                        labels.append(r[0])
            # average for each paired system paths
            if ARGS.per_sample:
                average_readings[i] += readings
            else:
                labels.append(str(i))
                average_readings[i].append(np.mean(readings))
        
    print(average_readings)
    print("Ranksums Wilcox (unpaired): ",stats.ranksums(average_readings[0],average_readings[1]))
    print("signed-rank Wilcox (paired): ",stats.wilcoxon(average_readings[0],average_readings[1]))
    print("System 1, mean, std")
    print(np.mean(average_readings[0]), np.std(average_readings[0]))
    print("System 2, mean, std")
    print(np.mean(average_readings[1]), np.std(average_readings[1]))

    import seaborn
    import matplotlib.pyplot as plt 

    diffs = [a-b for a,b in zip(average_readings[0],average_readings[1])]
    ordered_diffs  = np.argsort(diffs)

    print("Samples ordered by biggest difference (s1 - s2):")

    max_print= 5
    for i in ordered_diffs:
        print(f"{labels[i]} - difference: {diffs[i]:.3f}, s1: {average_readings[0][i]:.3f}, s2: {average_readings[1][i]}:.3f")
        if max_print <= 0:
            break 
        else:
            max_print -= 1

    ax = seaborn.histplot(data=diffs,kde=True,color='blue',legend=True,bins=[x for x in np.linspace(floor(np.min(diffs))-1,ceil(np.max(diffs)+1),num=40)])
    plt.vlines(np.mean(diffs),0,ax.get_ylim()[1],colors='red',label='mean')
    plt.legend(labels=['sys1 - sys2','mean'])
    plt.show()
