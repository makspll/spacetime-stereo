from scipy import stats 
import argparse 
import csv 
import numpy as np
import glob 
import os 

ARGS = argparse.ArgumentParser()
ARGS.add_argument('column',type=int)
ARGS.add_argument('-system1',nargs="+",required=True)
ARGS.add_argument('-system2',nargs="+",required=True)


if __name__ == "__main__":
    ARGS = ARGS.parse_args()

    system1_paths = ARGS.system1
    system2_paths = ARGS.system2

    average_readings = [[],[]]
    for i,system_paths in enumerate([system1_paths,system2_paths]):
        readings = []
        for s in system_paths:
            readings.append([])
            print(s)
            file = s if os.path.isfile(s) else  glob.glob(f'{s}/**/*results.csv',recursive=True)[0]
            with open(file ,'r')as f:
                reader = csv.reader(f)
                next(iter(reader))
                for r in reader:
                    if r[0] == 'AVG':
                        continue 
                    
                    readings[-1].append(float(r[ARGS.column]))

        for j in range(len(readings[0])):
            average_readings[i].append(
                sum([readings[y][j] for y in range(len(readings))]) / len(readings))

    print("Ranksums Wilcox (unpaired): ",stats.ranksums(average_readings[0],average_readings[1]))
    print("signed-rank Wilcox (paired): ",stats.wilcoxon(average_readings[0],average_readings[1]))
    print("System 1, mean, std")
    print(np.mean(average_readings[0]), np.std(average_readings[0]))
    print("System 2, mean, std")
    print(np.mean(average_readings[1]), np.std(average_readings[1]))

    import seaborn
    import matplotlib.pyplot as plt 

    ax = seaborn.histplot(data=average_readings[0],kde=True,color='blue',legend=True)
    seaborn.histplot(data=average_readings[1],ax=ax,kde=True,color='red',legend=True)
    plt.legend(labels=['system 1','system 2'])
    plt.show()
