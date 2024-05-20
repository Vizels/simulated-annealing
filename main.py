from typing import List
import numpy as np
import math
import time
from matplotlib import pyplot as plt
import random


class Task:
    def __init__(self, id: int, tpm: List[int], tpm_sum: int) -> None:
        self.id = id
        self.tpm = tpm
        self.tpm_sum = tpm_sum
    
    def __repr__(self) -> str:
        return f"id: {self.id} | {[time for time in self.tpm]}"


def calculate_time(func):
    """
        Decorator to calculate total execution time of a function.
    """
    def inner(*args, **kwargs):
        import time
        start = time.time()
        order = func(*args, **kwargs)
        end = time.time()
        totalTime = end - start
        print(f"Execution time: {totalTime:.3} s")
        return order
        
    return inner


def readData(filepath: str) -> dict[str: List[Task]]:
    """
    Reads data from a file with sections defined by "data.XXX" lines.

    Args:
        - `filepath: str` - Path to the file containing the data.

    Returns:
        - `dict` - A dictionary where keys are section names ("data.XXX") and values are lists of lines within that section.
    """
    data = {}
    current_section = None
    counter = 0
    with open(filepath, 'r') as f:
        saveData = False
        for line in f:
            line = line.strip()
            if not line:
                saveData = False
                continue    

            if line.startswith("data."):
                saveData = True
                counter = 0
                current_section = line[:-1]
                data[current_section] = []
            else:
                if current_section and saveData:
                    if counter == 0:
                        counter += 1    
                        continue
                    tpm = [int(item) for item in line.split(" ")]
                    newTask = Task(counter, tpm, np.sum(tpm))
                    data[current_section].append(newTask)
                    counter += 1 
    return data


def getTotalTime(data):
    M = len(data[0].tpm)
    machine_time = np.zeros(M)
    Cmax = 0
    for task in data:
        task_frees_at = 0
        for m in range(M):
            entry_time = max(machine_time[m], task_frees_at)
            exit_time = entry_time + task.tpm[m]
            machine_time[m] = exit_time
            Cmax = task_frees_at = exit_time
    return int(Cmax)

def printOrder(order):
    print("Order: " + " ".join([str(i+1) for i in order]))


def testSolution(data, datasetName: str, func) -> None:
    data = np.asarray(data[datasetName])
    start = time.time()
    Cmax, order = func(data)
    end = time.time()
    totalTime = end - start
    print(f"{datasetName} {getTotalTime(data[order])} {totalTime:.5} s")
    printOrder(order)
    return totalTime
    
    
def testMultiple(data, func):
    total_time = 0
    for key in data:
        total_time += testSolution(data, key, func)
    print(f"Total time: {total_time} s")
    
    
def getRandomIndices(start, stop):
    f, s = 0, 0
    while f == s:
        f, s = random.randint(start, stop), random.randint(start, stop)
    return f, s


def calculateMinMaxDelta(data, n_iter=1000):
    deltas = []
    N = len(data)
    order = [task.id-1 for task in data]
    Cmax = getTotalTime(data)
    for _ in range(n_iter):
        f, s = getRandomIndices(0, N-1)
        order[f], order[s] = order[s], order[f]
        new_cmax = getTotalTime(data[order])
        delta = abs(Cmax - new_cmax)
        if delta != 0:
            deltas.append(delta)
    return min(deltas), max(deltas)


def simulatedAnnealing(data, n_iter=100000, maxP=0.5, minP=0.1, n_temp_changes=None):
    data = np.array(data)
    N = len(data)
    order = [task.id-1 for task in data]
    Cmax = getTotalTime(data)
    y = []
    # temperatures = []
    
    minDelta, maxDelta = calculateMinMaxDelta(data)
    
    t_start = -maxDelta/math.log(maxP)
    t_stop = -minDelta/math.log(minP)

    update_t_iters = 1 if n_temp_changes is None else (n_iter // n_temp_changes)
    
    if n_temp_changes is None: 
        alpha = math.pow(t_stop/t_start, 1/(n_iter-1))
    else:
        alpha = math.pow(t_stop/t_start, 1/(n_temp_changes-1))
    
    print(f"alpha: {alpha}")
    t = t_start
    print(f"start T = {t_start}")
    print(f"end T = {t_stop}")
    

    for i in range(1, n_iter+1):
        f, s = getRandomIndices(0, N-1)
        order[f], order[s] = order[s], order[f]
        new_cmax = getTotalTime(data[order])
        delta = new_cmax - Cmax
        if delta < 0:
            Cmax = new_cmax
        else:
            probabilty = math.exp(-delta/t)
            if random.random() < probabilty:
                Cmax = new_cmax
            else:
                order[f], order[s] = order[s], order[f]

        # temperatures.append(t)
        if (i % update_t_iters == 0):
            t *= alpha

        if i%100 == 0:
            y.append(Cmax)

    print(f"Real end T = {t}")
    x = np.arange(len(y))

    plt.plot(x, y)
    # plt.plot(np.arange(len(temperatures)), temperatures)
    # plt.plot(np.arange(len(temperatures)), temperatures)
    plt.show()
    
    return Cmax, order


def main():
    data = readData("data/data.txt")
    data = data["data.001"] 
    print(f"Total time: {getTotalTime(data)}")  
    print(f"Simulated annealing Cmax: {simulatedAnnealing(data, n_iter = 100000, n_temp_changes=4)}") 
    

if __name__ == "__main__":
    main()

