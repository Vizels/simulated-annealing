from typing import List
import numpy as np
import math
import time


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
    order = func(data)
    end = time.time()
    totalTime = end - start
    print(f"{datasetName} {getTotalTime(data[order])} {totalTime:.5} s")
    return totalTime
    

def testMultiple(data, func):
    total_time = 0
    for key in data:
        total_time += testSolution(data, key, func)
    print(f"Total time: {total_time} s")
    

def simulatedAnnealing(data):
    data = np.array(data)
    N = len(data) 
    order = [task.id-1 for task in data]
    Cmax = getTotalTime(data)
    import random
    for _ in range(100000):
        f, s = random.randint(0, N-1), random.randint(0, N-1)
        order[f], order[s] = order[s], order[f]
        new_cmax = getTotalTime(data[order])
        if new_cmax < Cmax:
            Cmax = new_cmax
        else:
            order[f], order[s] = order[s], order[f]
    return Cmax


def main():
    # dataName = "data.100"
    data = readData("data/data.txt")
    data = data["data.001"] 
    print(f"Total time: {getTotalTime(data)}")  
    print(f"Simulated annealing Cmax: {simulatedAnnealing(data)}") 
    
if __name__ == "__main__":
    main()
    