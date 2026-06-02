import numpy as np
import matplotlib.pyplot as plt

NUMEXPERIMENTS = 10000
CORRECTSUCCESPROB = 7/19
WRONGSUCCESPROB = 2/57
def main():
    successList = []
    xAxis = range(5, 100, 2)
    for M in xAxis:
        successes = 0
        for _ in range(NUMEXPERIMENTS):
            trueCount = np.random.binomial(n=M, p=CORRECTSUCCESPROB)
            falseMaxCount = np.max(np.random.binomial(n=M, p=WRONGSUCCESPROB, size=18))
            if trueCount > falseMaxCount:
                successes += 1
        print(f"M: {M}, success rate: {successes/NUMEXPERIMENTS}")
        successList.append(successes/NUMEXPERIMENTS)
    plt.plot(xAxis, successList)
    plt.xlabel("M")
    plt.ylabel("Success Rate")
    plt.title("Success Rate vs M")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()


            