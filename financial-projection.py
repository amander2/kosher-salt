import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n = int(input('The number of weeks you would like to project for is: '))  # number of projected weeks
p = int(input('The pre-launch boost is: '))  # pre-launch boost, or initial user count
g = int(input('The weekly growth rate is: '))* 0.01  # growth rate
r = (np.logspace(0.477, 0.773, n + 1)) * 0.1  # retention rate is logarithmic (base=10)

# Below are parameters for ARPU/Revenue Calculations
N = int(input('The projected number of weeks to stretch ARPU is: '))  # number of projected weeks for ARPU
t = np.arange(N + 1)  # Create x-axis (time in # of weeks)
L = 0  # Lower (y) asymptote;
U = int(input('The maximum ARPU value is: '))  # Upper (y) asymptote;
B = .03  # growth rate; between 0 & 5
v = .80  # affects near which asymptote maximum growth occurs
Q = 182  # related to the value Y(0)
c = 1

def Growth_Projection(p, n):  # Takes initial inputs for user population, \
    # growth rate, retention rate, and estimates user population after \
    # n number of months.
    G = np.zeros(n + 1)  # Initialize Growth Projection Array
    G[0] = p  # Initialize User Population
    for i in range(1, n + 1):
        #r = (np.random.choice(11, 1, p=[0, 0, 0, 0.55, 0.25, 0.15, 0.05, \
        #                                0, 0, 0, 0])) * 0.1  # randomized retention rate
        gr = g * r[i]  # growth rate * retention rate
        s = (np.random.choice(13, 1, p=[0, 0, 0, 0, 0, 0.55, 0, 0, 0.3 \
            , 0.1, 0.03, 0, 0.02])) * 0.01  # sharing rate
        sr = s * r[i]  # sharing rate * retention rate
        p = p + (p * gr) + ((p * gr) * sr)
        G[i] = p
    return G


print(f'The growth projection for {n} weeks is ...')
print()  # this line provides padding
print(Growth_Projection(p, n))
plt.plot(Growth_Projection(p, n))
plt.xticks(np.arange(0, n + 1, step=52))
plt.xlabel('Number of Weeks', color='green')
plt.ylabel('User Population', color='blue')
plt.title(f'User Growth Over {n}-Week Period', fontsize=16)
plt.savefig('growth_proj.png')
plt.show()
# establish dataframe for .csv export
df = pd.DataFrame(Growth_Projection(p, n))
df.to_csv('growth_proj.csv', header=False, index=True)


def GenLog(t, L, U, B, v, Q, C):  # Establish ARPU Rate over time (t)
    M = np.exp(-B * t)
    Y = np.zeros(N + 1)
    for i in range(N + 1):
        Y[i] = L + ((U - L) / ((C + (M[i] * Q)) ** (np.float(1 / v))))
    return Y


def Revenue_Projection(n, N):
    RevProj = np.zeros(n + 1)  # Initialize Array
    GP = Growth_Projection(p, n)
    ARPU = GenLog(t, L, U, B, v, Q, c)

    for j in range(n + 1):
        RevProj[j] = GP[j] * ARPU[j]
    return ARPU, RevProj


def Cost_Projection(n):
    x = np.arange(n + 1)  # initialize array
    CostProj = np.piecewise(x, [x < 0, x >= 0, x >= 26, x >= 52, x >= 104, x >= 182, x > 260, x > 312], \
                            [0, 1500, 51500, 551500, 1551500, 1551500, 4551500, 10051500])
    return CostProj


def Profit_Projection(R, C):
    Profit = np.zeros(n + 1)
    for i in range(n + 1):
        Profit[i] = R[i] - C[i]
    return Profit


def Kosher(n, N):
    if N < n:
        print()
        print('Error: unsuccessful revenue projection. Please choose appropriate N.')
    else:
        ARPU, R = Revenue_Projection(n, N)
        C = Cost_Projection(n)
        P = Profit_Projection(R, C)

        print()
        print(f'The Projected Revenue over {n} weeks is')
        print(R)
        plt.plot(R)
        plt.xticks(np.arange(0, n + 1, step=52))
        plt.xlabel('Number of Weeks', color='green')
        plt.ylabel('Gross Revenue', color='blue')
        plt.title(f'Revenue Over {n}-Week Period', fontsize=16)
        plt.savefig('revenue_proj.png')
        plt.show()
        # establish dataframe for .csv export
        # note, this is file path specific!!!
        df = pd.DataFrame(R)
        df.to_csv(f'revenue_proj_{n}_{g}_{N}.csv', header=False, index=True)

        print()
        print(f'The ARPU rate over the course of {N} weeks is...')
        print(ARPU)
        plt.plot(ARPU)
        plt.xticks(np.arange(0, N + 1, step=52))
        plt.xlabel('Number of Weeks', color='green')
        plt.ylabel('ARPU Rate', color='blue')
        plt.title(f'ARPU Over {N}-Week Period', fontsize=16)
        plt.savefig('ARPU_proj.png')
        plt.show()

        print()
        print(f'The Cost Projection over {n} weeks is')
        print(C)
        plt.plot(C)
        plt.xticks(np.arange(0, n + 1, step=52))
        plt.xlabel('Number of Weeks', color='green')
        plt.ylabel('Gross Cost', color='blue')
        plt.title(f'Cost Over {n}-Week Period', fontsize=16)
        plt.savefig('cost_proj.png')
        plt.show()

        print()
        print(f'The Projected Profit over {n} weeks is')
        print(P)
        plt.plot(P)
        plt.xticks(np.arange(0, n + 1, step=52))
        plt.xlabel('Number of Weeks', color='green')
        plt.ylabel('Net Profit', color='blue')
        plt.title(f'Net Profit Over {n}-Week Period', fontsize=16)
        plt.savefig('profit_proj.png')
        plt.show()
        # establish dataframe for .csv export
        # note, this is file path specific!!!
        df = pd.DataFrame(P)
        df.to_csv(f'profit_proj_{n}_{g}_{N}.csv', header=False, index=True)
    return R, ARPU, C, P


R, ARPU, C, P = Kosher(n, N)
