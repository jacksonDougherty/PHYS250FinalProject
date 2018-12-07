import numpy as np 
import matplotlib.pyplot as plt 
import scipy.linalg as linalg 
import scipy.integrate as integrate 


def nonLinear(x):
    "Define the non-linear circuit element"
    m0 = -1/7
    m1 = 2/7
    return m1*x+0.5*(m0-m1)*(np.abs(x+1)-np.abs(x-1))

def chuasODE(t, y):
    "Provide the non-dimensionalized circui ODEs"
    alpha = 9
    beta = 14+2/7
    return alpha*(y[1]-nonLinear(y[0])), y[0]-y[1]+y[2], -beta*y[1]

def chuasHessian(t, y):
    "Provide the Hessian of the ODEs"
    alpha = 9
    beta = 14+2/7
    m0 = -1/7
    m1 = 2/7
    return np.array([[alpha*(0.5*(m0-m1)*((y[0]+1)/np.abs(y[0]+1)-(y[0]-1)/np.abs(y[0]-1))+m1), alpha, 0], [1,-1,1],[0, -beta, 0]], dtype='float64')

#Define the initial condition in space
y0 = np.array([1.5,-0.2,-0.1], dtype = "float64")

#Define the time interval of the 
timeStop = 20
timeDivision = 0.1

def attractorSolution(ODE, timeStop, timeStep, y0):
    timeSpan = (0, timeStop)
    timeEvaluations = np.linspace(0, timeStop, int(timeStop/timeDivision)+1)

    attractorTime = (0, 5)

    transientSolution = integrate.solve_ivp(chuasODE, attractorTime, y0, method = "LSODA")
    odeSolution = integrate.solve_ivp(chuasODE, timeSpan, transientSolution.y[:,len(transientSolution.t)-1], method = "LSODA", t_eval= timeEvaluations)
    return odeSolution

attractor = attractorSolution(chuasODE, timeStop, timeDivision, y0)

def Hny0(hessian, attractor):
    'return DM^n(y0).T DM^n(y0)'
    
    #Stack the arrays to form an array of data points
    dataPoints = np.stack((attractor.y[0], attractor.y[1], attractor.y[2]), axis =1)

    DMn = np.identity(3)
    for i, data in enumerate(dataPoints):
        DMn = np.matmul(hessian(attractor.t[i], data), DMn)

    return np.matmul(DMn.T, DMn)

def lyupanov(hessian, attractor):
    "Return an array of lyupanov exponents"
    H = Hny0(chuasHessian, attractor)
    
    numPoints = len(attractor.t)
    eigenValues , eigenVectors = linalg.eigh(H, type = 1)

    lam =  np.array([np.matmul(np.matmul(eigenVectors[:,i].T, H), eigenVectors[:, i]) for i in range(len(eigenVectors))])
    return np.log(lam)/(2*numPoints)

h = lyupanov(chuasHessian, attractor)
print(h)