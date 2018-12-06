import numpy as np 
import matplotlib.pyplot as plt 
import scipy.integrate as integrate 
from mpl_toolkits.mplot3d import Axes3D 
from scipy import optimize

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

#Define the initial condition in space
y0 = np.array([1.5,-0.2,-0.1], dtype = "float64")

#Define the time interval of the 
timeStop = 5000
timeDivision = 0.1

<<<<<<< HEAD
def attractorSolution(ODE, timeStop, timeStep, y0):
=======
def attractorSolution(ODE, timeStop, timeStep):
>>>>>>> fbcccfddb06a42f67b89db8238be09cac8f72609
    timeSpan = (0, timeStop)
    timeEvaluations = np.linspace(0, timeStop, int(timeStop/timeDivision)+1)

    attractorTime = (0, 5)

    transientSolution = integrate.solve_ivp(chuasODE, attractorTime, y0, method = "LSODA")
    odeSolution = integrate.solve_ivp(chuasODE, timeSpan, transientSolution.y[:,len(transientSolution.t)-1], method = "LSODA", t_eval= timeEvaluations)
    return odeSolution

<<<<<<< HEAD
attractor = attractorSolution(chuasODE, timeStop, timeDivision, y0)
=======
attractor = attractorSolution(chuasODE, timeStop, timeDivision)
>>>>>>> fbcccfddb06a42f67b89db8238be09cac8f72609

fig = plt.figure()
axes = fig.add_subplot(111, projection = "3d")
axes.plot(attractor.y[0], attractor.y[1], attractor.y[2])
axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_zlabel("z")
plt.savefig('./Figures/attractor.pdf')
plt.show()

def boxCounting(epsilon, data):
    "Calculate the box count for a given epsilon"
    xMin = data[0].min()
    xMax = data[0].max()
    yMin = data[1].min()
    yMax = data[1].max()
    zMin = data[2].min()
    zMax = data[2].max()
    xCurrent = xMin
    yCurrent = yMin
    zCurrent = zMin

    #Stack the arrays to form an array of data points
    dataPoints = np.stack((data[0], data[1], data[2]), axis =1)

    count = 0
    #Iterate over the grid
    while(zCurrent<zMax):
        while(yCurrent<yMax):
            while(xCurrent<xMax):
                for point in dataPoints:
                    # check if the box contains any points
                    if (point[0] >= xCurrent and point[0] < xCurrent + epsilon):
                        if (point[1] >= yCurrent and point[0] < yCurrent + epsilon):
                            if (point[2] >= zCurrent and point[2] < zCurrent + epsilon):
                                count += 1
                                dataPoints = dataPoints[dataPoints != point]
                                break
                xCurrent += epsilon
            xCurrent = xMin
            yCurrent += epsilon
        yCurrent = yMin
        zCurrent +=epsilon
    
    return count

def informationCount(epsilon, data):
    "Count the information dimension for a given epsilon"
    xMin = data[0].min()
    xMax = data[0].max()
    yMin = data[1].min()
    yMax = data[1].max()
    zMin = data[2].min()
    zMax = data[2].max()
    xCurrent = xMin
    yCurrent = yMin
    zCurrent = zMin

    #Stack the arrays to form an array of data points
    dataPoints = np.stack((data[0], data[1], data[2]), axis =1)
    numPoints = len(dataPoints)

    measure = 0
    #Iterate over the grid
    while(zCurrent<zMax):
        while(yCurrent<yMax):
            while(xCurrent<xMax):
                count = 0
                for point in dataPoints:
                    # check if the box contains any points
                    if (point[0] >= xCurrent and point[0] < xCurrent + epsilon):
                        if (point[1] >= yCurrent and point[0] < yCurrent + epsilon):
                            if (point[2] >= zCurrent and point[2] < zCurrent + epsilon):
                                count += 1
                                dataPoints = dataPoints[dataPoints != point]
                # Add the measure content for the box
                if(count != 0):
                    measure += count/numPoints*np.log(count/numPoints)
                xCurrent += epsilon
            xCurrent = xMin
            yCurrent += epsilon
        yCurrent = yMin
        zCurrent +=epsilon
    
    return measure 

def euclideanDistance(point1, point2):
    "Find the euclidean distance between two data points"
    return np.sqrt(np.sum(np.square(point1-point2)))

def correlationIntegral(epsilon, data):
    "Calculate the correlation integral for a given epsilon"
    
    #Stack the arrays to form an array of data points
    dataPoints = np.stack((data[0], data[1], data[2]), axis =1)
    numPoints = len(dataPoints)

    count = 1
    for i in range(numPoints-100):
            for j in range(i+100, numPoints):
                if (euclideanDistance(dataPoints[i], dataPoints[j]) < epsilon):
                    count += 1

    return count/np.square(numPoints)

epsilons = np.logspace(-4, 0, num = 10)

logEpsilons = np.log(epsilons)

fitPoints = np.log(np.logspace(-2, 0))

def fitfunc(p, x):
    return p[0]*x+p[1]

def linearModel(p, x, y):
    return fitfunc(p, x) - y

'''
# Box counting dimension
boxCounts = np.array([boxCounting(epsilon, attractor.y) for epsilon in epsilons])

p01 = [1, 0]
pf1, cov1, info1, mesg1, success1 = optimize.leastsq(linearModel, p01,
                                     args = (-logEpsilons, boxCounts), full_output=1)

if cov1 is None:
    print('Fit did not converge')
    print('Success code:', success1)
    print(mesg1)
else:
    print('Fit Converged')
    chisq1 = np.sum(np.square(info1['fvec']))
    dof1 = len(logEpsilons)-len(pf1)
    pferr1 = [np.sqrt(cov1[i,i]) for i in range(len(pf1))]
    print('Converged with chi-squared', chisq1)
    print('Number of degrees of freedom, dof =',dof1)
    print('Reduced chi-squared', chisq1/dof1)
    print('Inital guess values:')
    print('  p0 =', p01)
    print('Best fit values:')
    print('  pf =', pf1)
    print('Uncertainties in the best fit values:')
    print('  pferr =', pferr1)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(-logEpsilons, boxCounts, "r.", label = "\\tilde{N}(\\epsilon)")
ax.plot(-fitPoints, fitfunc(pf1, -fitPoints), "b-", label = "Linear Fit")
ax.set_title("Calculation of Box Counting Dimension")
ax.set_ylabel("Box Count, $\\tilde{N}(\\epsilon)$")
ax.set_xlabel("$\\ln (1/\\epsilon)$")
ax.legend()

textfit1 = '$\\tilde{N}(\\epsilon) = D_0 \\epsilon + \\alpha$ \n' \
                '$D_0 = %.2f \pm %.2f$ \n' \
                '$\\alpha = %0.2f \pm %0.2f$  \n' \
                '$\chi^2= %.2f$ \n' \
                '$N = %i$ (dof) \n' \
                '$\chi^2/N = % .2f$' \
                % (pf1[0], pferr1[0], pf1[1], pferr1[1], chisq1, dof1,
                   chisq1/dof1)
ax.text(0.55, .35, textfit1, transform=ax.transAxes, fontsize=10,
             verticalalignment='top')

plt.savefig('./Figures/BoxCounting.pdf')
plt.show()


#Information Dimension D_1
informationEstimate = np.array([informationCount(epsilon, attractor.y) for epsilon in epsilons])

p02 = [1, 0]
pf2, cov2, info2, mesg2, success2 = optimize.leastsq(linearModel, p02,
                                     args = (logEpsilons, informationEstimate), full_output=1)

if cov2 is None:
    print('Fit did not converge')
    print('Success code:', success2)
    print(mesg2)
else:
    print('Fit Converged')
    chisq2 = np.sum(np.square(info2['fvec']))
    dof2 = len(logEpsilons)-len(pf2)
    pferr2 = [np.sqrt(cov2[i,i]) for i in range(len(pf2))]
    print('Converged with chi-squared', chisq2)
    print('Number of degrees of freedom, dof =',dof2)
    print('Reduced chi-squared', chisq2/dof2)
    print('Inital guess values:')
    print('  p0 =', p02)
    print('Best fit values:')
    print('  pf =', pf2)
    print('Uncertainties in the best fit values:')
    print('  pferr =', pferr2)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(logEpsilons, informationEstimate, "r.", label = "$\\sum \\mu \\ln \\mu")
ax.plot(fitPoints, fitfunc(pf2, fitPoints), "b-", label = "Linear Fit")
ax.set_title("Calculation of Information Dimension")
ax.set_ylabel("Information Estimate")
ax.set_xlabel("$\\ln \\epsilon$")
ax.legend()

textfit2 = '$\\sum \\mu \\ln \\mu = D_1 \\epsilon + \\alpha$ \n' \
                '$D_1 = %.2f \pm %.2f$  \n' \
                '$\\alpha = %0.2f \pm %0.2f$  \n' \
                '$\chi^2= %.2f$ \n' \
                '$N = %i$ (dof) \n' \
                '$\chi^2/N = % .2f$' \
                % (pf2[0], pferr2[0], pf2[1], pferr2[1], chisq2, dof2,
                   chisq2/dof2)
ax.text(0.55, .35, textfit2, transform=ax.transAxes, fontsize=10,
             verticalalignment='top')

plt.savefig('./Figures/Information.pdf')
plt.show()
'''

#Correlation Integral
correlationInt = np.array([correlationIntegral(epsilon, attractor.y) for epsilon in epsilons])


p03 = [1, 0]
pf3, cov3, info3, mesg3, success3 = optimize.leastsq(linearModel, p03,
                                     args = (logEpsilons, correlationInt), full_output=1)

if cov3 is None:
    print('Fit did not converge')
    print('Success code:', success3)
    print(mesg3)
else:
    print('Fit Converged')
    chisq3 = np.sum(np.square(info3['fvec']))
    dof3 = len(logEpsilons)-len(pf3)
    pferr3 = [np.sqrt(cov3[i,i]) for i in range(len(pf3))]
    print('Converged with chi-squared', chisq3)
    print('Number of degrees of freedom, dof =',dof3)
    print('Reduced chi-squared', chisq3/dof3)
    print('Inital guess values:')
    print('  p0 =', p03)
    print('Best fit values:')
    print('  pf =', pf3)
    print('Uncertainties in the best fit values:')
    print('  pferr =', pferr3)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(logEpsilons, correlationInt, "r.", label = "$C(\\epsilon)$")
ax.plot(fitPoints, fitfunc(pf3, fitPoints), "b-", label = "Linear Fit")
ax.set_title("Calculation of Correlation Dimension")
ax.set_ylabel("Correlation Integral")
ax.set_xlabel("$\\ln \\epsilon$")
ax.legend()

textfit3 = '$C(\\epsilon) = D_2 \\epsilon + \\alpha$ \n' \
                '$D_2 = %.2f \pm %.2f$  \n' \
                '$\\alpha = %0.2f \pm %0.2f$  \n' \
                '$\chi^2= %.2f$ \n' \
                '$N = %i$ (dof) \n' \
                '$\chi^2/N = % .2f$' \
                % (pf3[0], pferr3[0], pf3[1], pferr3[1], chisq3, dof3,
                   chisq3/dof3)
ax.text(0.25, .85, textfit3, transform=ax.transAxes, fontsize=10,
             verticalalignment='top')

plt.savefig('./Figures/Correlation.pdf')
plt.show()

