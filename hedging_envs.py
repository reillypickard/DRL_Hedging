

import numpy as np
import gym
from scipy.stats import norm
import csv

import numpy as np
import Cheb1DApprox as cheb1d
import matplotlib.pyplot as plt
import AmericanOptionPricer_Binomial as aop
import EvolutionModels as em
import optionPayoff as op
from enum import Enum
import Eta as probConveyor
class FitMethod(Enum):
    numpy = 0
    scipy = 1
    regression = 2
    series = 3
    nodes = 4
    numpy_weights = 5
    barycentric1 = 6
    barycentric2 = 7
    mocax = 8
    finn = 9

class EnvelopMethod(Enum):
    theo_percentiles = 0
    brute_force = 1
    theo_evolved = 2
    brute_force_SABR = 3


class NodeBoundaryMethod(Enum):
    Naiive = 0
    SminToZero = 1
    CarryBackExtremes = 2
    Dynamic = 3
    SmintoZeroSmaxToMax = 4
    SminToZeroSmaxAsPercentile = 5
    ProportionalToBoundary = 6
    ProportionalToBoundaryCapped = 7
    UserDefined = 8
    ProportionalToBoundaryMAXOUT=9
def calcNodeBoundaries(path_envelope_method, nodeBNDMeth, evolution_model, numNodesMax, numNodesMin,
                       numEnvelopeScenarios, numTimeSteps, T, eta, node_density, SimulationModel, doPlotNodeEnvelop):
    '''
    :param path_envelope_method: class EnvelopMethod provides three options
    :param nodeBNDMeth: class NodeBoundaryMethod provides three options
    :param evolution_model: class that points to the evolution model
    :param numNodesMax:
    :param numNodesMin:
    :param numEnvelopeScenarios: number of paths used to calculate the max and min stock value at each time step
    :param numTimeSteps: number of time steps used to simulate the evolution model
    :param T: maturity of the option
    :param eta: class that is used to pass parameters
    :param doPlotNodeEnvelop: boolean (defaulted to False) that activates the plotting of the node envelop

    :return:
    :param Smin: minimum path envelope
    :param Smax: maximum path envelope
    :param numNodes: vector, number of nodes used at each time step
    :param tv: vector of times used to simulate the evolution model
    :param SimulationModel: string that stores name of evolution model
    '''
    assert (numNodesMin <= numNodesMax)
    assert (0 < numNodesMin)
    dt = T / (numTimeSteps)
    tv = [dt * j for j in range(0, numTimeSteps + 1)]
    Smin = np.zeros(numTimeSteps + 1)
    Smax = np.zeros(numTimeSteps + 1)
    # Here we choose either a percentile approach to calculate the node envelop
    # or, in the case that the process does not have a closed-form, we will need
    # to simulate and estimate the node envelop (brute force), or,
    # we will use the evolution model but set the probability of an upwards and downwards
    # movement via the eta class.
    if path_envelope_method == EnvelopMethod.theo_percentiles:  # use percentiles to estimate nodal boundaries
        Smin, Smax = evolution_model.percentileLowerUpper(eta)
    elif path_envelope_method == EnvelopMethod.brute_force:  # use simulated paths to estimate nodal boundaries
        if SimulationModel == 'GBM':
            for i in range(0, numEnvelopeScenarios):
                S = evolution_model.evolveFP(evolution_model.S0)
                for j in range(0, len(Smin)):
                    if Smin[j] > S[j]:
                        Smin[j] = S[j]
                    if Smax[j] < S[j]:
                        Smax[j] = S[j]
        if SimulationModel == 'SABR':
            S = evolution_model.evolveFP(evolution_model.S0, numEnvelopeScenarios)
            for i in range(0, numEnvelopeScenarios):
                for j in range(0, len(Smin)):
                    if Smin[j] > S[i][j]:
                        Smin[j] = S[i][j]
                    if Smax[j] < S[i][j]:
                        Smax[j] = S[i][j]

    elif path_envelope_method == EnvelopMethod.brute_force_SABR:  # use simulated paths to estimate nodal boundaries
        seed = 1
        Paths = evolution_model.evolveFP(numEnvelopeScenarios, seed)
        Smin = np.zeros(numTimeSteps + 1)
        Smax = np.zeros(numTimeSteps + 1)
        for S in Paths:
            for j in range(0, len(Smin)):
                if Smin[j] > S[j]:
                    Smin[j] = S[j]
                if Smax[j] < S[j]:
                    Smax[j] = S[j]

    elif path_envelope_method == EnvelopMethod.theo_evolved:
        Smin, Smax = evolution_model.percentileEvolvedLowerUpper(eta)

    else:
        raise ("envelop calculation option not recognized")

    if nodeBNDMeth == NodeBoundaryMethod.Naiive:  # naiive implementation - Set Smin, Smax = Smin[-1], Smax[-1]
        Smin = [Smin[-1] for i in range(0, len(Smin))]
        Smax = [Smax[-1] for i in range(0, len(Smax))]
    elif nodeBNDMeth == NodeBoundaryMethod.SminToZero:  # set Smin, Smax = [0, Smax[-1]]
        Smin = [0 for j in range(0, len(Smin))]
        Smax = [Smax[-1] for j in range(0, len(Smax))]
    elif nodeBNDMeth == NodeBoundaryMethod.SminToZeroSmaxAsPercentile:  # set Smin, Smax = [0, Smax[-1]]
        Smin = [1e-6 for j in range(0, len(Smin))]
    elif nodeBNDMeth == NodeBoundaryMethod.CarryBackExtremes:  # dynamic implementation + adjustment to envelop S0
        if Smin[-1] > evolution_model.S0:
            Smin = [evolution_model.S0 for j in range(0, numTimeSteps)]
        elif Smax[-1] < evolution_model.S0:
            Smax = [evolution_model.S0 for j in range(0, numTimeSteps)]
    elif nodeBNDMeth == NodeBoundaryMethod.Dynamic:  # dynamic implementation + adjustment to envelop S0
        Smin = Smin
        Smax = Smax
    elif nodeBNDMeth == NodeBoundaryMethod.SmintoZeroSmaxToMax:  # do any of the above - whichever is preferable
        Smin = [0 for j in range(0, len(Smin))]
        val = max(Smax[-1], evolution_model.S0)
        Smax = [val for j in range(0, len(Smax))]

    elif nodeBNDMeth == NodeBoundaryMethod.ProportionalToBoundary:
        numNodes = [int((Smax[j] - Smin[j]) * node_density) for j in range(0, len(Smin))]
        numNodes[0] = 1
        for j in range(1, len(numNodes)):
            if numNodes[j] < numNodesMin:
                numNodes[j] = numNodesMin

    elif nodeBNDMeth == NodeBoundaryMethod.ProportionalToBoundaryMAXOUT:
        stepsize = (Smax[-1] - Smin[-1])/(numNodesMax-1)
        numNodes = [int(numNodesMin+stepsize*j) for j in range(0, len(Smin))]

    elif nodeBNDMeth == NodeBoundaryMethod.ProportionalToBoundaryCapped:
        maxS = -1e99
        for j in range(0, len(Smin)):
            if Smax[j] - Smin[j] > maxS:
                maxS = Smax[j] - Smin[j]

        numNodes = [int((Smax[j] - Smin[j]) * node_density / maxS * numNodesMax) for j in range(0, len(Smin))]
        numNodes[0] = 1
        for j in range(1, len(numNodes)):
            if numNodes[j] < numNodesMin:
                numNodes[j] = numNodesMin
    elif nodeBNDMeth == NodeBoundaryMethod.UserDefined:
        pass  # do nothing - user has specified how many nodes to use for each time step
    else:
        raise ("Unrecognized node boundary method")
    if not (
            nodeBNDMeth == NodeBoundaryMethod.ProportionalToBoundary or nodeBNDMeth == NodeBoundaryMethod.ProportionalToBoundaryCapped or nodeBNDMeth == NodeBoundaryMethod.UserDefined):
        d = [abs(Smax[j] - Smin[j]) for j in range(0, numTimeSteps)]
        dmin = min(d)
        dmax = max(d)
        if dmax > dmin:
            numNodes = [int((numNodesMax - numNodesMin) / (dmax - dmin) * (d[j] - dmin) + numNodesMin) for j in
                        range(0, numTimeSteps)]
        else:
            numNodes = [int(numNodesMin) for j in range(0, numTimeSteps)]

        if dmin == 0:
            numNodes[0] = 1

    #plotNodeBoundaries(doPlotNodeEnvelop, tv, Smin, Smax, evolution_model.S0)

    # check is Smin < Smax, except at t=0
    for j in range(0, len(Smin)):
        if j == 0 and Smin[j] > Smax[j]:
            CRED = '\033[91m'
            CEND = '\033[0m'
            msg = "Error: Envelope boundaries are incorrect : Smin[" + str(j) + "]=" + str(Smin[j]) + " >= Smax[" + str(
                j) + "]=" + str(Smax[j]) + " is not permitted!"
            print(CRED + msg + CEND)
            assert (Smin[j] <= Smax[j])
        if j > 0 and Smin[j] >= Smax[j]:
            CRED = '\033[91m'
            CEND = '\033[0m'
            msg = "Error: Envelope boundaries are incorrect : Smin[" + str(j) + "]=" + str(Smin[j]) + " >= Smax[" + str(
                j) + "]=" + str(Smax[j]) + " is not permitted!"
            print(CRED + msg + CEND)
            assert (Smin[j] < Smax[j])

    return Smin, Smax, numNodes, tv

import gc

import numpy as np
import Cheb1DApprox as cheb1d
import matplotlib.pyplot as plt
import AmericanOptionPricer_Binomial as aop
import EvolutionModels as em
import optionPayoff as op
from enum import Enum
import Eta as probConveyor
import os
import random
import time
import numpy as np
import matplotlib.pyplot as plt

import BartcentricChebyshev as barychev
#import ChebDLL as cdll
os.system("")




class FitMethod(Enum):
    numpy = 0
    scipy = 1
    regression = 2
    series = 3
    nodes = 4
    numpy_weights = 5
    barycentric1 = 6
    barycentric2 = 7
    mocax = 8
    finn = 9


def myfunc(x):
    return mox.math.sin(x)


def testMOCAX():
    dim = 1
    variable_range = [[-1, 1]]
    domain = mox.MocaxDomain(variable_range)
    num_points = [10]
    max_deriv_order = 0
    mcpn = mox.MocaxNs(num_points)
    cheb_obj = mox.Mocax(None, num_dimensions=dim, domain=domain, error_threshold=None, n=mcpn,
                         max_derivative_order=max_deriv_order)

    nodes = cheb_obj.get_evaluation_points()
    print(nodes)

    xv = nodes
    yv = [myfunc(x[0]) for x in xv]

    cheb_obj.set_original_function_values(yv)
    derivative_value_id = cheb_obj.get_derivative_id([0])
    cheb_y = [cheb_obj.eval(x, derivative_value_id) for x in xv]
    cheb_y2 = [cheb_obj.eval(xv, derivative_value_id)]
    # test
    plt.plot(xv, yv, 'r+', label='Function Y')
    plt.plot(xv, cheb_y, color='black', label='Cheb Y')
    plt.show()


class EnvelopMethod(Enum):
    theo_percentiles = 0
    brute_force = 1
    theo_evolved = 2
    brute_force_SABR = 3


class NodeBoundaryMethod(Enum):
    Naiive = 0
    SminToZero = 1
    CarryBackExtremes = 2
    Dynamic = 3
    SmintoZeroSmaxToMax = 4
    SminToZeroSmaxAsPercentile = 5
    ProportionalToBoundary = 6
    ProportionalToBoundaryCapped = 7
    UserDefined = 8
    ProportionalToBoundaryMAXOUT=9


def plotNodeBoundaries(doPlotNodeBoundaries, tv, Smin, Smax, S0):
    if doPlotNodeBoundaries:
        plt.plot(tv, Smin)
        plt.plot(tv, Smin, '*b')
        plt.plot(tv, Smax)
        plt.plot(tv, Smax, '*g')
        plt.plot(tv, [S0] * len(tv), 'r.-')
        # plt.legend(str(config))
        plt.show()


def doPlot(cheb, v, t, doPlotChebyshevFit):
    if not doPlotChebyshevFit:
        return

    a = cheb.a
    b = cheb.b
    if a == b:
        return
    n = len(cheb.fnodes) * 50
    dn = (b - a) / n

    x = [a + i * dn for i in range(0, n)]
    y = [cheb.eval(x[i]) for i in range(0, n)]

    if len(v) == 1:
        v = v[0] * len(cheb.fnodes)

    plt.plot(x, y, 'b-', label='cheb')
    plt.plot(cheb.fnodes, v, 'r.')
    plt.ylabel("value")
    plt.xlabel("Stock Level")
    plt.legend()
    plt.title("Value vs Stock Level @ t = " + str(t))
    plt.show()


def calcNodeBoundaries(path_envelope_method, nodeBNDMeth, evolution_model, numNodesMax, numNodesMin,
                       numEnvelopeScenarios, numTimeSteps, T, eta, node_density, SimulationModel, doPlotNodeEnvelop):
    '''
    :param path_envelope_method: class EnvelopMethod provides three options
    :param nodeBNDMeth: class NodeBoundaryMethod provides three options
    :param evolution_model: class that points to the evolution model
    :param numNodesMax:
    :param numNodesMin:
    :param numEnvelopeScenarios: number of paths used to calculate the max and min stock value at each time step
    :param numTimeSteps: number of time steps used to simulate the evolution model
    :param T: maturity of the option
    :param eta: class that is used to pass parameters
    :param doPlotNodeEnvelop: boolean (defaulted to False) that activates the plotting of the node envelop

    :return:
    :param Smin: minimum path envelope
    :param Smax: maximum path envelope
    :param numNodes: vector, number of nodes used at each time step
    :param tv: vector of times used to simulate the evolution model
    :param SimulationModel: string that stores name of evolution model
    '''
    assert (numNodesMin <= numNodesMax)
    assert (0 < numNodesMin)
    dt = T / (numTimeSteps)
    tv = [dt * j for j in range(0, numTimeSteps + 1)]
    Smin = np.zeros(numTimeSteps + 1)
    Smax = np.zeros(numTimeSteps + 1)
    # Here we choose either a percentile approach to calculate the node envelop
    # or, in the case that the process does not have a closed-form, we will need
    # to simulate and estimate the node envelop (brute force), or,
    # we will use the evolution model but set the probability of an upwards and downwards
    # movement via the eta class.
    if path_envelope_method == EnvelopMethod.theo_percentiles:  # use percentiles to estimate nodal boundaries
        Smin, Smax = evolution_model.percentileLowerUpper(eta)
    elif path_envelope_method == EnvelopMethod.brute_force:  # use simulated paths to estimate nodal boundaries
        if SimulationModel == 'GBM':
            for i in range(0, numEnvelopeScenarios):
                S = evolution_model.evolveFP(evolution_model.S0)
                for j in range(0, len(Smin)):
                    if Smin[j] > S[j]:
                        Smin[j] = S[j]
                    if Smax[j] < S[j]:
                        Smax[j] = S[j]
        if SimulationModel == 'SABR':
            S = evolution_model.evolveFP(evolution_model.S0, numEnvelopeScenarios)
            for i in range(0, numEnvelopeScenarios):
                for j in range(0, len(Smin)):
                    if Smin[j] > S[i][j]:
                        Smin[j] = S[i][j]
                    if Smax[j] < S[i][j]:
                        Smax[j] = S[i][j]

    elif path_envelope_method == EnvelopMethod.brute_force_SABR:  # use simulated paths to estimate nodal boundaries
        seed = 1
        Paths = evolution_model.evolveFP(numEnvelopeScenarios, seed)
        Smin = np.zeros(numTimeSteps + 1)
        Smax = np.zeros(numTimeSteps + 1)
        for S in Paths:
            for j in range(0, len(Smin)):
                if Smin[j] > S[j]:
                    Smin[j] = S[j]
                if Smax[j] < S[j]:
                    Smax[j] = S[j]

    elif path_envelope_method == EnvelopMethod.theo_evolved:
        Smin, Smax = evolution_model.percentileEvolvedLowerUpper(eta)

    else:
        raise ("envelop calculation option not recognized")

    if nodeBNDMeth == NodeBoundaryMethod.Naiive:  # naiive implementation - Set Smin, Smax = Smin[-1], Smax[-1]
        Smin = [Smin[-1] for i in range(0, len(Smin))]
        Smax = [Smax[-1] for i in range(0, len(Smax))]
    elif nodeBNDMeth == NodeBoundaryMethod.SminToZero:  # set Smin, Smax = [0, Smax[-1]]
        Smin = [0 for j in range(0, len(Smin))]
        Smax = [Smax[-1] for j in range(0, len(Smax))]
    elif nodeBNDMeth == NodeBoundaryMethod.SminToZeroSmaxAsPercentile:  # set Smin, Smax = [0, Smax[-1]]
        Smin = [1e-6 for j in range(0, len(Smin))]
    elif nodeBNDMeth == NodeBoundaryMethod.CarryBackExtremes:  # dynamic implementation + adjustment to envelop S0
        if Smin[-1] > evolution_model.S0:
            Smin = [evolution_model.S0 for j in range(0, numTimeSteps)]
        elif Smax[-1] < evolution_model.S0:
            Smax = [evolution_model.S0 for j in range(0, numTimeSteps)]
    elif nodeBNDMeth == NodeBoundaryMethod.Dynamic:  # dynamic implementation + adjustment to envelop S0
        Smin = Smin
        Smax = Smax
    elif nodeBNDMeth == NodeBoundaryMethod.SmintoZeroSmaxToMax:  # do any of the above - whichever is preferable
        Smin = [0 for j in range(0, len(Smin))]
        val = max(Smax[-1], evolution_model.S0)
        Smax = [val for j in range(0, len(Smax))]

    elif nodeBNDMeth == NodeBoundaryMethod.ProportionalToBoundary:
        numNodes = [int((Smax[j] - Smin[j]) * node_density) for j in range(0, len(Smin))]
        numNodes[0] = 1
        for j in range(1, len(numNodes)):
            if numNodes[j] < numNodesMin:
                numNodes[j] = numNodesMin

    elif nodeBNDMeth == NodeBoundaryMethod.ProportionalToBoundaryMAXOUT:
        stepsize = (Smax[-1] - Smin[-1])/(numNodesMax-1)
        numNodes = [int(numNodesMin+stepsize*j) for j in range(0, len(Smin))]

    elif nodeBNDMeth == NodeBoundaryMethod.ProportionalToBoundaryCapped:
        maxS = -1e99
        for j in range(0, len(Smin)):
            if Smax[j] - Smin[j] > maxS:
                maxS = Smax[j] - Smin[j]

        numNodes = [int((Smax[j] - Smin[j]) * node_density / maxS * numNodesMax) for j in range(0, len(Smin))]
        numNodes[0] = 1
        for j in range(1, len(numNodes)):
            if numNodes[j] < numNodesMin:
                numNodes[j] = numNodesMin
    elif nodeBNDMeth == NodeBoundaryMethod.UserDefined:
        pass  # do nothing - user has specified how many nodes to use for each time step
    else:
        raise ("Unrecognized node boundary method")
    if not (
            nodeBNDMeth == NodeBoundaryMethod.ProportionalToBoundary or nodeBNDMeth == NodeBoundaryMethod.ProportionalToBoundaryCapped or nodeBNDMeth == NodeBoundaryMethod.UserDefined):
        d = [abs(Smax[j] - Smin[j]) for j in range(0, numTimeSteps)]
        dmin = min(d)
        dmax = max(d)
        if dmax > dmin:
            numNodes = [int((numNodesMax - numNodesMin) / (dmax - dmin) * (d[j] - dmin) + numNodesMin) for j in
                        range(0, numTimeSteps)]
        else:
            numNodes = [int(numNodesMin) for j in range(0, numTimeSteps)]

        if dmin == 0:
            numNodes[0] = 1

    plotNodeBoundaries(doPlotNodeEnvelop, tv, Smin, Smax, evolution_model.S0)

    # check is Smin < Smax, except at t=0
    for j in range(0, len(Smin)):
        if j == 0 and Smin[j] > Smax[j]:
            CRED = '\033[91m'
            CEND = '\033[0m'
            msg = "Error: Envelope boundaries are incorrect : Smin[" + str(j) + "]=" + str(Smin[j]) + " >= Smax[" + str(
                j) + "]=" + str(Smax[j]) + " is not permitted!"
            print(CRED + msg + CEND)
            assert (Smin[j] <= Smax[j])
        if j > 0 and Smin[j] >= Smax[j]:
            CRED = '\033[91m'
            CEND = '\033[0m'
            msg = "Error: Envelope boundaries are incorrect : Smin[" + str(j) + "]=" + str(Smin[j]) + " >= Smax[" + str(
                j) + "]=" + str(Smax[j]) + " is not permitted!"
            print(CRED + msg + CEND)
            assert (Smin[j] < Smax[j])

    return Smin, Smax, numNodes, tv


def simPathsVector_at_maturity_GBM(S0, mu, sigma, dt, num_paths, num_time_steps, option):
    St = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(num_paths, num_time_steps)).T
    )
    # include array of 1's
    St = np.vstack([np.ones(num_paths), St])
    # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
    St = S0 * St.cumprod(axis=0)

    result = map(lambda x: option.payoff(x), St[-1])
    lr = np.asarray(list(result))
    avg = np.mean(lr)
    contv = avg * np.exp(-mu * dt)

    return contv


def simPathsVector_at_t_GBM(S0, mu, sigma, dt, num_paths, num_time_steps, chebPricer):
    St = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(num_paths, num_time_steps)).T
    )
    # include array of 1's
    St = np.vstack([np.ones(num_paths), St])
    # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
    St = S0 * St.cumprod(axis=0)

    result = map(lambda x: chebPricer.evalLean(x), St[-1])
    avg = np.nanmean(np.asarray(list(result)))
    contv = avg * np.exp(-mu * dt)

    return max(contv, 0.0)


def simPathsVector_at_t_GBM_CHEBBARY1(S0, mu, sigma, dt, num_paths, num_time_steps, chebPricer):
    St = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(num_paths, num_time_steps)).T
    )
    # include array of 1's
    St = np.vstack([np.ones(num_paths), St])
    # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
    St = S0 * St.cumprod(axis=0)

    result = chebPricer.eval_values(St[-1])
    avg = np.nanmean(result)
    contv = avg * np.exp(-mu * dt)

    return max(contv, 0.0)

def simPathsVector_at_t_GBM_CHEBBARY2(S0, mu, sigma, dt, num_paths, num_time_steps, chebPricer):
    St = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(num_paths, num_time_steps)).T
    )
    # include array of 1's
    St = np.vstack([np.ones(num_paths), St])
    # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
    St = S0 * St.cumprod(axis=0)

    result = chebPricer.eval_values(St[-1])
    avg = np.nanmean(result)
    contv = avg * np.exp(-mu * dt)

    return max(contv, 0.0)
def simPathsVector_at_t_GBM_MOCAX(S0, mu, sigma, dt, num_paths, num_time_steps, chebPricer, smin, smax):
    St = np.exp(
        (mu - sigma ** 2 / 2) * dt
        + sigma * np.random.normal(0, np.sqrt(dt), size=(num_paths, num_time_steps)).T
    )
    # include array of 1's
    St = np.vstack([np.ones(num_paths), St])
    # multiply through by S0 and return the cumulative product of elements along a given simulation path (axis=0).
    St = S0 * St.cumprod(axis=0)
    derivative_value_id = chebPricer.get_derivative_id([0])
    # need to turn a list into a list of lists before calling eval
    xv = []
    for x in St[-1]:
        if smin <= x and x <= smax:
            xv.append([x])

    result = [chebPricer.eval(x, derivative_value_id) for x in xv]
    # result = map(lambda x: chebPricer.evalLean(x), St[-1])
    avg = np.nanmean(np.asarray(list(result)))
    contv = avg * np.exp(-mu * dt)

    return max(contv, 0.0)


class ChebyshevOptionsPricer_mocax:
    '''
    This class performs the backwards induction pricing used to evaluate an American-style option.
    It uses Chebyshev approximation to model the value function at the Chebyshev nodes at time step t.
    The Chebyshev nodes are placed between the highest and lowest limits computed for the possible pricing paths.
    At each node, and for each time step, paths are evolved from a node at t forwards in time to t + dt.
    As in the usual manner, the value of the option at that node is equal to the greater of the exercise value
    and the continuation value. Once all the nodes have a value for a given time t, Chebyshev approximation is used
    to estimate/interpolate the value function.  This value function then serves to evaluate the continuation value
    of the option from the next earliest time step. The backwards induction continues back to time 0, where only the
    paths from S0 are simulated to compute the options value.

    Constructor Inputs:
        :param evolution_model: pointer to the evolution model
        :param numEnvelopeScenarios: number of scenarios used to compute the nodal envelop
        :param numPricingPaths: a vector containing the number of inner pricing paths to simulate at each time step
        :param numTimeSteps: number of time steps used in the valuation of the option
        :param option: a pointer to the option class
        :param T: option maturity
        :param Smin: a vector containing the minium stock values used to bound the nodal points
        :param Smax: a vector containing the minium stock values used to bound the nodal points
        :param numNodes:  a vector containing the number of nodal points at each time step
        :param tv : a vector of times
        :param nodeBNDMeth: a class that defines which nodal boundary strategy to use in determining Smin and Smax
        :param doPlotPricingPaths: a boolean (defaulted to False) that activates the plotting of the pricing paths
        :param doRepairNode: a boolean (defaulted to False) that determines whether to repair nodes that do not have a value
        :param doPlotExerciseBoundary: boolean (defaulted to False) that activates the plotting of the exercise boundary,
        :param doPlotChebyshevFit: boolean (defaulted to False) that activates the plotting of the Chebyshev fit to the valuation function at time t
        :param doPlotChebyshevNodes: boolean (defaulted to False) that activates the plotting of the nodes used for interpolation
        :param doPlotExercisePointsConstellation: boolean (defaulted to False) that activates the plotting of the exercise constallation
        :param doPrintTimeStep: boolean prints timesteps if True
    '''

    def __init__(self,
                 evolution_model,
                 numEnvelopeScenarios,
                 numPricingPaths,
                 numTimeSteps,
                 option,
                 T,
                 Smin,
                 Smax,
                 numNodes,
                 tv,
                 nodeBNDMeth,
                 doInnerSimSlow=False,
                 doRepairNode=False,
                 SimulationModel='GBM',
                 doPrintTimeStep=False,
                 fit=FitMethod.mocax):

        self.evolution_model = evolution_model
        self.numEnvelopeScenarios = numEnvelopeScenarios
        self.numPricingPaths = numPricingPaths
        self.numTimeSteps = numTimeSteps
        self.T = T
        self.option = option
        self.numNodes = numNodes
        self.nodeBNDMeth = nodeBNDMeth
        self.doRepairNode = doRepairNode
        self.numNodes = numNodes
        self.dt = self.T / self.numTimeSteps
        self.Smin = Smin
        self.Smax = Smax
        self.tv = tv
        self.isInitialized = False
        self.doInnerSimSlow = doInnerSimSlow
        self.SimulationModel = SimulationModel
        self.doPrintTimeStep = doPrintTimeStep
        self.timer = {}
        self.ChebList = []
        self.fit = fit

    @property
    def calculate(self):
        cheb_obj = None
        # global cheb_obj
        self.isInitialized = True
        M = self.numTimeSteps + 1
        dt = self.T / self.numTimeSteps
        maxNumNodes = np.max(self.numNodes)
        if self.fit==FitMethod.finn:
            chebPricer = cheb1d.Chebyshev()
            chebSim = cheb1d.Chebyshev()
        elif self.fit == FitMethod.barycentric1:
            pass
        elif self.fit == FitMethod.barycentric2:
            pass

        if self.fit == FitMethod.mocax:
            dim = 1
            variable_range = [[self.Smin[-2], self.Smax[-2]]]
            domain = mox.MocaxDomain(variable_range)
            num_points = [self.numNodes[-1]]
            max_deriv_order = 0
            mcpn = mox.MocaxNs(num_points)
            chebPricer = mox.Mocax(None, num_dimensions=dim, domain=domain, error_threshold=None, n=mcpn,
                                        max_derivative_order=max_deriv_order)

            simNodes = chebPricer.get_evaluation_points()
            simNodes = [x for xs in simNodes for x in xs]

        elif self.fit == FitMethod.finn:
            simNodes = chebPricer.genNodes(self.Smin[-2], self.Smax[-2], self.numNodes[-1])

        elif self.fit == FitMethod.barycentric1:
            chebSim = barychev.ChebyshevBarycentric1(self.numNodes[-1]-1, self.Smin[-2], self.Smax[-2])
            simNodes = chebSim.xab

        elif self.fit == FitMethod.barycentric2:
            chebSim = barychev.ChebyshevBarycentric2(self.numNodes[-1]-1, self.Smin[-2], self.Smax[-2])
            simNodes = chebSim.xab

        elif self.fit == FitMethod.numpy:
            pass
        else:
            raise ('unrecognized fit method')
        maxNumNodes = len(simNodes)
        v = np.zeros([maxNumNodes, M])

        self.timer["sim"] = 0
        self.timer["fit"] = 0

        self.exerciseBoundary = np.zeros(M - 1)
        if self.doPrintTimeStep:
            print("time step : ", M - 2, " of ", M - 1)

        for i in range(len(simNodes) - 1, -1, -1):
            if i < len(simNodes) - 1 and v[i + 1, M - 2] == 0:
                v[i, M - 2] = 0
                continue
            nod = simNodes[i]
            start = time.time()
            if self.SimulationModel == 'GBM':
                if self.doInnerSimSlow:
                    contv = 0
                    for k in range(0, self.numPricingPaths[-2]):
                        S = self.evolution_model.evolveT(nod, dt)
                        contv = contv + self.option.payoff(S)

                    contv = contv * np.exp(-self.evolution_model.r * self.dt) / self.numPricingPaths[-2]
                else:
                    mu = self.evolution_model.r
                    sigma = self.evolution_model.sigma
                    contv = simPathsVector_at_maturity_GBM(nod, mu, sigma, dt, self.numPricingPaths[-2], 1, self.option)
            elif self.SimulationModel == 'SABR':
                if self.doInnerSimSlow:
                    contv = 0
                    for k in range(0, self.numPricingPaths[-2]):
                        S = self.evolution_model.evolveT(nod, dt)
                        contv = contv + self.option.payoff(S)
                else:
                    # vectorized SABR
                    paths = self.evolution_model.evolveT(nod, self.numPricingPaths[-2])
                    paths2 = [paths[i][0] for i in range(0, self.numPricingPaths[-2])]
                    result = map(lambda x: self.option.payoff(x), paths2)
                    lr = np.asarray(list(result))
                    avg = max(np.mean(lr), 0)
                    contv = avg * np.exp(-self.evolution_model.r * dt)
            else:
                raise ('unknown simulation model')
            end = time.time()
            et = end - start

            self.timer["sim"] += et
            ex = self.option.payoff(nod)
            v[i, M - 2] = max(ex, contv)
            if ex > contv:
                self.exerciseBoundary[M - 2] = max(self.exerciseBoundary[M - 2], nod)

        if self.fit == FitMethod.mocax:
            yvalues = list(v[0:len(simNodes), M - 2])
            chebPricer.set_original_function_values(yvalues)
        elif self.fit == FitMethod.finn:
            chebPricer.genCoeffGivenFromFuncValuesAndNodes(simNodes, v[0:len(simNodes), M - 2])
        elif self.fit == FitMethod.barycentric1:
            chebPricer = barychev.ChebyshevBarycentric1(self.numNodes[-1]-1, self.Smin[-2], self.Smax[-2])
            fk = v[0:len(simNodes), M - 2]
            chebPricer.set_fk(fk)
        elif self.fit == FitMethod.barycentric2:
            chebPricer = barychev.ChebyshevBarycentric2(self.numNodes[-1]-1, self.Smin[-2], self.Smax[-2])
            fk = v[0:len(simNodes), M - 2]
            chebPricer.set_fk(fk)
        elif self.fit == FitMethod.numpy:
            pass
        else:
            raise ('unrecognized fit method')
        ####################################################################################################
        '''
        coeff_fit = np.polynomial.chebyshev.chebfit(simNodes, v[0:len(simNodes), M - 2], len(simNodes))
        print("chebPricer.c:", chebPricer.c)
        print("coeff_fit:", coeff_fit)
        np.polynomial.chebyshev.chebval(self.evolution_model.S0, coeff_fit, tensor=True)

        NPTSPLT=100
        x = [self.Smin[M - 2] + k*( self.Smax[M-2]-self.Smin[M - 2])/NPTSPLT for k in range(0,NPTSPLT)]
        ych = [chebPricer.eval(x[k]) for k in range(0,NPTSPLT)]
        yfit = np.polynomial.chebyshev.chebval(x, coeff_fit, tensor=True)
        plt.plot(x, ych, 'k', label='cheb textbook')
        plt.plot(x, ych, 'k+')
        plt.plot(x, yfit, 'b', label='NumPy fit')
        plt.plot(x, yfit, 'b+')

        plt.xlabel('asset value ($)')
        plt.ylabel('Option value ($)')
        plt.legend()
        plt.savefig('Figure_cheb_fit.png')
        # plt.clf()
        plt.show()
        '''

        #############################################################################################
        self.ChebList.append(chebPricer)
        if self.numTimeSteps <= 1:
            # return chebPricer.eval(self.evolution_model.S0)
            if self.fit ==FitMethod.mocax:
                derivative_value_id = chebPricer.get_derivative_id([0])
                return chebPricer.eval(self.evolution_model.S0, derivative_value_id)
            elif self.fit == FitMethod.finn:
                return chebPricer.evalLeanIndices(self.evolution_model.S0)
            elif self.fit == FitMethod.barycentric1:
                return chebPricer.eval_values(self.evolution_model.S0)
            elif self.fit == FitMethod.barycentric2:
                return chebPricer.eval_values(self.evolution_model.S0)
            elif self.fit == FitMethod.numpy:
                pass
            else:
                raise ('unrecognized fit method')

        for j in range(M - 3, -1, -1):
            #if j==47:
            #    print("heere")
            if self.doPrintTimeStep:
                print("time step : ", j, " of ", M - 1)
            if self.fit == FitMethod.mocax:
                if not self.Smin[j] == self.Smax[j]:
                    dim = 1
                    variable_range = [[self.Smin[j], self.Smax[j]]]
                    # print("Variable_range : ", variable_range)
                    domain = mox.MocaxDomain(variable_range)
                    num_points = [self.numNodes[j]]
                    max_deriv_order = 0
                    mcpn = mox.MocaxNs(num_points)
                    cheb_obj_sim = mox.Mocax(None, num_dimensions=dim, domain=domain, error_threshold=None, n=mcpn,
                                             max_derivative_order=max_deriv_order)

                    simNodes = cheb_obj_sim.get_evaluation_points()
                else:
                    simNodes = [[self.Smin[j]]]

            elif self.fit == FitMethod.finn:
                simNodes = chebSim.genNodes(self.Smin[j], self.Smax[j], self.numNodes[j])
            elif self.fit == FitMethod.numpy:
                pass
            elif self.fit == FitMethod.barycentric1:
                chebSim = barychev.ChebyshevBarycentric1(self.numNodes[j]-1, self.Smin[j], self.Smax[j])
                simNodes = chebSim.xab
            elif self.fit == FitMethod.barycentric2:
                chebSim = barychev.ChebyshevBarycentric2(self.numNodes[j]-1, self.Smin[j], self.Smax[j])
                simNodes = chebSim.xab
            else:
                raise ('unrecognized fit method')

            badNodeIndices = []
            goodNodeIndices = []
            start = time.time()

            for i in range(len(simNodes) - 1, -1, -1):
                # if i < len(simNodes) - 1 and v[i + 1, j] == 0:
                #    v[i, j] = 0
                #    continue
                nod = simNodes[i]
                if self.SimulationModel == 'GBM':
                    count = 0
                    if self.doInnerSimSlow:
                        contv = 0
                        for k in range(0, self.numPricingPaths[j]):
                            S = self.evolution_model.evolveT(nod, dt)

                            if chebPricer.inbounds(S):
                                # val = max(chebPricer.eval(S), 0)
                                val = max(chebPricer.evalLeanIndices(S), 0)
                                contv = contv + val
                            else:
                                count = count + 1

                        if self.numPricingPaths[j] - count == 0:
                            badNodeIndices.append(i)
                        else:
                            goodNodeIndices.append(i)

                            contv = contv * np.exp(-self.evolution_model.r * self.dt) / (
                                    self.numPricingPaths[j] - count)
                    else:
                        mu = self.evolution_model.r
                        sigma = self.evolution_model.sigma
                        start = time.time()
                        if self.fit == FitMethod.mocax:
                            contv = simPathsVector_at_t_GBM_MOCAX(nod, mu, sigma, dt, self.numPricingPaths[j], 1,
                                                                  chebPricer, self.Smin[j + 1], self.Smax[j + 1])
                        elif self.fit == FitMethod.finn:
                            contv = simPathsVector_at_t_GBM(nod, mu, sigma, dt, self.numPricingPaths[j], 1, chebPricer)
                        elif self.fit == FitMethod.numpy:
                            pass
                        elif self.fit == FitMethod.barycentric1:
                            contv = simPathsVector_at_t_GBM_CHEBBARY1(nod, mu, sigma, dt, self.numPricingPaths[j], 1,
                                                                  chebPricer)
                        elif self.fit == FitMethod.barycentric2:
                            contv = simPathsVector_at_t_GBM_CHEBBARY2(nod, mu, sigma, dt, self.numPricingPaths[j], 1,
                                                                  chebPricer)
                        else:
                            raise ('unrecognized fit method')

                        end = time.time()
                        et = end - start
                        self.timer["sim"] += et
                elif self.SimulationModel == 'SABR':
                    if self.doInnerSimSlow:
                        contv = 0
                        for k in range(0, self.numPricingPaths[j]):
                            S = self.evolution_model.evolveT(nod, dt)

                            if chebPricer.inbounds(S):
                                # val = max(chebPricer.eval(S), 0)
                                val = max(chebPricer.evalLeanIndices(S), 0)
                                contv = contv + val
                            else:
                                count = count + 1

                        if self.numPricingPaths[j] - count == 0:
                            badNodeIndices.append(i)
                        else:
                            goodNodeIndices.append(i)

                            contv = contv * np.exp(-self.evolution_model.r * self.dt) / (
                                    self.numPricingPaths[j] - count)
                    else:
                        paths = self.evolution_model.evolveT(nod, self.numPricingPaths[j])
                        paths2 = [paths[i][0] for i in range(0, self.numPricingPaths[-2])]
                        result = map(lambda x: chebPricer.evalLean(x), paths2)
                        avg = max(np.nanmean(np.asarray(list(result))), 0.0)
                        contv = avg * np.exp(-self.evolution_model.r * dt)

                else:
                    raise ('unknown Simulation Model')
                if self.fit == FitMethod.mocax:
                    nod = nod[0]
                ex = self.option.payoff(nod)
                v[i, j] = max(ex, contv)
                if ex > contv:
                    self.exerciseBoundary[j] = max(self.exerciseBoundary[j], nod)

            v, msg = self.repairNodeValues(badNodeIndices, goodNodeIndices, v, j)
            start = time.time()
            self.ChebList.append(chebPricer)

            if self.Smin[j] < self.Smax[j]:
                if self.fit == FitMethod.mocax:
                    dim = 1
                    variable_range = [[self.Smin[j], self.Smax[j]]]
                    domain = mox.MocaxDomain(variable_range)
                    num_points = [len(simNodes) - 1]
                    max_deriv_order = 0
                    mcpn = mox.MocaxNs(num_points)
                    chebPricer = mox.Mocax(None, num_dimensions=dim, domain=domain, error_threshold=None, n=mcpn,
                                                max_derivative_order=max_deriv_order)
                    yvalues = np.array(v[0:len(simNodes), j]).tolist()
                    chebPricer.set_original_function_values(yvalues)
                elif self.fit == FitMethod.finn:
                    chebPricer.genCoeffGivenFromFuncValues(self.Smin[j], self.Smax[j], v[0:len(simNodes), j])
                elif self.fit == FitMethod.numpy:
                    pass
                elif self.fit == FitMethod.barycentric1:
                    nnn = len(v[0:len(simNodes), j])-1
                    chebPricer = barychev.ChebyshevBarycentric1(nnn, self.Smin[j], self.Smax[j])
                    chebPricer.set_fk(v[0:len(simNodes), j])
                elif self.fit == FitMethod.barycentric2:
                    nnn = len(v[0:len(simNodes), j])-1
                    chebPricer = barychev.ChebyshevBarycentric2(nnn, self.Smin[j], self.Smax[j])
                    chebPricer.set_fk(v[0:len(simNodes), j])
                else:
                    raise ('unrecognized fit method')

                ####################################################################################################
                '''
                coeff_fit = np.polynomial.chebyshev.chebfit(simNodes, v[0:len(simNodes), j], len(simNodes))
                print("chebPricer.c:", chebPricer.c)
                print("coeff_fit:", coeff_fit)
                np.polynomial.chebyshev.chebval(self.evolution_model.S0, coeff_fit, tensor=True)

                NPTSPLT = 100
                x = [self.Smin[j] + k * (self.Smax[j] - self.Smin[j]) / NPTSPLT for k in range(0, NPTSPLT)]
                ych = [chebPricer.eval(x[k]) for k in range(0, NPTSPLT)]
                yfit = np.polynomial.chebyshev.chebval(x, coeff_fit, tensor=True)
                plt.plot(x, ych, 'k', label='cheb')
                plt.plot(x, ych, 'k+')
                plt.plot(x, yfit, 'b', label='fit')
                plt.plot(x, yfit, 'b+')

                plt.title('Plot for timestep '+str(j))
                plt.ylabel('asset value ($)')
                plt.xlabel('nodes')
                plt.legend()
                plt.savefig('Figure_cheb_fit.png')
                # plt.clf()
                plt.show()
                '''
                #############################################################################################
            end = time.time()
            et = end - start
            self.timer["fit"] += et
        if len(simNodes) == 1:
            self.pv = v[0, 0]
        else:
            # self.pv = max(chebPricer.eval(self.evolution_model.S0), 0)
            self.pv = max(chebPricer.evalLeanIndices(self.evolution_model.S0), 0)

        # print("Chebyshev pv = ", self.pv)
        # print("v :\n", v)
        self.ChebList.reverse()
        return self.pv, self.exerciseBoundary

    def repairNodeValues(self, badNodeIndices, goodNodeIndices, vin, j):
        '''
        :param badNodeIndices: list of nodes that do not value a value
        :param goodNodeIndices: list of nodes that do have a value
        :param vin: input value matrix
        :param j: time index
        :return:
        :param vout: repaired output valuation matrix
        :param msg: a string detailing warnings or errors
        '''
        # find the neighbouring value in v that exists and assign the missing value entry to it
        vout = vin.copy()
        msg = "Number of nodes repaired : " + str(len(badNodeIndices))
        if not self.doRepairNode or len(badNodeIndices) == 0:
            return vout, msg
        # print("repairing indices : \n", badNodeIndices)
        if len(goodNodeIndices) == 0:
            # this is hopeless...there's no value we can ue here
            return vout
        igmin = min(goodNodeIndices)
        igmax = max(goodNodeIndices)
        for i in badNodeIndices:
            if i < igmin:
                vout[i, j] = max([vin[i, j], vin[igmin, j]])
            elif i > igmax:
                vout[i, j] = max([vin[i, j], vin[igmax, j]])

        # print(v)
        return vout, msg

    def fitExerciseBoundary(self, numNodes):
        '''
        This routine uses chebyshev interpolation to approximate the exercise boundary previous computed by
        the self.calculate() method.  It does so by first linearly interpolating the exercise boundary at the
        Chebyshev points, and then approximating the linearly interpolated function using Chebyshev approximation.

        :param numNodes: number of Chebyshev nodes to use for the interpolation of the exercise boundary function
        :return:

        nothing is returned but the attribute self.chebEx is created and readied
        '''
        assert (self.isInitialized)
        self.chebEx = cheb1d.Chebyshev()
        interpNodes = self.chebEx.genNodes(self.tv[0], self.tv[-1], numNodes)
        # interpolate to nodal points

        values = np.interp(interpNodes, self.tv[0:len(self.tv) - 1], self.exerciseBoundary, left=None, right=None,
                           period=None)
        self.chebEx.genCoeffGivenFromFuncValues(self.tv[0], self.tv[-2], values)

    def fitExerciseBoundaryNo0(self, numNodes):
        '''
        This routine uses chebyshev interpolation to approximate the exercise boundary previous computed by
        the self.calculate() method.  It does so by first linearly interpolating the exercise boundary at the
        Chebyshev points, and then approximating the linearly interpolated function using Chebyshev approximation.

        :param numNodes: number of Chebyshev nodes to use for the interpolation of the exercise boundary function
        :return:

        nothing is returned but the attribute self.chebEx is created and readied
        '''
        assert (self.isInitialized)
        self.chebEx = cheb1d.Chebyshev()
        interpNodes = self.chebEx.genNodes(self.tv[0], self.tv[-1], numNodes)
        # interpolate to nodal points

        ebn0 = []
        tvn0 = []
        for j in range(0, len(self.exerciseBoundary)):
            if self.exerciseBoundary[j] > 0:
                ebn0.append(self.exerciseBoundary[j])
                tvn0.append(self.tv[j])

        values = np.interp(interpNodes, tvn0, ebn0, left=None, right=None, period=None)
        self.chebEx.genCoeffGivenFromFuncValues(tvn0[0], tvn0[-1], values)

    def evalExerciseBoundary(self, t, n=-1):
        '''
        This function evaluated the exercise boundary at time t and returns the interpolated value.
        The constraints for the inputs are:
            i. t in [0,T]
            ii. n in [1, numNodes], where numNodes is as passed into fitExerciseBoundary.
        :param t: the point in time where we want to evaluate the exercise boundary
        :param n: the number of Chebyshev polynomials we want to use in the approximation
        :return:
        returns the exercise boundary approximation at time t
        '''
        assert (self.isInitialized)
        assert (n <= len(self.chebEx.xnodes))

        return self.chebEx.eval(t, n)

    def evalOption(self, n, S):
        M = self.numTimeSteps + 1
        if n < 0 or n > M - 2:
            raise ("t must reside in [0,T]")
        if n == M - 1:
            return self.option(S)

        cheb = self.ChebList[n]

        if S <= cheb.a:
            return cheb.eval(cheb.a)
        elif S >= cheb.b:
            return cheb.eval(cheb.b)

        return cheb.eval(S)


def fitExerciseBoundary(numNodes, tv, exerciseBoundary):
    '''
    This method uses Chebushev approximation to interpolate the exercise boundary.
    The first step is to use linear interpolation to obtain points in tv that correspond to
    the locations of the Chebyshev nodes, and the to fit the Chebyshev polynomials.
    :param numNodes: Number of Chebyshev nodes we want to use n the polynomial approximation
    :param tv: a vector of times at which the boundary is defined
    :param exerciseBoundary: the vector of exercise boundary points tht corresponds to times tv
    :return:
    None
    '''
    cheb = cheb1d.Chebyshev()
    interpNodes = cheb.genNodes(tv[0], tv[-1], numNodes)
    # interpNodes.reverse()
    # print(type(interpNodes))
    # print("node:\n", interpNodes)
    # print("tv:\n", tv)
    # interpolate to nodal points
    values = np.interp(interpNodes, tv, exerciseBoundary, left=None, right=None, period=None)

    cheb.genCoeffGivenFromFuncValues(tv[0], tv[-1], values)
    # print("exerciseBoundary\n", exerciseBoundary)
    # print("values\n", values)

    return cheb


def list2text(v):
    s = "["
    comma = ""
    for t in v:
        s = s + comma + str(t)
        comma = ", "

    return s + "]"


def writeArrayToFile(filename, tv, eb, label_name, colour):
    f = open(filename, "w")
    tv_txt = list2text(tv)
    eb_txt = list2text(eb)

    f.writelines("import matplotlib.pyplot as plt\n")
    f.writelines("tv=" + tv_txt + "\n")
    f.writelines("eb=" + eb_txt + "\n")
    f.writelines("plt.plot(tv, eb, '*" + colour + "', label='" + label_name + "')" + "\n")
    f.writelines("plt.plot(tv, eb, '" + colour + "')" + "\n")
    f.writelines("plt.ylabel('stock level')" + "\n")
    f.writelines("plt.xlabel('time')" + "\n")
    f.writelines("plt.legend()" + "\n")
    f.writelines("plt.show()" + "\n")

    f.close()


def gen_plots(f1, f2, nodes):
    b = nodes[0]
    a = nodes[len(nodes) - 1]
    n = 100
    dx = (b - a) / n
    x = a
    f1v = []
    f2v = []
    xv = []

    while x < b:
        f1v.append(f1.eval(x))
        f2v.append(f2.eval(x))
        xv.append(x)
        x = x + dx

    return f1v, f2v, xv


def FindCrossings(nodes, ev, cv, doPlot):
    crossing_found = False
    zero_found = False
    indicies = [-1, -1]
    dv = [ev[j] - cv[j] for j in range(0, len(ev))]
    for j in range(len(dv) - 2, -1, -1):
        if zero_found == False and dv[j] * dv[j + 1] == 0:
            # print("zero : ", j, j + 1)
            zero_found = True
            indicies = [j, j + 1]
        if dv[j] * dv[j + 1] < 0:
            crossing_found = True
            indicies = [j, j + 1]
            if doPlot:
                print("crossing : ", j, j + 1)
                plt.plot(nodes[j], dv[j], 'r*')
                plt.plot(nodes[j + 1], dv[j + 1], 'r*')
                plt.plot([nodes[j], nodes[j + 1]], [dv[j], dv[j + 1]], 'r')
            break
    if doPlot:
        plt.plot(nodes, dv, 'r', label='ev-cv')
        plt.plot(nodes, ev, 'b', label='ev')
        plt.plot(nodes, cv, 'g', label='cv')
        plt.ylabel('ev-cv')
        plt.xlabel('nodes')
        plt.legend()
        plt.show()

    return zero_found, crossing_found, indicies


def Gen_NN_Data(cop):
    print(cop.ChebList)
    print(cop.evalOption(4, 90))

    NP = 100000
    s_train = np.zeros(NP)
    t_train = np.zeros(NP)
    y_train = np.zeros(NP)
    smin = cop.Smin
    smax = cop.Smax
    k = 0
    for i in range(0, len(cop.ChebList)):
        for j in range(0, NP):
            s = random.random(smin[j], smax[j])
            s_train[k] = s
            t_train[k] = i

            y_train[k] = cop.evalOption(i, s)
            k = k + 1


## my code starts here
class OptionHedgingEnv_GBM(gym.Env):
    def __init__(self, strike_price, initial_stock_price, risk_free_rate, volatility, time_horizon, max_steps, kappa):
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.mu = risk_free_rate
        self.sigma = volatility
        self.max_steps = max_steps
        self.time_horizon = time_horizon
        self.dt = self.time_horizon / self.max_steps
        self.dt_tree = 1/5000
        self.initial_stock_price = initial_stock_price
        self.stock_price = initial_stock_price
        self.kappa = kappa
        self.current_shares = 0
        self.step_count = 0
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0,0.0]), high=np.array([float(initial_stock_price) * 10.0, time_horizon, 1.0]))
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.stock_prices = self.generate_stock_tree()
        self.option_prices = self.generate_option_tree_()
        #self.alpha_tree = self.calculate_alpha_tree()
        
        #### Cheb stuff
        path_envelope_method = EnvelopMethod.theo_percentiles
        numEnvelopeScenarios = 100000

        T = 1
        q = 0
        eta = probConveyor.Eta(0.9999)
        SimulationModel = 'GBM'
        nodeBNDMeth = NodeBoundaryMethod.ProportionalToBoundaryCapped
        numNodesMax =300
        numNodesMin = 10
        node_density =1
        doPlotNodeEnvelop = False
        npp =100
        numTimeSteps = self.max_steps
        numPricingPaths = [npp]*numTimeSteps
        evolution_model = em.GBM(self.initial_stock_price, self.risk_free_rate, self.sigma, numTimeSteps, self.time_horizon)
        fit = FitMethod.finn
        doPrintTimeStep = False
        doRepairNode = False
        doPlotNodeEnvelop = False
        # numPricePaths is the number of inner paths used to calculate the continuation value
        doInnerSimSlow = False
        path_envelope_method = EnvelopMethod.theo_percentiles
        nodeBNDMeth = NodeBoundaryMethod.ProportionalToBoundaryMAXOUT


        Smin, Smax, numNodes, tv = calcNodeBoundaries(path_envelope_method, nodeBNDMeth, evolution_model,
                                                                  numNodesMax,
                                                                  numNodesMin, numEnvelopeScenarios,
                                                                  numTimeSteps, T, eta, node_density, SimulationModel,
                                                                  doPlotNodeEnvelop)
                    #print(numNodes)
                    # instantiate a Chebyshev pricer object
        option = op.PutOption(self.strike_price)
        self.cop = ChebyshevOptionsPricer_mocax(evolution_model, numEnvelopeScenarios, numPricingPaths, numTimeSteps,
                                                       option, T, Smin, Smax, numNodes, tv, nodeBNDMeth,
                                                       doInnerSimSlow, doRepairNode, SimulationModel, doPrintTimeStep,
                                                       fit)
        a,b = self.cop.calculate
        
        


        self.reset()

    def reset(self):
        option_path = []
        self.step_count = 0
        self.stock_price = self.initial_stock_price
        self.current_shares = 0
        time_to_maturity = self.time_horizon
        
        # Set initial option price
        self.option_price = self.option_prices[0, 0]
        self.cheb_option_price = self.cop.evalOption(0, self.initial_stock_price)
        print(self.option_price, self.cheb_option_price)
        #print(self.option_price)

        self.initial_portfolio = self.option_price
        self.invested_portfolio = self.initial_portfolio
      
        return np.array([self.initial_stock_price, time_to_maturity, 0])

    def step(self, action):
        self.step_count += 1
        prev= self.cheb_option_price 
        dt = self.time_horizon / self.max_steps
        time_to_maturity = self.time_horizon - dt * self.step_count
        prev_shares = self.current_shares
        self.current_shares = action[0]
        dWt = np.random.normal(0, np.sqrt(dt))
        new_stock_price = self.stock_price * np.exp((self.risk_free_rate - 0.5 * self.sigma**2) * dt + self.sigma * dWt)

        new_option_price = self.cheb_option_price
        if self.step_count<self.max_steps:
            new_option_price  =  self.cop.evalOption(self.step_count, new_stock_price)
        else:
            new_option_price = self.cheb_option_price 
            
        option_change = new_option_price - self.cheb_option_price
        #print(, new_option_price)
        stock_value_change = self.current_shares * (new_stock_price - self.stock_price)
        reward = -np.abs(stock_value_change + option_change)
        reward -= self.kappa*((self.current_shares - prev_shares )**2)*(new_stock_price)
        self.stock_price = new_stock_price
        self.option_price = new_option_price
        self.cheb_option_price = new_option_price

        done = self.step_count >= self.max_steps
        next_state = np.array([self.stock_price, time_to_maturity, self.current_shares])
        

        return next_state, reward, done, {}
    
    
    def calculate_alpha_tree(self):
        alpha_tree = np.zeros((5000, 5000))

        for i in range(5000):
            for j in range(0, i + 1):
                alpha_tree[j, i] = (self.option_prices[j+1, i+1] - self.option_prices[j, i+1])/(self.stock_prices[j, i]*(np.exp(self.risk_free_rate*self.dt_tree+self.sigma*np.sqrt(self.dt_tree))-np.exp(self.risk_free_rate*self.dt_tree-self.sigma*np.sqrt(self.dt_tree))))
        
        return alpha_tree


    def interpolate_alpha(self, arbitrary_stock_price):
        # Generate the stock and alpha trees
        stock_tree = self.stock_prices
        alpha_tree = self.alpha_tree

        # Find the time step corresponding to the given arbitrary_time_step
        timesteps = np.linspace(0, self.time_horizon, 5001)  
        closest_timestep_index = min(np.abs(timesteps - self.step_count * self.dt).argmin(), 5000 - 1) 

        # Find the stock price indices closest to the arbitrary_stock_price
        stock_price_indices = np.abs(stock_tree[:, closest_timestep_index] - arbitrary_stock_price).argmin()

        # Use linear interpolation to estimate the alpha value
        alpha_at_arbitrary_point = np.interp(
            arbitrary_stock_price,
            stock_tree[stock_price_indices:stock_price_indices + 2, closest_timestep_index],
            alpha_tree[stock_price_indices:stock_price_indices + 2, closest_timestep_index]
        )

        return alpha_at_arbitrary_point


    def normalize_state(self, state):
        normalized_state = np.array([
            (state[0] - self.initial_stock_price) / self.initial_stock_price,
            state[1] / self.time_horizon,
            state[2]
        ])
        return normalized_state
    def get_q(self):
        return (np.exp((self.risk_free_rate - self.sigma**2/2) * self.dt_tree) - np.exp(-self.sigma * np.sqrt(self.dt_tree))) / (np.exp(self.sigma * np.sqrt(self.dt_tree)) - np.exp(-self.sigma * np.sqrt(self.dt_tree)))
    def calculate_stock_up_down_factors(self):
        u = np.exp((self.mu - self.sigma**2/2) * self.dt_tree + self.sigma * np.sqrt(self.dt_tree))
        d = np.exp((self.mu - self.sigma**2/2) * self.dt_tree - self.sigma * np.sqrt(self.dt_tree))
        return u, d

    def generate_stock_tree(self):
        stock_asset_prices = np.zeros((5001, 5001))  
        u, d = self.calculate_stock_up_down_factors()
        for i in range(5001):
            for j in range(i + 1):
                stock_asset_prices[j, i] = self.stock_price * (u ** (i - j)) * (d ** j)
        return stock_asset_prices

    def generate_BA_tree(self):
        bank_account_prices = np.zeros((5001, 5001))  
        u, d = self.calculate_bank_account_up_down_factors()
        for i in range(5001):
            for j in range(i + 1):
                bank_account_prices[j, i] = self.initial_stock_price * (u ** (i - j)) * (d ** j)
        return bank_account_prices

    def generate_option_tree_(self):
        q = self.get_q()
        put_opt_prices = np.zeros((5001, 5001)) 
        stock_tree = self.stock_prices
        for i in range(5001):
            put_opt_prices[i, -1] = max(0, self.strike_price - stock_tree[i, -1])

        for i in range(4999, -1, -1):
            for j in range(0, i + 1):
                hold_value = np.exp(-self.risk_free_rate * self.time_horizon / 5000) * (
                    q * put_opt_prices[j, i + 1] + (1 - q) * put_opt_prices[j + 1, i + 1]
                )
                exercise_value = max(0, self.strike_price - stock_tree[j, i])
                put_opt_prices[j, i] = max(exercise_value, hold_value)
        return put_opt_prices

    def interpolate_option_price(self, arbitrary_stock_price):
        # Generate the stock and option trees
        stock_tree = self.stock_prices
        option_tree = self.option_prices

        # Find the time step corresponding to the given arbitrary_time_step
        timesteps = np.linspace(0, self.time_horizon, 5001)  
        closest_timestep_index = np.abs(timesteps - self.step_count * self.dt).argmin()

        # Find the stock price indices closest to the arbitrary_stock_price
        stock_price_indices = np.abs(stock_tree[:, closest_timestep_index] - arbitrary_stock_price).argmin()

        # Use linear interpolation to estimate the option price
        option_price_at_arbitrary_point = np.interp(
            arbitrary_stock_price,
            stock_tree[stock_price_indices:stock_price_indices + 2, closest_timestep_index],
            option_tree[stock_price_indices:stock_price_indices + 2, closest_timestep_index]
        )

        return option_price_at_arbitrary_point
    
    def get_option_price(self, time_to_maturity, new_stock_price):
        d1 = (np.log(new_stock_price / self.strike_price) +
              (self.risk_free_rate + 0.5 *self.sigma ** 2) * time_to_maturity) / (
                     self.sigma * np.sqrt(time_to_maturity))
        d2 = d1 - self.sigma * np.sqrt(time_to_maturity)

        put_option_price =  self.strike_price * np.exp(
            -self.risk_free_rate * time_to_maturity) * norm.cdf(-d2) -new_stock_price * norm.cdf(-d1) 
        delta = norm.cdf(d1)-1
        return put_option_price,delta


class OptionHedgingEnv_svol(gym.Env):
    def __init__(self, strike_price, initial_stock_price, risk_free_rate, volatility, time_horizon, max_steps,  rho, nu,day,prices, symbol, real, kappa):
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.sigma = volatility
        self.day = day
        self.max_steps = max_steps
        self.time_horizon = time_horizon
        self.initial_stock_price = initial_stock_price
        self.stock_price = initial_stock_price
        self.current_shares = 0
        self.prices = prices
        self.step_count = 0
        self.real =real
        self.kappa = kappa
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0
                                                              #, 0.0
                                                              ]),
                                                high=np.array([float(initial_stock_price) * 10.0, time_horizon
                                                               #, 1.0
                                                               ]))
        self.action_space = gym.spaces.Box(low=0, high=1, shape =(1,))
        self.q=0.
        self.sigma0=self.sigma
        self.symbol = symbol
        self.rho=rho
        self.nu=nu
        self.dt = self.time_horizon/self.max_steps
        

    def reset(self):
        self.step_count = 0
        if self.real:
            self.stock_price = self.prices[0]
        else:
            self.stock_price =self.initial_stock_price
        
        self.current_shares = 0
        time_to_maturity = self.time_horizon
        self.sigma = self.sigma0
        self.option_price = self.sim_option_price( self.stock_price, time_to_maturity, self.sigma, self.step_count) 
        
        return np.array([self.initial_stock_price, time_to_maturity,
                         self.current_shares
                         ])
    
    

    def step(self, action):
        self.step_count += 1
        dt = self.time_horizon / self.max_steps
        time_to_maturity = max(self.time_horizon - self.step_count*dt, 1e-16)
        
        prev_shares = self.current_shares
        self.current_shares = action[0]
         # Clip shares within valid range
        Z1=np.random.randn()
        Z2=self.rho*Z1+np.sqrt(1-self.rho**2)*np.random.randn()
        sigma_new=self.sigma+self.nu*self.sigma*np.sqrt(dt)*Z2
        
        
        
        if self.real:
            new_stock_price=self.prices[self.step_count] ## for real data test
        else:
           new_stock_price= self.stock_price +self.risk_free_rate*self.stock_price*dt + sigma_new*self.stock_price*np.sqrt(dt)*Z1

        new_option_price =self.sim_option_price( new_stock_price, time_to_maturity, sigma_new, self.step_count)
    
        
        option_change = (new_option_price - self.option_price)
        
        stock_value_change =  self.current_shares *( new_stock_price -self.stock_price)
        reward = -np.abs(stock_value_change + option_change)
        reward -= self.kappa*((self.current_shares - prev_shares )**2)*(new_stock_price)
        self.stock_price = new_stock_price
        self.option_price = new_option_price
      
        done = self.step_count >= self.max_steps
        next_state = np.array([self.stock_price, time_to_maturity
                              , self.current_shares
                              ])
        prev_shares = self.current_shares
        self.sigma = sigma_new
        return next_state, reward, done, {}

    def get_option_price(self, time_to_maturity, new_stock_price):
        d1 = (np.log(new_stock_price / self.strike_price) +
              (self.risk_free_rate + 0.5 * self.sigma ** 2) * time_to_maturity) / (
                     self.sigma * np.sqrt(time_to_maturity))
        d2 = d1 - self.sigma * np.sqrt(time_to_maturity)

        put_option_price = -new_stock_price * norm.cdf(-d1) + self.strike_price * np.exp(
            -self.risk_free_rate * time_to_maturity) * norm.cdf(-d2)
        delta = (norm.cdf(d1) -1)
        return put_option_price,delta
    
    def normalize_state(self, state):
        normalized_state = np.array([
            (state[0] - self.initial_stock_price) / (self.initial_stock_price ),  # Normalize stock price
            state[1] / self.time_horizon,  # Normalize time to maturity
            state[2]  # Current shares
        ])
        
        return normalized_state
    def load_exercise_boundary_from_csv(self, filename):
        time_points = []
        exercise_boundary = []
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                time_points.append(float(row['Time']))
                exercise_boundary.append(float(row['Exercise Boundary']))
        return time_points, exercise_boundary
    def sim_option_price(self, stock_price, time_to_maturity, sigma, step):
        strike_price = self.strike_price
        option_prices = []
        num_paths = 10000
        num_steps = self.max_steps - step +1
        option_exercised = False
        Z1 = np.random.normal(size=(num_steps, num_paths))
        Z2 = np.zeros((num_steps, num_paths))

        Sout=np.full((num_paths, num_steps),np.nan)
        Sigma=np.full(Sout.shape,np.nan)
        Sout[:,0]=stock_price
        Sigma[:,0]=sigma
        Z1=np.random.randn(num_paths,num_steps)
        Z2=self.rho*Z1+np.sqrt(1-self.rho**2)*np.random.randn(num_paths,num_steps)

        if self.symbol is None:
            exercise_boundary_filename = 'cheb_fit2.csv' 
        else:
            exercise_boundary_filename = f'{self.symbol}_fit{self.strike_price}-{self.day}.csv'   # Adjust the filename 
        time_points, exercise_boundary = self.load_exercise_boundary_from_csv(exercise_boundary_filename)
        new_time_points = np.linspace(0, self.time_horizon, self.max_steps) ## day is calendar days

        self.exercise_boundary = np.interp(new_time_points, time_points, exercise_boundary)
        self.exercise_boundary = self.exercise_boundary[step:]
        for i in range(1,num_steps):
            Sigma[:,i]=Sigma[:,i-1]+self.nu*Sigma[:,i-1]*np.sqrt(self.dt)*Z2[:,i]
            Sout[:,i]=Sout[:,i-1]+(self.risk_free_rate)*Sout[:,i-1]*self.dt+Sigma[:,i-1]*Sout[:,i-1]*np.sqrt(self.dt)*Z1[:,i]

        for j in range(num_paths):
            option_exercised = False

            for i in range(0, num_steps-1):  
                # Check if the stock price hits the exercise boundary
                if Sout[j, i] <= self.exercise_boundary[i]:
                    option_prices.append(max(strike_price - Sout[j, i], 0))
                    option_exercised = True
                    break


            if not option_exercised:
                option_prices.append(max(strike_price - Sout[j, -1], 0))

        return np.mean(option_prices)*np.exp(-self.risk_free_rate*time_to_maturity)

class OptionHedgingEnv_svol_w(gym.Env):
    def __init__(self, strike_price, initial_stock_price, risk_free_rate, volatility, time_horizon, max_steps,  rho, nu,day, symbol, week, kappa):
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.sigma = volatility
        self.day = day
        self.week = week
        self.kappa = kappa
        self.max_steps = max_steps
        self.time_horizon = time_horizon
        self.initial_stock_price = initial_stock_price
        self.stock_price = initial_stock_price
        self.current_shares = 0

        self.step_count = 0
        self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0
                                                              #, 0.0
                                                              ]),
                                                high=np.array([float(initial_stock_price) * 10.0, time_horizon
                                                               #, 1.0
                                                               ]))
        self.action_space = gym.spaces.Box(low=0, high=1, shape =(1,))
        self.q=0.
        self.sigma0=self.sigma
        self.symbol = symbol
        self.rho=rho
        self.nu=nu
        self.dt = self.time_horizon/self.max_steps
        

    def reset(self):
        self.step_count = 0
        self.stock_price = self.initial_stock_price
        self.current_shares = 0
        time_to_maturity = self.time_horizon
        self.sigma = self.sigma0
        self.option_price = self.sim_option_price( self.stock_price, time_to_maturity, self.sigma, self.step_count)
        #print(self.option_price)
        self.initial_portfolio = self.option_price
        self.invested_portfolio= self.initial_portfolio
        sigma = self.sigma
        return np.array([self.initial_stock_price, time_to_maturity,
                         self.current_shares
                         ])
    
    

    def step(self, action):
        self.step_count += 1
        dt = self.time_horizon / self.max_steps
        
        time_to_maturity = self.time_horizon - self.step_count*dt
        
        
        prev_shares = self.current_shares
        self.current_shares = action[0]
        Z1=np.random.randn()
        Z2=self.rho*Z1+np.sqrt(1-self.rho**2)*np.random.randn()
        sigma_new=self.sigma+self.nu*self.sigma*np.sqrt(dt)*Z2
        prev_shares = self.current_shares
        self.current_shares = action[0]
        
        new_stock_price = self.stock_price +self.risk_free_rate*self.stock_price*dt + sigma_new*self.stock_price*np.sqrt(dt)*Z1
        
        new_option_price = self.sim_option_price( new_stock_price, time_to_maturity, sigma_new, self.step_count)

        option_change = (new_option_price - self.option_price)
        
        stock_value_change =  self.current_shares *( new_stock_price -self.stock_price)
        
        reward = -np.abs(stock_value_change + option_change)
        reward -= self.kappa*((self.current_shares - prev_shares )**2)*(new_stock_price)
        self.stock_price = new_stock_price
        self.option_price = new_option_price
        
        done = self.step_count >= self.max_steps
        next_state = np.array([self.stock_price, time_to_maturity
                              , self.current_shares
                              ])
        prev_shares = self.current_shares
        self.sigma = sigma_new
        return next_state, reward, done, {}

    def get_option_price(self, time_to_maturity, new_stock_price, sigma):
        d1 = (np.log(new_stock_price / self.strike_price) +
              (self.risk_free_rate + 0.5 * sigma ** 2) * time_to_maturity) / (
                     sigma * np.sqrt(time_to_maturity))
        d2 = d1 - sigma * np.sqrt(time_to_maturity)

        put_option_price =  self.strike_price * np.exp(
            -self.risk_free_rate * time_to_maturity) * norm.cdf(-d2) -new_stock_price * norm.cdf(-d1) 
        delta = norm.cdf(-d1)
        return put_option_price,delta
    
    def normalize_state(self, state):
        normalized_state = np.array([
            (state[0] - self.initial_stock_price) / (self.initial_stock_price ),  # Normalize stock price
            state[1] / self.time_horizon,  # Normalize time to maturity
            state[2]  # Current shares
        ])
        
        return normalized_state
    def load_exercise_boundary_from_csv(self, filename):
        time_points = []
        exercise_boundary = []
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                time_points.append(float(row['Time']))
                exercise_boundary.append(float(row['Exercise Boundary']))
        return time_points, exercise_boundary
    def sim_option_price(self, stock_price, time_to_maturity, sigma, step):
        strike_price =self.strike_price

        option_prices = []
        num_paths = 10000
        num_steps = self.max_steps - step +1
        option_exercised = False



        
        Z1 = np.random.normal(size=(num_steps, num_paths))
        Z2 = np.zeros((num_steps, num_paths))

        Sout=np.full((num_paths, num_steps),np.nan)
        Sigma=np.full(Sout.shape,np.nan)
        Sout[:,0]=stock_price
        Sigma[:,0]=sigma
        Z1=np.random.randn(num_paths,num_steps)
        Z2=self.rho*Z1+np.sqrt(1-self.rho**2)*np.random.randn(num_paths,num_steps)
        if self.symbol is None:
            exercise_boundary_filename = 'cheb_fit2.csv' 
        else:
            exercise_boundary_filename = f'{self.symbol}_fit{self.strike_price}-{self.week}_2.csv'  
        time_points, exercise_boundary = self.load_exercise_boundary_from_csv(exercise_boundary_filename)
        new_time_points = np.linspace(0, self.time_horizon, self.max_steps)
        
    # Interpolate the exercise boundary values at the new time points
        self.exercise_boundary = np.interp(new_time_points, time_points, exercise_boundary)
        self.exercise_boundary = self.exercise_boundary[step:]
        #print(self.exercise_boundary)
        for i in range(1,num_steps):
            Sigma[:,i]=Sigma[:,i-1]+self.nu*Sigma[:,i-1]*np.sqrt(self.dt)*Z2[:,i]
            Sout[:,i]=Sout[:,i-1]+(self.risk_free_rate)*Sout[:,i-1]*self.dt+Sigma[:,i-1]*Sout[:,i-1]*np.sqrt(self.dt)*Z1[:,i]
       
        # Vectorized calculation for new_stock_prices


        for j in range(num_paths):
            option_exercised = False

            for i in range(0, num_steps-1):  # Changed the loop range
                # Check if the stock price hits the exercise boundary
                
                if Sout[j, i] <= self.exercise_boundary[i]:
                    
                    #print(S[i,j])
                    option_prices.append(max(strike_price - Sout[j, i], 0))
                    option_exercised = True
                    break

            # If the option was not exercised, calculate the option price at maturity
            if not option_exercised:
                option_prices.append(max(strike_price - Sout[j, -1], 0))
        
        return np.mean(option_prices)*np.exp(-self.risk_free_rate*time_to_maturity)
