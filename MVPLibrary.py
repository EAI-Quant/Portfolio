#Import libraries

import numpy as np
import pandas as pd
import cvxpy as cp
import math


def mean_var_optimal_given_alpha(dfExpectedReturns, dfCovarReturns, alpha):
	'''mean_var_optimal_given_alpha takes a two pandas df objects, and a float alpha, returns the optimal portfolio weights
	INPUTS:
		dfMeans: pandas df object, should be 1 row by n columns.  The columns should be the names of the assets
		dfCovar: pandas df object, should be n x n.  Should have the column names as dfCovar
		alpha: minimum expected return allowed
	Outputs:
		3 values
		1st position: the expected return
		2nd position: the volatility of returns
		3rd position: the weights vector
	'''
	Q = dfCovarReturns.values
	expectedReturns = dfExpectedReturns.values
	nAssets = dfCovarReturns.shape[1]
	e = np.ones((nAssets,1))
	w = cp.Variable(nAssets)
	objective = cp.Minimize(cp.quad_form(w,Q))
	constraints = []
	constraints.append(0 <= w)
	constraints.append(w.T*e == 1)
	constraints.append(expectedReturns*w >= alpha)
	prob = cp.Problem(objective, constraints)
	result = prob.solve()
	#return expected return, expected vol, and the portfolio weights
	return (expectedReturns*w).value[0], math.sqrt(result), w.value



def calc_efficient_frontier(dfExpectedReturns, dfCovarReturns, riskFreeRate, nPoints = 1000, scaling=10):
	'''mean_var_optimal_given_alpha takes a two pandas df objects, and a float alpha, returns the optimal portfolio weights
	INPUTS:
		dfMeans: pandas df object, should be 1 row by n columns.  The columns should be the names of the assets
		dfCovar: pandas df object, should be n x n.  Should have the column names as dfCovar
		riskFreeRate: float, represents the risk free rate
		nPoints: int, number of point to plot in the efficient frontier
		scaling: float: number used to scale inputs to avoid overflow error
	Outputs:
		thing1: returns the optimal weights
		data: pandas df, three columns, expectedReturn, vol, and sharp ratio
	'''
	alphas = np.linspace(dfExpectedReturns.min(axis=1)[0], dfExpectedReturns.max(axis=1)[0], num=nPoints, endpoint=True)
	dfExpectedReturns = dfExpectedReturns*scaling
	dfCovarReturns = dfCovarReturns*(scaling**2)

	data = np.zeros((nPoints, 3))
	#loop over the data
	for i in range(nPoints):
		expectedReturn, vol, _ = mean_var_optimal_given_alpha(dfExpectedReturns, dfCovarReturns, scaling*alphas[i])
		data[i,0] = expectedReturn/scaling
		data[i,1] = vol/scaling
		data[i,2] = (data[i,0] - riskFreeRate)/(data[i,1])
	data = pd.DataFrame(data, columns=['Expected Return', 'Volatility', 'Sharp Ratio'])

	#Return Sharpe Ratio Optimal Weights, and print them
	scaledAlphaForMaxSharp = data.loc[data['Sharp Ratio'].idxmax()]['Expected Return']
	_, _, weights = mean_var_optimal_given_alpha(dfExpectedReturns, dfCovarReturns, scaledAlphaForMaxSharp)
	dfWeights = pd.DataFrame(weights)
	dfWeights = dfWeights.transpose()
	dfWeights.columns = list(dfCovarReturns.columns)

	return data, dfWeights
