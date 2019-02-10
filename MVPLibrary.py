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
	nAssets = dfCovar.shape[1]
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
