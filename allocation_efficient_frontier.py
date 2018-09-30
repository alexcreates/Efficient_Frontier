import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


"""
    The 'Frontier Horizon Optimization' calculates a high amount of possible
    portfolio allocation possibilities on a historical data set(s) paired with a
    trading algo or multiple trading strategy algos, to calculate the peak optimization
    of the combined or singluar strategie(s). This would be considered the "Final-Frontier" a final
    mathematical evaluation of a singular or a set of trading strategies. This can be
    altered into a method, use conditions for strategy amounts provided, to fine
    tune the best possible outcomes. This file does not currently have a strategy
    built in the current default is to go long on four stocks and hold for a year.
    The evaluation of the efficient frontier calculates the best possible return
    value for each volatility value paired with a strategy / multiple strategies.
    This would be useful to remain flexible for either of the combinations:
    1.) one algo || one dataframe
    2.) one algo || multiple dataframes
    3.) multiple algos || one dataframe (rare)
    4.) multiple algos || multiple dataframes
    This is a frame work for an allocation and strategy returns analysis 
"""

###############################
#       Load Data Frames      #
###############################
# Assuming we have the data frames stored.
# These are series of Adj. Close Values 5 years back, each
Apple = pd.read_csv('/Target file path', index_col='Date', parse_dates=True)
Cisco = pd.read_csv('/Target file path', index_col='Date', parse_dates=True)
Ibm = pd.read_csv('/Target file path', index_col='Date', parse_dates=True)
Amazon = pd.read_csv('/Target file path', index_col='Date', parse_dates=True)


stocks = pd.concat([Apple, Cisco, Ibm, Amazon], axis=1)
stocks.columns = ['Apple', 'Cisco', 'Ibm', 'Amazon']

percent_change = stocks.pct_change(1)
average_pct_change = stocks.pct_change(1).mean()


# If there is more than one data set
# go ahead and check for any correlations
# for further research
if len(stocks) > 1:
    df_correlation_result = stocks.pct_change(1).corr()

# calculate the logarithmic return of each data set
logarithmic_return = np.log(stocks/stocks.shift(1))


####################################
####################################
#      Monte Carlo Simulation      #
####################################
####################################
#  seed random values so they are consistent
np.random.seed(101)

# set loop count 5000 random portfolio allocations
num_ports = 5000

# initialize arrays to hold the data generated
all_weights = np.zeros((num_ports, len(stocks.columns)))
return_arr = np.zeros(num_ports)
volatility_arr = np.zeros(num_ports)
sharpe_ratios_arr = np.zeros(num_ports)


for idx in range(num_ports):
    # Create the weights with seeded random values and rebalance them
    weights = np.array(np.random.random(4))
    weights = weights/np.sum(weights)

    # Save the weights into the initialized array using pandas broadcasting ---> ( : = "all" )
    all_weights[idx, :] = weights

    # create and store the expected return ----- 252 is a trading year
    return_arr[idx] = np.sum( (logarithmic_return.mean() * weights) * 252)

    # create the expected volatility
    # this calculates the square root of
    # 1.) the dot product of the weights transposed multiplied by the rebalanced weights
    # 2.) and the dot prodcut of the covariance of the logarithmic return multiplied by the length of time the test will run here its 252 for a trading year
    volatility_arr[idx] = np.sqrt(np.dot(weights.T, np.dot(logarithmic_return.cov() * 252, weights)))

    #  create the sharpe ratio
    sharpe_ratios_arr[idx] = return_arr[idx] / volatility_arr[idx]

#  store the max sharpe ratio result from the simulation
max_sharpe_result  = sharpe_ratios_arr.max()
#  store the max return and volatility from the simulation
max_sharpe_return = return_arr[sharpe_ratios_arr.argmax()]
max_sharpe_volatility = volatility_arr[sharpe_ratios_arr.argmax()]

# store optimized position allocation percentages for maximum return from strategy
optimized_position_alloc = all_weights[sharpe_ratios_arr.argmax(), :]



# get Return, Volatility, Sharpe Ratio
def get_RVS(weights):
    weights = np.array(weights)
    returns = np.sum(logarithmic_return.mean() * weights) * 252
    volatility = np.sqrt(np.dot(weights.T, np.dot(logarithmic_return.cov() * 252, weights)))
    sharpe_ratio = (returns / volatility)
    return np.array([returns, volatility, sharpe_ratio])

# helper function
# minimalize the sharpe ratio
# [2] locates the sharpe_ratio value in the return of the get_RVS method just above ^^
def negative_sharpe(weights):
    return get_RVS(weights)[2] * -1

# helper function / error checking for percentage allocation of users account.
# returns 0 if the sum of your positions equals 1 (%100)
def check_sum(weights):
    return np.sum(weights) - 1


# constraint type = equation, function = check_sum,
constraints = ({'type': 'eq', 'fun': check_sum})
# set the bounds to not exceed 0 or 1 for each portfolio allocation
bounds = ((0,1), (0,1), (0,1), (0,1))
# create your initial guess of best portfolio allocation percentages
#  default = [0.25, 0.25, 0.25, 0.25]
init_guess = [0.25, 0.25, 0.25, 0.25]
# Calculate the optimized RVS results  -----> RVS = Return, Volatility, Sharpe Ratio
# the method value selects the algo from scipy to minimize
# Sequential least squares
optimized_results = minimize(negative_sharpe, init_guess, method='SLSQP', bounds = bounds, constraints = constraints)
# Store the optimized Return, Volatility, Sharpe Ratio results
optimized_RVS = get_RVS(optimized_results.x)
# optimized_RVS =  an array ['Returns', 'Volatility', 'Sharpe Ratio']



##############################
####   Efficient Frontier  ###
##############################
# frontier_y = Y axis bounds
frontier_y = np.linspace(0, 0.3, 100)

def minimize_volatility(weights):
    return get_RVS(weights)[1]

frontier_volatility = []
for possible_return in frontier_y:
    constraints = ({'type': 'eq', 'fun': check_sum},
                   {'type': 'eq', 'fun': lambda w: get_RVS(w)[0] - possible_return})
    result = minimize(minimize_volatility, init_guess, method='SLSQP', bounds = bounds, constraints = constraints)
    frontier_volatility.append(result['fun'])








####################################
####################################
#              Graphs              #
####################################
####################################


####################################
#     Graph: Logarithmic Return
####################################
# logarithmic_return.hist(bins=100, figsize(12,8))
plt.tight_layout()

####################################
#       Graph: Sharpe Ratio
#       Monte Carlo Strategy
#           Optimization
####################################
#   color by the sharpe ratio and use the plasma color scheme
#   to show the transition in value
plt.figure(figsize=(12,8))
plt.scatter(volatility_arr, return_arr, c=sharpe_ratios_arr, cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.text(-0.7, -0.25, 'sminimum')
# Add red dot for max SR
plt.scatter(max_sharpe_volatility, max_sharpe_return, c='red', s=50, edgecolors='black')

####################################
#     Graph: Efficient Frontier
#       Monte Carlo Strategy
#           Optimization
####################################

# Add frontier line
plt.plot(frontier_volatility, frontier_y,'g--', linewidth=1)

fig = plt
# save the plot as a .png file 
fig.savefig('/Target file path.png')




####################################
####################################
#           Tear Sheet             #
####################################
####################################
print 'Percent Change: ' + str(percent_change)
print 'Average Percent Change: ' + str(average_pct_change)
print 'Logarithmic Return: ' + str(logarithmic_return)
print 'Number of Portfolios Tested: ' + str(num_ports)
print 'Max Sharpe Ratio Result: ' + str(max_sharpe_result)
print 'Max Sharpe Return: ' + str(max_sharpe_return)
print 'Max Sharpe Volatility: ' + str(max_sharpe_volatility)
print 'Optimized Return: ' + str(optimized_RVS[0])
print 'Optimized Allocation Guess: ' + str(init_guess)
print 'Initial Allocation Guess: ' + str(init_guess)
print 'Optimized Return, Volatility, Sharpe Ratio: ' + str(optimized_RVS)
print 'Correlation Result: ' + str(df_correlation_result)
