The monte carlo simulation runs many trials on many possible 
market conditions that follows a "drift" and "shock" model. 
A "drift" is a normalized oscillation within reasonable bounds 
of random average movement around each datapoint in the dataframe 
provided for the simulation which has somewhat similar behavior 
with mean-reversion.
A "shock" is a calculation that sends the price randomly up or down 
to simulate market conditions such as bad news or good news. 
The drift and shock approach covers the possibilities of trends.
Up, down, and consolidation (sideways). 

This allows us to test the robustness of a trading strategy 
against 5000 (or more) possible market 'conditions' and provides us 
quantifiable values listed here:
1.) Percentage Change
2.) Average percentage change
3.) Logarithmic return
4.) Number of trials (loops)
5.) Maximum Sharpe Ratio
6.) Maximum Sharpe Return 
7.) Maximum Sharpe Volatility 
8.) Optimized Return 
9.) Optimized Volatility
10.) Optimized Sharpe Ratio
11.) Our initial position allocation guess 
12.) Optimized Position Allocations

Provides a Histogram of the logarithmic return value
and 
a scatter plot of the monte carlo simulation 
Main Plot:   xaxis = Volatility, yaxis = Returns, ColoredBy = Sharpe Ratio 
Red target:  xaxis = max sharpe volatility, yaxis = max sharpe return
Green Line:  xaxis = efficient frontier volatility, yaxis = efficient frontier return
