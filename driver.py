
import TrainingLoop as Train
import TestingLoop as Test
import numpy as np

def GBM_experiments_train():
    sabr = False
    symbol = None
    initial_stock_price =100
    strike =100
    time_horizon=1
    rho = 0
    nu = 0
    day = 365
    volatility = 0.2
    Train.TrainingLoop(symbol, sabr, strike, volatility, rho, nu, day, initial_stock_price)

def GBM_experiments_test():
    sabr = False
    symbol = None
    initial_stock_price =100
    strike =100
    time_horizon=1
    rho = 0
    nu = 0
    day = 365
    volatility = 0.2
    N =100
    tc = 3
    path = 'TC_Put42'
    real = False
    prices = None
    Test.TestingLoop(symbol, sabr, strike, volatility, rho, nu, day, initial_stock_price, N, tc,prices, path, real)


def sabr_experiment1_train():
    sabr = True
    symbol = None
    initial_stock_price =100
    strike =100
    time_horizon=31/365
    rho = -0.4
    nu = 0.1
    day = 31
    volatility = 0.2
    Train.TrainingLoop(symbol, sabr, strike, volatility, rho, nu, day, initial_stock_price)

def sabr_experiment1_test():
    sabr = True
    symbol = None
    initial_stock_price =100
    strike =100
    time_horizon=31/365
    rho = -0.4
    nu = 0.1
    day = 31
    volatility = 0.2
    N= 21
    tc = 3
    path = 'TC_Put42_cheb_1m' ## update with sabr agent
    real =False
    prices = None
    Test.TestingLoop(symbol, sabr, strike, volatility, rho, nu, day, initial_stock_price, N, tc, prices,path, real)

def market_calibrated_DRL_train():
    symbols = ['AAPL'] ## add symbols here
    for symbol in symbols:
            sabr = True
            import pandas as pd
            x = f'{symbol}.csv'
            data = pd.read_csv(x)
            market_vols = np.array(data['IVM']/100)
            strikes = np.array(data['Strike'].unique())
            x = f'{symbol}_sabr_params.csv'
            pars =pd.read_csv(x)
            rho = pars['rho'].mean()
            nu =pars['nu'].mean()
            days = data['Days'].unique()
            for market_vol in market_vols:
                    option = data[data['IVM']/100==market_vol].iloc[0]
                    day  = option['Days']
                    strike = option['Strike']
                    pname = f'{symbol}_prices.csv'
                    prices = pd.read_csv(pname)
                    prices =np.array(prices['Price'])
                    initial_stock_price= prices[0]
                    Train.TrainingLoop(symbol, sabr, strike, market_vol, rho, nu, day, initial_stock_price)


def market_calibrated_DRL_test():
    symbols = ['AAPL'] ## add symbols here
    for symbol in symbols:
            sabr = True
            import pandas as pd
            x = f'{symbol}.csv'
            data = pd.read_csv(x)
            market_vols = np.array(data['IVM']/100)
            x = f'{symbol}_sabr_params.csv'
            pars =pd.read_csv(x)
            rho = pars['rho'].mean()
            nu =pars['nu'].mean()

            for market_vol in market_vols:
                    option = data[data['IVM']/100==market_vol].iloc[0]
                    day  = option['Days']
                    strike = option['Strike']
                    pname = f'{symbol}_prices.csv'
                    prices = pd.read_csv(pname)
                    prices =np.array(prices['Price'])
                    if day == 29:
                        prices = prices[0:21]
                    initial_stock_price= prices[0]
                    path = f'{symbol}{strike}-{day}'
                    N=  len(prices)
                    tc=3
                    real =True
                    print(market_vol)
                    Test.TestingLoop(symbol, sabr, strike, market_vol, rho, nu, day, initial_stock_price,N, tc,prices, path, real  )

GBM_experiments_train()