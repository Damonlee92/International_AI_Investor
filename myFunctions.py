import pandas as pd
import requests # Making API requests to url
import json # for reading json file format
from io import StringIO # read string as file for pandas
import requests

def get_eod_data(startDate, endDate, symbol='MCD.US', api_token='OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX', session=None):
    '''
    Make API call to EODHistoricalData to get financial price series for a ticker.
    Returns a DataFrame.
    e.g.
    testData = get_eod_data('2016-01-01','2016-05-30', symbol='MCD.US', api_token='OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX')
    testData = get_eod_data('2016-01-01','2016-05-30', symbol='LPPF.JK', api_token=myToken)
    
    On rare occasion a ticker isn't available, return 1's as if just holding cash: a compromise for simplicities sake.
    '''
    if session is None:
        session = requests.Session()
    url = 'https://eodhistoricaldata.com/api/eod/%s' % symbol
    params = {'api_token': api_token, 'from':startDate, 'to':endDate}
    r = session.get(url, params=params)
    if r.status_code == requests.codes.ok:
        df = pd.read_csv(StringIO(r.text), skipfooter=1, parse_dates=[0], engine='python', index_col=0)
        if df.shape[0] == 0: # in case of error return constant price.
            df=pd.DataFrame(index=pd.date_range(start=startDate, end=endDate), columns=['Adjusted_close'])
            df['Adjusted_close']=1
        return df
    else:
        #raise Exception(r.status_code, r.reason, url)
        with open("StockWithIssues.txt", "a") as myfile:
            myfile.write(symbol+'\n')
            myfile.close()
        #return pd.DataFrame()
        tempDf=pd.DataFrame(index=pd.date_range(start=startDate, end=endDate), columns=['Adjusted_close'])
        tempDf['Adjusted_close']=1
        return tempDf


def get_exchanges_list(api_token='OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX', session=None): #Move to myFunctions.py
    '''
    Get list of stock exchanges. 
    Returns a list of dictionaries.
    '''
    if session is None:
        session = requests.Session()
    url = 'https://eodhistoricaldata.com/api/exchanges-list/'
    params = {'api_token': api_token, 'fmt': json}
    r = session.get(url, params=params)
    if r.status_code == requests.codes.ok:
        return json.loads(r.text)
    else:
        raise Exception(r.status_code, r.reason, url)


def get_stocks_list(exchange_symbol='LSE', api_token='OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX', session=None): #Move to myFunctions.py
    '''
    Returns list of stocks available for a given exchange symbol.
    Returns a DataFrame.
    '''
    if session is None:
        session = requests.Session()
    url = 'https://eodhistoricaldata.com/api/exchange-symbol-list/%s' % exchange_symbol
    params = {'api_token': api_token}
    r = session.get(url, params=params)
    if r.status_code == requests.codes.ok:
        df = pd.read_csv(StringIO(r.text), skipfooter=1, engine='python')
        return df
    else:
        raise Exception(r.status_code, r.reason, url)


def get_fundamental_data(symbol='MCD.US', api_token='OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX', 
                         session=None, continueWhenError=False):
    '''Get fundamental data as a dictionary.'''
    if session is None:
        session = requests.Session()
    url = 'https://eodhistoricaldata.com/api/fundamentals/'+ symbol
    params = {'api_token': api_token}
    r = session.get(url, params=params)
        
    if r.status_code == requests.codes.ok:
        return json.loads(r.text) # returns a dictionary
    else:
        if continueWhenError:
            with open("get_fundamental_data_issues.txt", "a") as myfile:
            	myfile.write(symbol+' '+str(r.status_code)+\
			     ' '+str(r.reason)+' '+str(url)+'\n')
            	myfile.close()
        else:
            raise Exception(r.status_code, r.reason, url)


def getXandyFundamentalStocksAnnual(stockData, priceData, verbose=False):
    '''
    Takes in the raw stockData and priceData DataFrames.
    Returns the features for stock selection as a DataFrame.
    Does not edit stockData or priceData input.
    Features data will have index corresponding to stockData input list.
    '''
    if verbose:
        print('priceData shape before filter',priceData.shape)
        print('stockData shape before filter',stockData.shape)
    priceData = priceData[priceData['startAdjusted_close']>0] # buggy data
    priceData = priceData[priceData['endAdjusted_close']>0]   # buggy data
    priceData = priceData.dropna()
    stockData = stockData.loc[priceData.index]
    stockData['Market Cap'] = (stockData['SharesOutstanding'] * priceData['startClose']).clip(0,2e13)
    stockData=stockData[stockData['Market Cap']>0] # Imperfect data
    #stockData=stockData[stockData['netIncome']>0] # Want only companies that earn something
    stockData=stockData[stockData['totalRevenue']>0] # Want only companies that have revenue
    stockData=stockData[stockData['Exchange']!='PINK'] # don't want pink sheets
    stockData=stockData[stockData['CountryName']!='Germany'] # many overseas stock listings
    stockData=stockData[stockData['Industry']!='Other Industrial Metals & Mining'] # dodgy mining companies
    stockData=stockData[stockData['Industry']!='Oil & Gas E&P'] # dodgy oil and gas companies
    priceData = priceData.loc[stockData.index]
    if verbose:
        print('priceData shape after filter',priceData.shape)
        print('stockData shape after filter',stockData.shape)

    # Data housekeeping
    stockData.fillna(0, inplace=True)
    stockData['interestExpense']=stockData['interestExpense'].apply(np.abs)
    stockData['cashAndShortTermInvestments'] = np.where(stockData['cashAndShortTermInvestments']<stockData['cash'], 
                                                       stockData['cash'],
                                                       stockData['cashAndShortTermInvestments'])
    stockData['totalCurrentLiabilities'] = np.where(stockData['totalCurrentLiabilities']<stockData['otherCurrentLiab'], 
                                                       stockData['otherCurrentLiab'],
                                                       stockData['totalCurrentLiabilities'])
    stockData['totalCurrentAssets'] = np.where(stockData['totalCurrentAssets']<stockData['otherCurrentAssets'], 
                                                       stockData['otherCurrentAssets'],
                                                       stockData['totalCurrentAssets'])
    #pd.DataFrame(stockData['totalCurrentAssets'] / stockData['totalAssets']).clip(0,1).mean() # 0.288
    stockData['totalCurrentAssets'] = np.where(stockData['totalCurrentAssets']==0, 
                                                       stockData['totalAssets']*0.288,
                                                       stockData['totalCurrentAssets'])
    stockData['nonCurrentAssetsTotal'] = np.where(stockData['totalCurrentAssets']==0, 
                                                       stockData['totalAssets']-stockData['totalCurrentAssets'],
                                                       stockData['nonCurrentAssetsTotal'])

    stockData['totalStockholderEquity'] = stockData['totalAssets'] - stockData['totalLiab']
    stockData['grossProfit'] = stockData['totalRevenue']-stockData['costOfRevenue']

    # StockData enhancement
    stockData["EBIT"] = stockData["netIncome"] \
        - stockData["interestExpense"] \
        - stockData["incomeTaxExpense"]

    stockData['enterpriseValue'] = stockData['Market Cap']\
        +stockData['longTermDebtTotal']\
        +stockData['shortTermDebt']\
        -stockData['cashAndShortTermInvestments']

    stockData['1YPerf'] = (priceData['endAdjusted_close'] - priceData['startAdjusted_close'])/priceData['startAdjusted_close']
    stockData['1YPerf'] = stockData['1YPerf'].clip(-1,10)

    # Getting features
    features = pd.DataFrame()
    features['1YPerf'] = stockData['1YPerf']
    features['P/E'] = (stockData['Market Cap']/stockData['netIncome']).clip(-1000,5000)
    features['P/S'] = (stockData['Market Cap']/stockData['totalRevenue']).clip(0,5000)
    features['P/B'] = (stockData['Market Cap']/stockData['totalStockholderEquity']).clip(-100,1000)

    features['RoE'] = (stockData['netIncome']/stockData['totalStockholderEquity']).clip(-10,10)
    features['ROCE'] = (stockData['EBIT']/(stockData['totalAssets']-stockData['totalCurrentLiabilities'])).clip(-2,2)
    features['grossProfitMargin'] = (stockData['grossProfit']/stockData['totalRevenue']).clip(-10,10)

    #features['assetTurnover'] = (stockData['propertyPlantEquipment']/stockData['totalRevenue']).clip(0,100)
    features['fixedAssetTurnover'] = (stockData['totalRevenue']/stockData['propertyPlantEquipment']).clip(0,300)
    features['workingAssetTurnover'] = (stockData['totalRevenue']\
                                        /(stockData['totalCurrentAssets']-stockData['totalCurrentLiabilities'])).clip(-200,200)

    features['workingCapitalRatio'] = (stockData['totalCurrentAssets']/stockData['totalCurrentLiabilities']).clip(-5,100)
    #features['BookEquity/TL'] = (stockData['totalStockholderEquity']/stockData['totalLiab']).clip(-10,100)
    features['Debt/Equity'] = (stockData['totalLiab']/stockData['totalStockholderEquity']).clip(-20,100)
    features['cashRatio'] = (stockData['cashAndShortTermInvestments']/stockData['totalCurrentLiabilities']).clip(0,50)
    features['debtRatio'] = (stockData['totalAssets']/stockData['totalLiab']).clip(-100,100)

    features['EV/EBIT'] = (stockData['enterpriseValue'] / stockData["EBIT"]).clip(-50000,50000)


    ### Greenblatt ratios ###
    features['Op. In./(NWC+FA)'] = (stockData['operatingIncome'] /\
                                (stockData['totalCurrentAssets']-stockData['totalCurrentLiabilities']
                                +stockData['propertyPlantEquipment'])).clip(-20,20)
    features['Op. In./interestExpense'] = (stockData['operatingIncome']/stockData['interestExpense']).clip(-1000,1000)

    ### Altman ratios ###
    features['EBIT/TA'] = (stockData['EBIT']/stockData['totalAssets']).clip(-2,2)
    features['RE/TA'] = (stockData['retainedEarnings']/stockData['totalAssets']).clip(-20,5)

    features['(CA-CL)/TA'] = ((stockData['totalCurrentAssets']\
                       - stockData['totalCurrentLiabilities'])\
                        /stockData['totalAssets']).clip(-4,4)
    
    return features

def calcZScores(X):
    '''
    Calculate Altman Z'' scores 1995.
    Basically capacity to pay interest.
    Trouble < 2 < Grey Zone < 3 < Safe < 4 < Super safe
    '''
    Z = pd.DataFrame(index=X.index)
    Z = 3.25\
        + 6.51 * X['(CA-CL)/TA']\
        + 3.26 * X['RE/TA']\
        + 6.72 * X['EBIT/TA']\
        + 1.05 * (1/X['Debt/Equity'])
    return Z

#################################################
# Stock Picker Functions
#################################################

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

def newSKLearnRegressorModelAndPrediction(X_train, y_train, X_test, 
                                          randomState=42, 
                                          modelName='randomForest'):
    '''
    Contains all regression models for stock selection
    '''
    if modelName == 'randomForest':
        model = RandomForestRegressor(random_state=randomState, 
                                      n_estimators=100,
                                      max_depth=30).fit(X_train,
                                                        y_train.values.ravel())  
    elif modelName == 'KNN':
        model = Pipeline([('powerTransformer', PowerTransformer()),
                          ('knn', KNeighborsRegressor(n_neighbors=40))]).fit(X_train, 
                                                        y_train.values.ravel())  
        
    elif modelName == 'extraTrees':
        model = ExtraTreesRegressor(random_state=randomState, 
                                    n_estimators=100, 
                                    max_depth=200).fit(X_train, #maxdepth30
                                                        y_train.values.ravel())  
    
    elif modelName == 'gradBoost':
        model = GradientBoostingRegressor(random_state=randomState,
                                          learning_rate=0.1,
                                          n_estimators=300,
                                          subsample=1.0,
                                          max_features=None,
                                          max_depth=10).fit(X_train, 
                                                                 y_train.values.ravel()) 
    
    y_pred = model.predict(X_test)
    return y_pred

def fundamentalStockPicker(windowDateFrom, windowDateTo, 
                           X_fundamentalData, y_performanceData, 
                           stockData, priceData,
                           country=None, zScore=None, randomState=42, 
                           modelName='randomForest',
                           numberOfStocks=10):
    '''
    Select stocks in the given date window, using information from X and y before 
    the investment window start date to train the model.
    The investment window is a few months backwards as this is when the actual 
    data would be available for us (publish date).
    
    Stock selection done with random forest by default.
    
    Returns DataFrame of prediction, target, stock ticker, exchange, 
    dates and prices bought/sold of best predicted returns
    '''
    # Boolean mask of stocks before the investment window, that
    # can be used as training data
    beforeWindow = stockData['date']<\
                    pd.to_datetime(windowDateFrom)-pd.Timedelta(days=120) 
    
    # Boolean mask of stocks within the investment window, that
    # we can select stocks from for investment
    inWindow = stockData['date'].between(\
                     pd.to_datetime(windowDateFrom)-pd.Timedelta(days=120),\
                     pd.to_datetime(windowDateTo)-pd.Timedelta(days=120))
    
    X_train = X_fundamentalData.loc[beforeWindow]
    y_train = y_performanceData.loc[beforeWindow]

    X_test = X_fundamentalData.loc[inWindow]
    y_test = y_performanceData.loc[inWindow]
    
    # Filter by country if desired
    if country:
        X_test = X_test.loc[stockData['CountryName']==country]
        y_test = y_test.loc[stockData['CountryName']==country]
        
    # Filter stock selection by Z-Score if desired
    if zScore:
        X_test=X_test.loc[calcZScores(X_test) > zScore]
        #y_test=y_test.loc[X_test.index]
        y_test=y_test.reindex(X_test.index)
    
    # Now select which regression model to use
    if modelName == 'neuralNet':
        y_pred = newNNRegressorModelAndPrediction(X_fundamentalData, 
                                                  y_performanceData, 
                                                  X_train, y_train, X_test)
    elif modelName in ['randomForest', 'extraTrees', 'KNN', 'gradBoost']:
        y_pred = newSKLearnRegressorModelAndPrediction(X_train, y_train, X_test, 
                                                       randomState=randomState,
                                                       modelName=modelName)
    else:
        y_pred = np.random.rand(X_test.shape[0]) # Random seleciton for testing
    
    # Create the output DataFrame containing ancillary data and the 
    # predicted stock performance. We select the best stocks from here.
    listOfPickedStocks = pd.DataFrame()
    listOfPickedStocks['Prediction'] = y_pred
    listOfPickedStocks.index = X_test.index
    listOfPickedStocks['Target'] = y_test
    listOfPickedStocks['Code'] = stockData['Code'].loc[X_test.index]
    listOfPickedStocks['Exchange'] = stockData['Exchange'].loc[X_test.index]
    
    listOfPickedStocks['buyDate'] = \
    pd.to_datetime(priceData.loc[X_test.index]['startDate'])
    listOfPickedStocks['sellDate'] = \
    pd.to_datetime(priceData.loc[X_test.index]['endDate'])
    
    listOfPickedStocks['buyPrice'] = priceData.loc[X_test.index]['startClose']
    listOfPickedStocks['sellPrice'] = priceData.loc[X_test.index]['endClose']
    
    listOfPickedStocks['startAdjusted'] = \
    priceData.loc[X_test.index]['startAdjusted_close']
    listOfPickedStocks['endAdjusted'] = \
    priceData.loc[X_test.index]['endAdjusted_close']
    
    listOfPickedStocks['zScore'] = calcZScores(X_test)
    
    #print('Train size=', y_train.shape[0])
    #print('Test size=', y_test.shape[0])
    return listOfPickedStocks.sort_values(by=['Prediction'], ascending=False).head(numberOfStocks) # must be ascending = False.

def getPortfolioRelativeTimeSeries(stockRet):
    '''
    Normalise a dataframe of stock prices of a portfolio over time.
    As a column for the overall portfolio value change.
    '''
    stockRetRel = stockRet.copy()
    for key in stockRetRel.keys():
        stockRetRel[key]=stockRetRel[key]/stockRetRel[key][0]
        
    stockRetRel["Portfolio"] = stockRetRel.sum(axis=1)/stockRetRel.iloc[0].sum()#(stockRetRel.keys().shape[0])
    return stockRetRel


#################################################
# Modern Portfolio Theory Functions
#################################################

import scipy as sp
from scipy import optimize

def calcPortfolioVar(portfolio, weights=None):
    '''
    Input is portfolio price history DataFrame.
    Calculate the variance of the portfolio given the asset weighting.
    Assumes equal asset weighting if no weighting given.
    '''
    if weights is None:
        weights = [1 / portfolio.keys().size] * portfolio.keys().size
    
    #Calculate variance
    var = np.dot(np.dot(
                np.array(np.log(portfolio/portfolio.shift(-1)).cov()),
          weights),weights)
    
    return var

# Scipy has no maximize function, so invert Sharpe ratio and minimize.
def negativeSharpeRatio(weights, stockHistory, predReturns, risk_free_rate=0.05):
    '''
    Input DataFrame containing price history of all securities, also weighting,
    and the predicted returns for those securities.
    Calculates negative Sharpe Ratio.
    Assumes equal asset weighting if no weighting given.
    '''
    if weights is None: 
        weights = [1 / stockHistory.keys().size] * stockHistory.keys().size
    
    var = calcPortfolioVar(stockHistory, weights)
    portPredReturn = np.mean(predReturns * weights)
    return (risk_free_rate - portPredReturn) / np.sqrt(var) # negative

def optimisePortfolioMPT(stockHistory, predReturns, 
                         risk_free_rate=0.05, maxWeight=0.5):
    '''
    Use Markowitz portfolio optimisation to find the 'optimal' asset weighting.    
    By default the max. asset allocation is 50% of the portfolio.
    Use scipy.optimize to solve for asset allocation.
    https://docs.scipy.org/doc/scipy/reference/optimize.html
    https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
    
    Markowitz, H.M. (March 1952). "Portfolio Selection". 
    The Journal of Finance. 7 (1): 77–91. doi:10.2307/2975974. JSTOR 2975974
    '''
    # Set the initial weights for the portfolio
    initialWeights = [1 / stockHistory.keys().size] * stockHistory.keys().size 
    
    # Set the bounds for each of the variables we will allow the solver to use
    # list of (0, 0.5) tuples
    bounds = [(0, maxWeight) for i in range(stockHistory.keys().size)] 
    
    # Set the constraint that the portfolio weights have to sum to one.
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    
    # Call the SciPy optimisation function
    results = sp.optimize.minimize(negativeSharpeRatio, 
                                   initialWeights, 
                                   (stockHistory, predReturns, risk_free_rate),
                                   method='SLSQP',#'SLSQP' or 'trust-constr'
                                   constraints=constraints,
                                   bounds=bounds)
                            #, options={'disp': True}) # To see the printout
    
    return results.x
