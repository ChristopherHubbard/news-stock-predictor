import imp
import sys
sys.modules["sqlite"] = imp.new_module("sqlite")
sys.modules["sqlite3.dbapi2"] = imp.new_module("sqlite.dbapi2")
import boto3

# Probably will just make a crawling microservice eventually
from ConfigManager import ConfigManager
from StockDataCollector import StockDataCollector

from datetime import datetime, timedelta

# Run a crawl for this symbol -- this should be a post request, since it updates the Dynamo table
def run_crawl(event, context):

    # Create the required config -- might use context instead of event
    config = ConfigManager(event['env'])

    # Create a stock data collector
    collector = StockDataCollector()

    # Get which symbol should be crawled
    symbol = event['symbol']

    # Get the headlines
    headlines = collector.collectHeadlinesForSymbol(symbol, pages=10)

    # Create dynamo connection
    dynamoConfig = config.config['Dynamodb']

    if event['env'] == 'LOCAL':
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'], endpoint_url=dynamoConfig['Endpoint'])
    else:
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'])

    # Store into DB
    collector.storeHeadlinesForSymbol(symbol, headlines, dynamo)

    # Return the headlines -- just to know what was posted
    return {
        'symbol': symbol,
        'headlines': headlines
    }

# Function to initialize the company objects in the dynamo database -- needs to be done to setup the table
def initializeCompanies(event, context):

    # Create the required config -- might use context instead of event
    config = ConfigManager(event['env'])

    # Create a stock data collector
    collector = StockDataCollector()

    dynamoConfig = config.config['Dynamodb']
    if event['env'] == 'LOCAL':
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'], endpoint_url=dynamoConfig['Endpoint'])
    else:
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'])

    # Store the company info
    collector.storeAllCompanyInfo(dynamo)

# Get headlines for this symbol -- store in the dynamo table
def getHeadlines(event, context):

    # Create the required config -- might use context instead of event
    config = ConfigManager(event['env'])

    symbol = event['symbol']

    # Create a stock data collector
    collector = StockDataCollector()

    dynamoConfig = config.config['Dynamodb']
    if event['env'] == 'LOCAL':
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'], endpoint_url=dynamoConfig['Endpoint'])
    else:
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'])

    # Get the headlines from the dynamo table
    headlines = collector.getHeadlinesForSymbol(symbol, dynamo)

    # Return the headlines from the call
    return {
        'symbol': symbol,
        'headlines': headlines
    }

# Post request to remove headlines that have expired from the database
def removeExpiredHeadlines(event, context):

    # Handler to remove any headlines that are over 30 days old -- won't factor into prediction

    # Create the required config -- might use context instead of event
    config = ConfigManager(event['env'])

    # Create dynamo config
    dynamoConfig = config.config['Dynamodb']
    if event['env'] == 'LOCAL':
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'], endpoint_url=dynamoConfig['Endpoint'])
    else:
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'])

    # Now scan and delete
    collector = StockDataCollector()
    stockInfo = collector.getAllCompanyInfo(fromDB=True, dynamo=dynamo)

    for stock in stockInfo:

        # Go through the headlines -- should be quick operation
        stock['headlines'] = [headline for headline in stock['headlines'] if not datetime.strptime(headline['date'], '%Y-%m-%d').date() < datetime.now().date() - timedelta(days=30)]

        # Update the stock in the Dynamo table -- should update with stale entries removed
        collector.setHeadlinesForSymbol(stock['symbol'], stock['headlines'], dynamo)

    print('Removed dead headlines')

# Handler to get all the company info -- name/symbol -- from dynamo table
def getCompanies(event, context):
    # Create the required config -- might use context instead of event
    config = ConfigManager(event['env'])

    # Create dynamo config
    dynamoConfig = config.config['Dynamodb']
    if event['env'] == 'LOCAL':
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'], endpoint_url=dynamoConfig['Endpoint'])
    else:
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'])

    collector = StockDataCollector()

    # Now format and return
    return {
        'companies': collector.getAllCompanyInfo(fromDB=True, dynamo=dynamo)
    }

if __name__ == '__main__':

    run_crawl(event={
        'symbol': 'MSFT',
        'env': 'DEV'
    }, context={})