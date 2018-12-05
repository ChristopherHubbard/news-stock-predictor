from iexfinance import get_historical_data, get_available_symbols
import boto3

# Probably will just make a crawling microservice eventually
from NASDAQSpider import NASDAQSpider
from SpiderRunner import SpiderRunner
from ConfigManager import ConfigManager

from datetime import datetime, timedelta

# Run a crawl for this symbol -- this should be a post request, since it updates the Dynamo table
def run_crawl_handler(event, context):

    # Create the required config -- might use context instead of event
    config = ConfigManager(event['env'])

    # Create a stock data collector
    collector = StockDataCollector()

    # Get which symbol should be crawled
    symbol = event['symbol']

    # Get the headlines
    headlines = collector.collectHeadlinesForSymbol(symbol, pages=20)

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
def initializeCompanies_handler(event, context):

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
def getHeadlines_handler(event, context):

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
def removeExpiredHeadlines_handler(event, context):

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
def getCompanies_handler(event, context):
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

# This class collects a months worth of headlines for an symbol
class StockDataCollector():

    def __init__(self):

        self.spiderRunner = SpiderRunner(useCache=False)

    def getAllCompanyInfo(self, fromDB=False, dynamo=None):

        # Get the company information either from the DB or from IEX -- Should only need to IEX once
        companyInformation = {}
        if fromDB:

            # Retrieve from dynamo -- scan should work
            table = dynamo.Table('StockHeadlines')
            companyInformation = table.scan()['Items']
        else:

            # Get all the indices that the IEX finance API can server
            companyData = get_available_symbols()

            # Set up the company's information object
            for company in companyData:

                # Only include company values -- not dividends, crypto, ect
                if company['type'] == 'cs':
                    companyInformation[company['symbol']] = company['name']

        return companyInformation

    def storeAllCompanyInfo(self, dynamo):

        # Store all the company information from the company dictionary in Dynamo
        companyInfo = self.getAllCompanyInfo(fromDB=False)

        # Store into dynamo
        table = dynamo.Table('StockHeadlines')

        # Insert the entries
        for symbol, name in companyInfo.items():
            # Store in dynamo -- no headlines -- this only needs to be done on initialization
            table.put_item(
                Item={
                    'symbol': symbol,
                    'name': name,
                    'headlines': []
                }
            )

    def collectHeadlinesForSymbol(self, symbol, pages=10):

        # Collect the headlines for this symbol
        return self.spiderRunner.run_scrapydoProcess(NASDAQSpider, symbol, pages)

    def storeHeadlinesForSymbol(self, symbol, headlines, dynamo):

        # Store into dynamoDB
        table = dynamo.Table('StockHeadlines')

        # Need to convert datetime's
        for headline in headlines:
            headline['date'] = headline['date'].__str__()

        # Gets the stock for this symbol and only adds new headlines
        item = table.get_item(
            Key={
                'symbol': symbol
            }
        )['Item']

        for headline in headlines:

            # Only add if not already in headlines
            if headline not in item['headlines']:
                item['headlines'].append(headline)

        # Update the table
        table.update_item(
            Key={
                'symbol': symbol
            },
            UpdateExpression='SET headlines = :r',
            ExpressionAttributeValues={
                ':r': item['headlines']
            }
        )

    def getHeadlinesForSymbol(self, symbol, dynamo):

        # Get table
        table = dynamo.Table('StockHeadlines')

        # Get the headlines
        response = table.get_item(
            Key={
                'symbol': symbol
            }
        )

        stockInfo = response['Item']

        # Fix the headline dates
        for headline in stockInfo['headlines']:
            headline['date'] = datetime.strptime(headline['date'], '%Y-%m-%d').date()

        return stockInfo['headlines']

    def setHeadlinesForSymbol(self, symbol, headlines, dynamo):

        table = dynamo.Table('StockHeadlines')

        table.update_item(
            Key={
                'symbol': symbol
            },
            UpdateExpression='SET headlines = :r',
            ExpressionAttributeValues={
                ':r': headlines
            }
        )

if __name__ == '__main__':

    crawl_handler(event={
        'symbol': 'MSFT',
        'env': 'DEV'
    }, context={})