from iexfinance import get_historical_data, get_available_symbols
import boto3

# Probably will just make a crawling microservice eventually
from NASDAQSpider import NASDAQSpider
from SpiderRunner import SpiderRunner
from ConfigManager import ConfigManager

from datetime import datetime, timedelta

# Lambda handler function
def crawl_handler(event, context):

    # Create the required config -- might use context instead of event
    config = ConfigManager(event['env'])

    # Create a stock data collector
    collector = StockDataCollector()

    # Get which index should be crawled
    index = event['index']

    # Get the headlines
    headlines = collector.collectHeadlinesForIndex(index, pages=20)

    # Create dynamo connection
    dynamoConfig = config.config['Dynamodb']

    if event['env'] == 'LOCAL':
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'], endpoint_url=dynamoConfig['Endpoint'])
    else:
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'])

    # Store into DB
    collector.storeHeadlinesForIndex(index, headlines, dynamo)

    # Return the headlines
    return {
        'symbol': index,
        'headlines': headlines
    }

def getCompany_handler(event, context):

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

def getHeadlines_handler(event, context):

    # Create the required config -- might use context instead of event
    config = ConfigManager(event['env'])

    index = event['index']

    # Create a stock data collector
    collector = StockDataCollector()

    dynamoConfig = config.config['Dynamodb']
    if event['env'] == 'LOCAL':
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'], endpoint_url=dynamoConfig['Endpoint'])
    else:
        dynamo = boto3.resource('dynamodb', region_name=dynamoConfig['Region'])

    # Store the company info
    collector.getHeadlinesForIndex(index, dynamo)

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
    table = dynamo.Table('StockHeadlines')
    items = table.scan()['Items']

    for stock in items:

        # Go through the headlines -- should be quick operation
        stock['headlines'] = [headline for headline in stock['headlines'] if not datetime.strptime(headline['date'], '%Y-%m-%d').date() < datetime.now().date() - timedelta(days=30)]

        # Update the stock in the Dynamo table -- should update with stale entries removed
        table.update_item(
            Key={
                'symbol': stock['symbol']
            },
            UpdateExpression='SET headlines = :r',
            ExpressionAttributeValues={
                ':r': stock['headlines']
            }
        )

    print('Removed dead headlines')

# This class collects a months worth of headlines for an index
class StockDataCollector():

    def __init__(self):

        self.spiderRunner = SpiderRunner(useCache=False)

    def getAllCompanyInfo(self, fromDB=True, dynamo=None):

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
        for index, name in companyInfo.items():
            # Store in dynamo -- no headlines -- this only needs to be done on initialization
            table.put_item(
                Item={
                    'symbol': index,
                    'name': name,
                    'headlines': []
                }
            )

    def collectHeadlinesForIndex(self, index, pages=10):

        # Collect the headlines for this index
        return self.spiderRunner.run_scrapydoProcess(NASDAQSpider, index, pages)

    def storeHeadlinesForIndex(self, index, headlines, dynamo):

        # Store into dynamoDB
        table = dynamo.Table('StockHeadlines')

        # Need to convert datetime's
        for headline in headlines:
            headline['date'] = headline['date'].__str__()

        # Gets the stock for this index and only adds new headlines
        item = table.get_item(
            Key={
                'symbol': index
            }
        )['Item']

        for headline in headlines:

            # Only add if not already in headlines
            if headline not in item['headlines']:
                item['headlines'].append(headline)

        # Update the table
        table.update_item(
            Key={
                'symbol': index
            },
            UpdateExpression='SET headlines = :r',
            ExpressionAttributeValues={
                ':r': item['headlines']
            }
        )

    def getHeadlinesForIndex(self, index, dynamo):

        # Get table
        table = dynamo.Table('StockHeadlines')

        # Get the headlines
        response = table.get_item(
            Key={
                'symbol': index
            }
        )

        stockInfo = response['Item']

        # Fix the headline dates
        for headline in stockInfo['headlines']:
            headline['date'] = datetime.strptime(headline['date'], '%Y-%m-%d').date()

        return stockInfo['headlines']

if __name__ == '__main__':

    removeExpiredHeadlines_handler(event={
        'env': 'DEV'
    }, context={})

    crawl_handler(event={
        'index': 'MSFT',
        'env': 'DEV'
    }, context={})