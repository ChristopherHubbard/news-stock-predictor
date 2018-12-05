from iexfinance import get_historical_data, get_available_symbols

from NASDAQSpider import NASDAQSpider
from SpiderRunner import SpiderRunner

from datetime import datetime, timedelta

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