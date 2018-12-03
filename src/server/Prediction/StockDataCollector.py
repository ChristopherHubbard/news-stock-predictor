from iexfinance import get_historical_data, get_available_symbols
from datetime import datetime, timedelta
from redis import Redis
import torch
import boto3
import pickle

# Probably will just make a crawling microservice eventually
from src.server.Crawler.NASDAQSpider import NASDAQSpider
from src.server.Crawler.SpiderRunner import SpiderRunner
from ConfigManager import ConfigManager

dynamodb = boto3.resource('dynamodb', region_name='us-west-2', endpoint_url="http://localhost:8000")

# Lambda handler function
def crawl_handler(event, context):

    # Create the required config -- might use context instead of event
    config = ConfigManager(event['env'])

    # Create a stock data collector
    collector = StockDataCollector()

    # Get which index should be crawled
    index = event['index']

    # Get the headlines
    headlines = collector.collectHeadlinesForIndex(index, pages=10)

    # Create dynamo connection
    dynamo = boto3.resource('dyanmodb', region_name=config.config['Region'], endpoint_url=config.config['Endpoint'])

    # Store into DB
    collector.storeHeadlinesForIndex(index, headlines, dynamo)

    # Return the headlines
    return {
        'index': index,
        'headlines': headlines
    }


# This class collects a months worth of headlines for an index
class StockDataCollector():

    def getAllCompanyInfo(self, fromDB=True, dynamo=None):

        # Get the company information either from the DB or from IEX -- Should only need to IEX once
        companyInformation = {}
        if fromDB:

            # Retrieve from dynamo -- scan should work
            table = dynamo.Table('Headlines')
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
        table = dynamo.Table('Headlines')

        # Insert the entries
        for index, name in companyInfo.values():

            # Store in dynamo -- no headlines -- this only needs to be done on initialization
            table.put_item(
                Item={
                    'index': index,
                    'name': name,
                    'headlines': []
                }
            )

    def collectHeadlinesForIndex(self, index, pages=10):

        # Collect the headlines for this index
        return self.spiderRunner.run_crawlProcess(NASDAQSpider, index, pages)

    def storeHeadlinesForIndex(self, index, headlines, dynamo):

        # Store into dynamoDB
        table = dynamo.Table('Headlines')

        # Update the table
        table.update_item(
            Key={
                'index': index
            },
            UpdateExpression='set headlines = :r',
            ExpressionUpdateValues={
                ':r': headlines
            }
        )

    def _collectHeadlinesForIndex(self, index, spiderRunner, useCache=True, pages=10):

        # Collect the headlines for the given index -- make sure has timestamp -- Use NASDAQ website?
        r = Redis()
        if not useCache or (r.get(index) is None or pickle.loads(r.get(index)) == []):
            # Start the crawling process and then get the headlines from Redis
            spiderRunner.run_crawlProcess(NASDAQSpider, index, pages)
            return True
        return False

    def collectHeadlinesForIndex(self, index):

        # Call necessary functionality and start reactor
        spiderRunner = SpiderRunner(useCache=False)
        self._collectHeadlinesForIndex(index, spiderRunner, useCache=False, pages=50)
        spiderRunner.run()

        # Should have called a callback to stop the reactor
        return pickle.loads(Redis().get(index))

    def collectHeadlines(self):

        # Collect the headlines for every index
        companyInfo = self.getAllCompanyInfo(fromDB=False)
        spiderRunner = SpiderRunner()

        processExists = False
        for symbol in list(companyInfo.keys())[1700:2250]:

            # Web crawl for headlines with this index or company name
            if self._collectHeadlinesForIndex(index=symbol, spiderRunner=spiderRunner):
                processExists = True

        if processExists:
            spiderRunner.run()

        # Go through and get the headlines by the company
        return self.getHeadlinesFromRedis()

    def getHeadlinesFromRedis(self):

        headlinesByCompany = {}
        companyInfo = self.getAllCompanyInfo(fromDB=False)
        for symbol in companyInfo:
            # Might have to add try-catch on the pickle loading
            try:
                headlinesByCompany[symbol] = pickle.loads(Redis().get(symbol))

                # Convert the str to datetime
                for headline in headlinesByCompany[symbol]:
                    headline['date'] = datetime.strptime(headline['date'], '%Y-%m-%d')
            except:
                pass

        print(len(headlinesByCompany))
        return headlinesByCompany

    def getIndexInformationOnDate(self, index, date):

        # Make sure date is an instance of datetime
        if not isinstance(date, datetime):
            date = datetime.strptime(date, '%d %b %Y')

        # Get the historical data on this day
        d = get_historical_data(index, start=date, end=date, output_format='pandas')
        print(d)
        return d

    def getIndexRiseFallOnDate(self, index, date):

        # This accounts for weekend issue -- stocks arent open on weekends
        for i in range(3):

            # Try to create a date -- will error on non open days
            try:
                # Retrieve the data for this date
                data = self.getIndexInformationOnDate(index=index, date=date)

                # Return the close - the open normalized to +1 or -1
                if data.open.values[0] <= data.close.values[0]:

                    # Needs to return a tensor of class +1
                    return torch.tensor([[[1, 0]]]).type(torch.DoubleTensor)
                else:

                    # Needs to return a tensor of class -1
                    return torch.tensor([[[0, 1]]]).type(torch.DoubleTensor)
            except:

                # Add to the date
                date += timedelta(days=1)
                print('Error number: {i}'.format(i=i + 1))
        return torch.tensor([[[0.5, 0.5]]]).type(torch.DoubleTensor)

if __name__ == '__main__':
    sdc = StockDataCollector()
    print(sdc.getIndexRiseFallOnDate(index='MSFT', date='5 Nov 2018'))

    # Collect the headline for the index
    h = sdc.collectHeadlines()
    x = sdc.getHeadlinesFromRedis()