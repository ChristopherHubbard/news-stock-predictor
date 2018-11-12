from iexfinance import get_historical_data, get_available_symbols
from datetime import datetime, timedelta
from redis import Redis
import torch
import scrapy
import json
import pprint
import pickle
from twisted.internet import reactor
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from scrapy.crawler import CrawlerRunner

# Probably will just make a crawling microservice eventually
from src.server.Crawler.NASDAQSpider import NASDAQSpider
from src.server.Crawler.SpiderRunner import SpiderRunner

# This class collects a months worth of headlines for an index
class StockDataCollector():

    def getAllCompanyInfo(self, fromDB=True):

        # Get the company information either from the DB or from IEX -- Should only need to IEX once
        companyInformation = {}
        if fromDB:
            pass
        else:
            # Get all the indices that the IEX finance API can server
            companyData = get_available_symbols()

            # Set up the company's information object
            for company in companyData:

                # Only include company values -- not dividends, crypto, ect
                if company['type'] == 'cs':
                    companyInformation[company['symbol']] = company['name']

        return companyInformation

    def storeAllCompanyInfo(self):

        # Store all the company information from the company dictionary in Dynamo
        companyInfo = self.getAllCompanyInfo(fromDB=False)

    def _collectHeadlinesForIndex(self, index, spiderRunner):

        # Collect the headlines for the given index -- make sure has timestamp -- Use NASDAQ website?
        r = Redis()
        if r.get(index) is None or pickle.loads(r.get(index)) == []:
            # Start the crawling process and then get the headlines from Redis
            spiderRunner.run_crawlProcess(NASDAQSpider, index)
            return True
        return False

    def collectHeadlinesForIndex(self, index):

        # Call necessary functionality and start reactor
        spiderRunner = SpiderRunner()
        self._collectHeadlinesForIndex(index, spiderRunner)
        spiderRunner.run()

        # Should have called a callback to stop the reactor
        return pickle.loads(Redis().get(index))

    def collectHeadlines(self):

        # Collect the headlines for every index
        companyInfo = self.getAllCompanyInfo(fromDB=False)
        spiderRunner = SpiderRunner()

        processExists = False
        for symbol in list(companyInfo.keys())[0:10]:

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
                break

        pprint.pprint(headlinesByCompany)
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
        for i in range(8):

            # Try to create a date -- will error on non open days
            try:
                # Retrieve the data for this date
                data = self.getIndexInformationOnDate(index=index, date=date)

                # Return the close - the open normalized to +1 or -1
                if data.open.values[0] <= data.close.values[0]:

                    # Needs to return a tensor of class +1
                    return torch.tensor([[[1, 0]]])
                else:

                    # Needs to return a tensor of class -1
                    return torch.tensor([[[0, 1]]])
            except:

                # Add to the date
                date += timedelta(days=1)
                print('Error number: {i}'.format(i=i + 1))

if __name__ == '__main__':
    sdc = StockDataCollector()
    sdc.getIndexRiseFallOnDate(index='MSFT', date='5 Nov 2018')

    # Collect the headline for the index
    h = sdc.collectHeadlines()