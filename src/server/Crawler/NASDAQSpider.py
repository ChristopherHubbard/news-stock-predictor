import scrapy
from twisted.internet import reactor
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings, Settings
from scrapy.crawler import CrawlerRunner
from datetime import datetime

from src.server.Crawler.SpiderConstants import PAGES_NASDAQ

# Spider to extract stories from CNBC News
class NASDAQSpider(scrapy.Spider):

    def __init__(self, symbol):

        self.set_symbol(symbol=symbol)

    def set_symbol(self, symbol):

        self.symbol = symbol
        self.name = symbol

    # Where the scraping requests begin
    def start_requests(self):
        # URLs to go through for the scraper -- Need to go through a whole bunch of pages to get all necessary data
        urls = ['https://www.nasdaq.com/symbol/{symbol}/news-headlines?page={pageNum}'.format(pageNum=(pageNum + 1), symbol=self.symbol) for pageNum in range(PAGES_NASDAQ)]

        # Set up the requests
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):

        # Extract the story titles and strp the headline -- Can make a more complex routine to extract for single news source?
        headlines = response.css('div.news-headlines')
        for story in headlines.xpath('.//div[not(@*)]'):

            # Strip the headline text and get the date
            headline = story.css('a::text').extract_first().strip()
            dt = story.css('small::text').extract_first().strip().split(' ')[0].strip()

            # Try to create the date from the passed in time
            try:
                date = datetime.strptime(dt, '%m/%d/%Y').date()
            except:
                date = datetime.now().date()

            # Return the parsed headline and time
            yield {
                'headline': headline,
                'date': date
            }

    @staticmethod
    def runSpider(symbol):

        # Scrapy needs to run inside twisted reactor -- Start the process
        configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
        settings = Settings({
            'ITEM_PIPELINES': {
                'src.server.Crawler.JSONPipeline.JSONPipeline': 100,
                'src.server.Crawler.RedisPipeline.RedisPipeline': 200
            },
            'DOWNLOAD_DELAY': 2,
            'ROBOTSTXT_OBEY': True,
            'USER_AGENT': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
            'AUTOTHROTTLE_ENABLED': True,
            'HTTPCACHE_ENABLED': False
        })

        runner = CrawlerRunner(settings=settings)

        d = runner.crawl(NASDAQSpider, symbol=symbol)
        d.addBoth(lambda _: reactor.stop())  # Callback to stop the reactor
        reactor.run()  # the script will block here until the crawling is finished

if __name__ == '__main__':
    # Scrapy needs to run inside twisted reactor -- Start the process
    configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
    settings = get_project_settings()
    runner = CrawlerRunner(settings=settings)

    d = runner.crawl(NASDAQSpider, symbol='AAPL')
    d.addBoth(lambda _: reactor.stop())  # Callback to stop the reactor
    reactor.run()  # the script will block here until the crawling is finished