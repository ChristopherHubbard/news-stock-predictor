import scrapy
from twisted.internet import reactor
from scrapy.utils.log import configure_logging
from scrapy.crawler import CrawlerRunner

from SpiderConstants import PAGES_PER_RUN

# Each website needs a separate spider
class ReutersSpider(scrapy.Spider):

    # Name of this spider
    name = 'Reuters'

    # Where the scraping requests begin
    def start_requests(self):

        # URLs to go through for the scraper -- Need to go through a whole bunch of pages to get all necessary data
        urls = ['https://www.reuters.com/news/archive/businessNews?view=page&page={pageNum}&pageSize=10'.format(pageNum=pageNum) for pageNum in range(PAGES_PER_RUN)]

        # Set up the requests
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    # Parse the reuters page
    def parse(self, response):
        pass

if __name__ == '__main__':
    # Scrapy needs to run inside twisted reactor -- Start the process
    configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
    runner = CrawlerRunner()

    d = runner.crawl(ReutersSpider)
    d.addBoth(lambda _: reactor.stop()) # Callback to stop the reactor
    reactor.run()  # the script will block here until the crawling is finished