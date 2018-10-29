import scrapy
from datetime import datetime
from twisted.internet import reactor
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from scrapy.crawler import CrawlerRunner

from ReutersSpider import ReutersSpider
from CNBCSpider import CNBCSpider

class SpiderRunner():

    def __init__(self, spider_list):

        # Initialize the required resources
        # Scrapy needs to run inside twisted reactor -- Start the process
        configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
        self.settings = get_project_settings()
        self.crawlRunner = CrawlerRunner(settings=self.settings)

        # Add a crawl for each spider
        for spider in spider_list:
            self.crawlProcess = self.crawlRunner.crawl(spider)

        # Callback to stop the reactor on completion
        self.crawlProcess.addBoth(lambda _: reactor.stop())

    # Run the reactor
    def run(self):

        # The script will block here until the crawling is finished
        reactor.run()

    # Stop the run mid-stream
    def stop(self):

        reactor.stop()

if __name__ == '__main__':

    start = datetime.now()
    s = SpiderRunner([ReutersSpider, CNBCSpider])
    s.run()
    print('All spiders ran')
    print(datetime.now() - start)