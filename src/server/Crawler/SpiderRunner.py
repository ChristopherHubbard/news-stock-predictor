import scrapy
from datetime import datetime
from twisted.internet import reactor
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings, Settings
from scrapy.crawler import CrawlerRunner, CrawlerProcess

class SpiderRunner():

    def __init__(self):

        # Initialize the required resources
        # Scrapy needs to run inside twisted reactor -- Start the process
        configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
        self.settings = Settings({
            'ITEM_PIPELINES': {
                # 'src.server.Crawler.JSONPipeline.JSONPipeline': 100,
                'src.server.Crawler.RedisPipeline.RedisPipeline': 200
            },
            'DOWNLOAD_DELAY': 2,
            'ROBOTSTXT_OBEY': True,
            'USER_AGENT': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
            'AUTOTHROTTLE_ENABLED': True,
            'HTTPCACHE_ENABLED': False,
            'TELNETCONSOLE_PORT': None
        })

    def run_crawlProcess(self, spider, index):

        self.crawlRunner = CrawlerRunner(self.settings)
        return self.crawlRunner.crawl(spider, index)

    # Run the reactor
    def run(self):

        # The script will block here until the crawling is finished
        reactor.run()

    def callNext(self, null, spider, index):

        reactor.callLater(0, self.run_crawlProcess, spider=spider, index=index)

    # Stop the run mid-stream
    def stop(self):

        reactor.stop()

if __name__ == '__main__':

    start = datetime.now()
    s = SpiderRunner([])
    s.run()
    print('All spiders ran')
    print(datetime.now() - start)