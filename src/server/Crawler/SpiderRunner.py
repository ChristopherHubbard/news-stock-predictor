import scrapy
import scrapydo
import logging
from datetime import datetime
from twisted.internet import reactor
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings, Settings
from scrapy.crawler import CrawlerRunner, CrawlerProcess
from src.server.Crawler.SpiderConstants import PROXY_PATH, USER_PATH

# Setup scrapydo
scrapydo.setup()
scrapydo.default_settings.update({
    'LOG_LEVEL': 'DEBUG',
    'CLOSESPIDER_PAGECOUNT': 20
})
logging.root.setLevel(logging.INFO)

class SpiderRunner():

    def __init__(self, useCache=True):

        # Initialize the required resources
        # Scrapy needs to run inside twisted reactor -- Start the process
        configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
        self.settings = Settings({
            'ITEM_PIPELINES': {
                # 'src.server.Crawler.JSONPipeline.JSONPipeline': 100,
                'src.server.Crawler.RedisPipeline.RedisPipeline': 200
            },
            'DOWNLOAD_DELAY': 3,
            'CONCURRENT_REQUESTS': 10,
            'ROBOTSTXT_OBEY': True,
            'USER_AGENT': 'Mozilla/5.0 (X11; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0',
            'AUTOTHROTTLE_ENABLED': True,
            'HTTPCACHE_ENABLED': useCache, # Cache enabled for testing
            'HTTPCACHE_EXPIRATION_SECS': 0,
            'TELNETCONSOLE_PORT': None,
            'RETRY_ENABLED': False,
            'REDIRECT_ENABLED': False,
            'COOKIES_ENABLED': False,
            'REACTOR_THREADPOOL_MAXSIZE': 20,
            'DOWNLOAD_TIMEOUT': 10, # To avoid loss of entries?
            # Retry many times since proxies often fail
            'RETRY_TIMES': 10,
            # Retry on most error codes since proxies fail for different reasons
            'RETRY_HTTP_CODES': [500, 503, 504, 400, 403, 404, 408],
            'DOWNLOADER_MIDDLEWARES': {
                'scrapy.contrib.downloadermiddleware.useragent.UserAgentMiddleware': None,
                'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
                'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
                'random_useragent.RandomUserAgentMiddleware': 400,
                'rotating_proxies.middlewares.RotatingProxyMiddleware': 610,
                'rotating_proxies.middlewares.BanDetectionMiddleware': 620,
            },
            'PROXY_LIST': './proxy_list.txt',
            'PROXY_MODE': 0,
            'USER_AGENT_LIST': USER_PATH
        })
        self.crawlRunner = CrawlerRunner(self.settings)

    def run_scrapydoProcess(self, spider, index, pages=10):

        # Run with scrapydo -- setup spider args
        spider_args = {
            'symbol': index,
            'pages': pages,
            'capture_items': True,
            'timeout': 360,
            'settings': {
                'DOWNLOAD_DELAY': 3,
                'CONCURRENT_REQUESTS': 20,
                'ROBOTSTXT_OBEY': False,
                'USER_AGENT': 'Mozilla/5.0 (X11; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0',
                'AUTOTHROTTLE_ENABLED': True,
                'HTTPCACHE_ENABLED': False, # Cache enabled for testing
                'HTTPCACHE_EXPIRATION_SECS': 0,
                'TELNETCONSOLE_PORT': None,
                'RETRY_ENABLED': False,
                'REDIRECT_ENABLED': False,
                'COOKIES_ENABLED': False,
                'REACTOR_THREADPOOL_MAXSIZE': 20,
                'DOWNLOAD_TIMEOUT': 30, # To avoid loss of entries?
                # Retry many times since proxies often fail
                'RETRY_TIMES': 10,
                # Retry on most error codes since proxies fail for different reasons
                'RETRY_HTTP_CODES': [500, 503, 504, 400, 403, 404, 408],
                'DOWNLOADER_MIDDLEWARES': {
                    'scrapy.contrib.downloadermiddleware.useragent.UserAgentMiddleware': None,
                    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
                    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 110,
                    'random_useragent.RandomUserAgentMiddleware': 400,
                    'rotating_proxies.middlewares.RotatingProxyMiddleware': 610,
                    'rotating_proxies.middlewares.BanDetectionMiddleware': 620,
                },
                'PROXY_LIST': './proxy_list.txt',
                'PROXY_MODE': 0,
                'USER_AGENT_LIST': USER_PATH
            }
        }
        return scrapydo.run_spider(spider, **spider_args)

    def run_crawlProcess(self, spider, index, pages=10):

        return self.crawlRunner.crawl(spider, index, pages)

    # Run the reactor
    def run(self):

        # The script will block here until the crawling is finished
        d = self.crawlRunner.join()
        d.addBoth(lambda _: self.stop())
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