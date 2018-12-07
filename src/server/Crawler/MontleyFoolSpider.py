import scrapy
from scrapy_splash import SplashRequest
from twisted.internet import reactor
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings, Settings
from scrapy.crawler import CrawlerRunner
from datetime import datetime

from SpiderConstants import PROXY_PATH, USER_PATH

# Spider to extract stories from CNBC News
class MontleyFoolSpider(scrapy.Spider):

    def __init__(self, symbol, pages=10):

        self.symbol = symbol
        self.name = symbol
        self.pages = pages
        self.baseUrl = 'https://www.fool.com'

    def set_properties(self, symbol):

        self.symbol = symbol

    # Where the scraping requests begin
    def start_requests(self):

        # Assume Url fail so exchange and name of security dont need to be known
        urls = ['{baseUrl}/quote/failed-lookup/{symbol}'.format(baseUrl=self.baseUrl, symbol=self.symbol.lower())]

        # Set up the requests
        for url in urls:
            yield SplashRequest(url=url, callback=self.parse)

    def parse(self, response):

        # Try extract on nasdaq -- else follow the next link to real exchange
        section = response.css('section.page-body')
        failedText = section.css('h2::text').extract_first().strip()
        wrongExchange = failedText == 'It looks like that lookup failed.'
        if wrongExchange:
            url = section.css('a::attr(href)').extract_first()
            yield SplashRequest(url='{base}{url}/content#article-{pages}'.format(base=self.baseUrl, url=url, pages=self.pages), callback=self.parse)
        # If an exception is thrown then this is the correct page
        else:
            # Extract the story titles and strp the headline -- Can make a more complex routine to extract for single news source?
            articleList = response.xpath('//div[@id="article-list"]')
            for storyNum in range(self.pages):

                # Strip the headline text and get the date
                story = articleList.xpath('//article[@id="article-{i}"]'.format(i=storyNum + 1))
                headline = story.css('a::text').extract_first().strip()
                dt = ' '.join(story.css('span.article-meta::text').extract()[1].strip().split(' ')[0:3])

                # Try to create the date from the passed in time
                try:
                    date = datetime.strptime(dt, '%b %d %Y').date()
                except:
                    date = datetime.now().date()

                # Return the parsed headline and time
                yield {
                    'headline': headline,
                    'date': date
                }

if __name__ == '__main__':
    # Scrapy needs to run inside twisted reactor -- Start the process
    configure_logging({'LOG_FORMAT': '%(levelname)s: %(message)s'})
    runner = CrawlerRunner(settings=Settings({
            'DOWNLOAD_DELAY': 3,
            'CONCURRENT_REQUESTS': 20,
            'ROBOTSTXT_OBEY': False,
            'USER_AGENT': 'Mozilla/5.0 (X11; Linux x86_64; rv:48.0) Gecko/20100101 Firefox/48.0',
            'AUTOTHROTTLE_ENABLED': True,
            'HTTPCACHE_ENABLED': False, # Cache enabled for testing
            'HTTPCACHE_EXPIRATION_SECS': 0,
            'TELNETCONSOLE_PORT': None,
            'RETRY_ENABLED': False,
            'REDIRECT_ENABLED': True,
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
                'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 610,
                'random_useragent.RandomUserAgentMiddleware': 400,
                'rotating_proxies.middlewares.RotatingProxyMiddleware': 110,
                'rotating_proxies.middlewares.BanDetectionMiddleware': 620,
            },
            'PROXY_LIST': PROXY_PATH,
            'PROXY_MODE': 0,
            'USER_AGENT_LIST': USER_PATH
        }))

    d = runner.crawl(MontleyFoolSpider, symbol='MSFT', pages=50)
    d.addBoth(lambda _: reactor.stop())  # Callback to stop the reactor
    reactor.run()  # the script will block here until the crawling is finished