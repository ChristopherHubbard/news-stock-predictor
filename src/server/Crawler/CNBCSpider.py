import scrapy
from twisted.internet import reactor
from scrapy.utils.log import configure_logging
from scrapy.utils.project import get_project_settings
from scrapy.crawler import CrawlerRunner
from datetime import datetime

from SpiderConstants import PAGES_CNBC

# Spider to extract stories from CNBC News
class CNBCSpider(scrapy.Spider):

    # Name of this spider
    name = 'cnbc'

    # Where the scraping requests begin
    def start_requests(self):
        # URLs to go through for the scraper -- Need to go through a whole bunch of pages to get all necessary data
        urls = ['https://www.cnbc.com/id/19916736/page/{pageNum}'.format(pageNum=(pageNum + 1)) for pageNum in range(PAGES_CNBC)]

        # Set up the requests
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):

        # Extract the story titles and strp the headline -- Can make a more complex routine to extract for single news source?
        for story in response.css('div.asset'):

            # Strip the headline text and get the date
            headline = story.css('div.headline').css('a::text').extract_first().strip()
            dt = story.css('time::text').extract_first().strip().split(',')[1].strip()

            # Try to create the date from the passed in time
            try:
                date = datetime.strptime(dt, '%d %b %Y').date()
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
    settings = get_project_settings()
    runner = CrawlerRunner(settings=settings)

    d = runner.crawl(CNBCSpider)
    d.addBoth(lambda _: reactor.stop())  # Callback to stop the reactor
    reactor.run()  # the script will block here until the crawling is finished