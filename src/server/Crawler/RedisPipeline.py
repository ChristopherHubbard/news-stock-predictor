import json
import pickle
from redis import Redis
from functools import reduce

class RedisPipeline():

    def open_spider(self, spider):

        # Create Redis connection
        self.headlines = []
        self.r = Redis()

    def close_spider(self, spider):
        pass
        #Insert into the Redis store
        #self.headlines = reduce(lambda li, el: li.append(el) or li if el not in li else li, self.headlines, [])
        #self.r.set(spider.symbol, pickle.dumps(sorted(self.headlines, key=lambda x: x['date'])))
        # print(self.r.get(spider.symbol))

    def process_item(self, item, spider):

        # Dump the item into the json file
        line = json.loads(json.dumps(item, indent=4, sort_keys=True, default=str))
        self.headlines.append(line)

        # Return the item to the next pipeline
        return item