import json
from bson import json_util

from SpiderConstants import JSON_DATA_FILE_NAME

class JSONPipeline():

    def open_spider(self, spider):

        # Open the file for writing
        self.file = open(spider.name + JSON_DATA_FILE_NAME, 'w')
        self.headlines = []

    def close_spider(self, spider):

        # Insert into the JSON file
        obj = json.dumps({
            'Count': len(self.headlines),
            'Headlines': sorted(self.headlines, key=lambda x: x['date'])
        }, indent=4, sort_keys=True, default=str)
        self.file.write(obj)

        # Close the file when the spider ends
        self.file.close()

    def process_item(self, item, spider):

        # Dump the item into the json file
        line = json.loads(json.dumps(item, indent=4, sort_keys=True, default=str))
        self.headlines.append(line)

        # Return the item to the next pipeline
        return item