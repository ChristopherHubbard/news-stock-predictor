
class DatabasePipeline():

    # Initialize the pipeline using the passed in settings
    def __init__(self, config, dbType='dynamodb'):

        # Initialize the database config infomration
        self.databaseType = dbType
        self.config = config

        # Validate the config works for this type

    @classmethod
    def from_crawler(cls, crawler):

        # Return an instance of this class
        return cls(
            config=crawler.settings.get('DB_CONFIG'),
            dbType=crawler.settings.get('DB_TYPE')
        )

    def open_spider(self, spider):

        # Open the client connection
        if self.databaseType == 'dynamodb':

        elif self.databaseType == 'mongodb':

        elif self.databaseType == 'sql':

    def close_spider(self, spider):

        # Close the client connection with the database
        if self.databaseType == 'dynamodb':

        elif self.databaseType == 'mongodb':

        elif self.databaseType == 'sql':

    def process_item(self, item, spider):

        # Insert this item into the table
        self.db


