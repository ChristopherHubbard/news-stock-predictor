import torch

# This class collects all event embeddings in a certain date range, combines same day event embeddings, and then outputs LT, MT, ST
class EventCollector():

    def __init__(self):

    def collect(self, eventsWithTimestamps):

        # Take the events with the timestamps and sort them by date -- assuming this is a list of events (not concat yet)
        eventsWithTimestamps.sort()

        # Now sorted by date -- go through
        for event in eventsWithTimestamps:



