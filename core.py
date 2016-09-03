# coding: utf-8

from ConfigParser import ConfigParser
from datetime import datetime

import requests
import matplotlib.pyplot as plt

from watson_developer_cloud import AlchemyLanguageV1 as alchemy


class Sentimental():
    def __init__(self):
        self.alchemy_client = self._get_alchemy_client()
        self.prev_score = 0.5

    @staticmethod
    def _get_alchemy_client():
        alchemy_config = ConfigParser()
        alchemy_config.read(u'alchemy.conf')
        API_KEY = alchemy_config.get(u'credentials', 'API_KEY')

        return alchemy(api_key=API_KEY)

    def _get_sentiment(self, text):
        return self.alchemy_client.combined(
            text=text, extract=['doc-sentiment'], language='russian'
        ).get(u'docSentiment')

    def sentiment_score(self, text):
        score = self._get_sentiment(text).get(u'score')
        if score:
            self.prev_score = score
        else:
            return self.prev_score

        return score

    def sentiment_over_time(self, data, output='sentimental.png'):
        """
        Draws a time series chart of text sentiment given:
            :param data: list of tuples (str, datetime object)
            :param output: (optional) output file
        """
        text_list, time_list = zip(*data)
        plt.plot(time_list, [self.sentiment_score(text) for text in text_list])
        plt.savefig(output)

        # you can add mean for smoothing


if __name__ == '__main__':
    data = []  # e.x. Comment.objects.values_list('text', 'created_at')

    sent = Sentimental()
    sent.sentiment_over_time(data)
