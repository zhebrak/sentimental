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
        self.prev_score = self._get_sentiment(text).get(u'score', self.prev_score)
        return self.prev_score

    def sentiment_over_time(self, data, avg_window=3, output='sentimental.png'):
        """
        Draws a time series chart of text sentiment given:
            :param data: list of tuples (str, datetime object)
            :param avg_window: size of moving average window
            :param output: (optional) output file
        """
        text_list, time_list = zip(*data)

        score_list = [float(self.sentiment_score(text)) for text in text_list]
        avg = lambda l: sum(l) / len(l)
        avg_score_list = [
            avg(score_list[idx:idx + avg_window])
            for idx in range(len(score_list) - avg_window - 1)
        ]

        plt.plot(time_list, score_list)
        plt.savefig(output)


if __name__ == '__main__':
    data = []  # e.x. Comment.objects.values_list('text', 'created_at')

    sent = Sentimental()
    sent.sentiment_over_time(data)
