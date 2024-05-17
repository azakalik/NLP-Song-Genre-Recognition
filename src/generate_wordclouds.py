import os
import csv
from .utils import tokenize_and_format
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from .utils.find_n_most_used_words import find_n_most_used_words
from typing import Set


def generate_wordclouds(input_csv, output_folder, min_word_len: int | None = None, stopwords: Set[str] | None = None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not stopwords:
        stopwords = set()

    bag_of_words_by_genre = {}
    bag_of_words = {}
    with open(input_csv, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for line in reader:
            genre = line['Genre']
            lyrics = line['Lyrics']
            tokenized = tokenize_and_format(lyrics, min_word_len)
            bag_of_words_by_genre.setdefault(genre, {})
            for word in tokenized:
                if word in stopwords:
                    continue
                bag_of_words_by_genre[genre].setdefault(word, 0)
                bag_of_words_by_genre[genre][word] += 1
                bag_of_words.setdefault(word, 0)
                bag_of_words[word] += 1

    for genre, bag_of_words in bag_of_words_by_genre.items():
        create_wordcloud(bag_of_words, genre, output_folder, stopwords)
    create_wordcloud(bag_of_words, 'All genres', output_folder, stopwords)
    pass


def create_wordcloud(bag_of_words, title: str, output_folder: str, stopwords: Set[str] | None):
    # Create a WordCloud object
    wordcloud = (WordCloud(width=800, height=800,
                           background_color='white',
                           stopwords=stopwords,
                           min_font_size=10)
                 .generate_from_frequencies(bag_of_words))

    # Plot the WordCloud
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    if len(stopwords) > 0:
        title += ' (filtered words)'

    plt.title(title)

    plt.savefig(output_folder + '/' + title + '.png')


if __name__ == '__main__':
    MIN_WORD_LEN = 4

    generate_wordclouds('csv/train_reduced.csv', 'outputs/word_clouds', min_word_len=MIN_WORD_LEN, stopwords=set())

    most_used_words = find_n_most_used_words(10, 'csv/train_reduced.csv', MIN_WORD_LEN)
    generate_wordclouds('csv/train_reduced.csv', 'outputs/word_clouds_filtered', min_word_len=MIN_WORD_LEN, stopwords=set(most_used_words.keys()))
