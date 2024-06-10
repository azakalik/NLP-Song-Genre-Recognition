from collections import Counter
import re

import numpy as np
import torch
from matplotlib import pyplot as plt
from nltk import word_tokenize
from sklearn.metrics import ConfusionMatrixDisplay
from torch.nn.utils.rnn import pad_sequence
from nltk.corpus import stopwords


def remove_special_chars(raw_text):
    mapping_table = str.maketrans({'\n': ' ', '\t': ' ', '\x85': ' ', '\xa0': ' ', '\u2028': ' ', '\u3000': ' '})
    text = re.sub(r'[^a-zA-Z\s]', '', raw_text)
    return text.translate(mapping_table)


def format_training_data(data):
    min_words = 100
    stop_words = set(stopwords.words('english'))

    data = data[data['Language'] == 'en'].reset_index()
    data['Lyrics_Without_Special_Chars'] = data['Lyrics'].apply(remove_special_chars)
    data['lengths'] = data['Lyrics_Without_Special_Chars'].str.split(' ').str.len()
    data = data[data['lengths'] >= min_words]
    data['Limited_Lyrics'] = data['Lyrics_Without_Special_Chars'].str.split(' ').apply(lambda x: x[:min_words]).apply(
        lambda x: ' '.join(x))
    data['Lyrics_Without_Stopwords'] = data['Lyrics_Without_Special_Chars'].apply(
        lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
    data['words_without_stopwords'] = data['Lyrics_Without_Stopwords'].apply(lambda x: x.split())
    data['word_freq_without_stopwords'] = data['words_without_stopwords'].apply(
        lambda x: dict(Counter(x).most_common(50)))
    return data


def collate_fn(data):
    lyrics, genres = zip(*data)
    lyrics_indices = [torch.tensor(seq) for seq in lyrics]
    genres = torch.tensor(genres)
    lyrics_padded = pad_sequence(lyrics_indices, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in lyrics_indices])
    return lyrics_padded, genres, lengths


def plot_accuracies(accuracies):
    colors = ['blue', 'cyan', 'purple', 'red']
    labels = ['lstm_lyrics', 'lstm_lyrics_without_special_chars', 'lstm_lyrics_without_stopwords',
              'lstm_limited_lyrics']
    for i in range(len(accuracies)):
        plt.plot(accuracies[i], color=colors[i], label=labels[i])

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('LSTM Accuracies for all genres')
    plt.legend()
    plt.show()


def plot_confusion_matrices(confusion_matrices):
    for i in range(len(confusion_matrices)):
        disp = ConfusionMatrixDisplay(confusion_matrices[i])
        disp.plot()
