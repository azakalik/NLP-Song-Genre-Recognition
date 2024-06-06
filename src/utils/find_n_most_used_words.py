import csv
from tokenize_and_format import tokenize_and_format


def find_n_most_used_words(n, input_csv_file, min_word_len: int or None) -> dict[str, int]:
    bag_of_words = {}
    with open(input_csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for line in reader:
            lyrics = line['Lyrics']
            tokenized = tokenize_and_format(lyrics, min_word_len)
            for word in tokenized:
                bag_of_words.setdefault(word, 0)
                bag_of_words[word] += 1

    sorted_bag_of_words = dict(sorted(bag_of_words.items(), key=lambda item: item[1], reverse=True))
    return dict(list(sorted_bag_of_words.items())[:n])
