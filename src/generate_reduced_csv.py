import csv
import os


def generate_reduced_csv(input_path, output_path):
    lines_by_genre = {}
    # open csv file as dict
    with open(input_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for line in reader:
            language = line['Language']
            if language == 'en':
                genre = line['Genre']
                lines_by_genre.setdefault(genre, [])
                if len(lines_by_genre[genre]) < 1890:
                    lines_by_genre[genre].append(line)

    reduced_csv_lines = []
    for lines in lines_by_genre.values():
        for line in lines:
            reduced_csv_lines.append(line)

    if not os.path.exists('csv'):
        os.makedirs('csv')

    with open(output_path, mode='w', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(reduced_csv_lines)


if __name__ == '__main__':
    generate_reduced_csv('original_csv/train.csv', 'csv/train_reduced.csv')
