import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import nltk
from sklearn.metrics import confusion_matrix
from lyrics_dataset import LyricsDataset
from utils.lstm_functions import format_training_data, plot_accuracies, plot_confusion_matrices, collate_fn
from lstm import LSTM

nltk.download('punkt')
nltk.download('stopwords')


def main():
    corpus = pd.read_csv('./csv/train_reduced.csv').dropna()
    train_data = format_training_data(corpus)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    lstm_accuracies = []
    lstm_confusion_matrices = []

    lyrics_types = ['Lyrics', 'Lyrics_Without_Special_Chars', 'Lyrics_Without_Stopwords', 'Limited_Lyrics']

    # Data has been preprocessed to obtain perfect balancing
    songs_per_genre = 1890

    for lyrics_type in lyrics_types:
        print('----------', lyrics_type, '----------')
        lyrics = list(train_data.groupby(['Genre']).head(songs_per_genre).reset_index()[lyrics_type])
        genres = list(train_data.groupby(['Genre']).head(songs_per_genre).reset_index()['Genre'])

        dataset = LyricsDataset(lyrics, genres)

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset)-int(0.8 * len(dataset))])

        # Initialize the LSTM model with hyperparameters
        model = LSTM(vocab_size=len(dataset.vocab) + 1, embedding_dim=200, hidden_dim=512,
                     output_dim=len(dataset.genre_idx), num_epochs=50, batch_size=32, learning_rate=0.0001)
        model = model.to(device)

        # Create the dataloaders with custom collate function
        train_dataloader = DataLoader(train_dataset, batch_size=model.batch_size, shuffle=True, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=model.batch_size, shuffle=False, collate_fn=collate_fn)

        # Define the loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)

        accuracies = []

        for epoch in range(model.num_epochs):
            # Training
            model.train()
            for lyrics_indices, genres, lengths in train_dataloader:
                lyrics_indices = lyrics_indices.to(device)
                genres = genres.to(device)
                lengths = lengths.cpu()
                optimizer.zero_grad()
                logits = model(lyrics_indices, lengths)
                loss = loss_fn(logits, genres)
                loss.backward()
                optimizer.step()

            # Testing
            model.eval()
            with torch.no_grad():
                total_correct = 0
                total_samples = 0
                true_labels = []
                predicted_labels = []

                for lyrics_indices, genres, lengths in test_dataloader:
                    lyrics_indices = lyrics_indices.to(device)
                    genres = genres.to(device)
                    lengths = lengths.cpu()

                    logits = model(lyrics_indices, lengths)
                    _, predictions = torch.max(logits, 1)

                    total_correct += (predictions == genres).sum().item()
                    total_samples += genres.size(0)
                    true_labels.extend(genres.cpu().numpy())
                    predicted_labels.extend(predictions.cpu().numpy())

                accuracy = total_correct / total_samples
                accuracies.append(accuracy)
                print(f'Epoch {epoch + 1}: Accuracy = {accuracy:.4f}')

        lstm_accuracies.append(accuracies)
        lstm_confusion_matrices.append(confusion_matrix(true_labels, predicted_labels))

    plot_accuracies(lstm_accuracies)
    plot_confusion_matrices(lstm_confusion_matrices)


if __name__ == '__main__':
    main()
