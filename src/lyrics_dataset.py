import torch
from torch.utils.data import Dataset


class LyricsDataset(Dataset):
    def __init__(self, lyrics, genres):
        self.lyrics = lyrics
        self.genres = genres
        self.vocab = set([word for lyrics in self.lyrics for word in lyrics.split()])
        self.word_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.genre_idx = {genre: idx for idx, genre in enumerate(set(self.genres))}

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        lyrics = self.lyrics[idx]
        genre = self.genres[idx]
        lyrics_indices = [self.word_idx[word] for word in lyrics.split()]
        genre_index = self.genre_idx[genre]
        return torch.tensor(lyrics_indices), torch.tensor(genre_index)
