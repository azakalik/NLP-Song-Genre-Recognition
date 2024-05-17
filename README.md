# NLP-Song-Genre-Recognition
## Project Configuration
To manage the project dependencies, we use `pipenv`. To install the dependencies, run `pipenv install`.
If any problems arise, try running the following commands:
```bash
apt install pip
pip install pipenv
pipenv install
```
## Reduced Train Set
The original file, called `train.csv`, was very heavy and contained multiple languages.
For that reason, we opted to generate a subset of this csv, called `train_reduced.csv`, which contains only the English songs (exactly 1890 of each of the 10 genres).
The test and train files can be found in the `csv` folder.
To run the script to generate this reduced subset, create a folder `original_csv` that contains the file `train.csv` and run `pipenv generate_reduced_subset`.
NOTE: This repo does NOT include the original file `train.csv`, as it is too heavy. For this reason, don't attempt to run the script without the original file.

## WordClouds
To generate a series of wordclouds, run the script `pipenv run generate_wordclouds`.
This will create two folders: `wordclouds` and `wordclouds_reduced`, each containing 11 wordclouds.
From the 11 wordclouds, one of them applies to all music genres, while the other 10 are specific to each genre.
The idea behind `wordclouds_reduced` is not considering the 10 most used words in all genres for each wordcloud, so we can analyze more accurately the difference between the most used words in each genre.