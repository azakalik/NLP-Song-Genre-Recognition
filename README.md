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