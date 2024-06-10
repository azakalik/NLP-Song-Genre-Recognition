import nltk
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tokenize_and_format import tokenize_and_format
from find_n_most_used_words import find_n_most_used_words
from sklearn.linear_model import LinearRegression

most_used_words = find_n_most_used_words(100, '../csv/train_reduced.csv', 3)

train = pd.read_csv('../csv/train_reduced.csv')
test = pd.read_csv('../csv/test.csv')

# Vectorizar los textos utilizando solo las palabras más usadas
vectorizer = TfidfVectorizer(vocabulary=most_used_words.keys(), stop_words='english')
print("Stopwords utilizadas:", vectorizer.get_stop_words())
X = vectorizer.fit_transform(train['Lyrics'])

# Convertir la matriz TF-IDF a un DataFrame de pandas
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print("Valores de TF-IDF:")
print(tfidf_df.head())

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(train['Genre'])

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_train_encoded, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluar el modelo
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

# R2 indica qué tan bien los datos se ajustan al modelo de regresión.
y_train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, y_train_pred)

y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)

print(f'R2 en el conjunto de entrenamiento: {train_r2}')
print(f'R2 en el conjunto de prueba: {test_r2}')
print(f'Puntuación de entrenamiento: {train_score}')
print(f'Puntuación de prueba: {test_score}')
plt.scatter(y_train, y_train_pred, color='blue', label='Entrenamiento')
plt.scatter(y_test, y_test_pred, color='red', label='Prueba')
plt.xlabel('Valores reales')
plt.ylabel('Predicciones')
plt.legend()
plt.show()

# esto se encarga de entrenar un modelo de regresion lineal y evaluar su rendimienrto de prediccion  de generos musicales
# a partir de las letras de las canciones representadas como vectores TF-IDF
