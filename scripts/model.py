import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, TextVectorization
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import json
import sys # Importé pour lire les arguments de ligne de commande
from sklearn.metrics import f1_score # Importé pour l'évaluation

# Chemin où le modèle sera sauvegardé
MODEL_DIR = "./models_dummy"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_dummy_model.h5")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json") # Pour sauvegarder le vocabulaire du TextVectorization (utile à titre informatif ou pour d'autres usages)

def train_and_save_model():
    """
    Crée, entraîne (très légèrement) et sauvegarde un modèle de classification de sentiment factice
    avec TensorFlow/Keras.
    """
    print("--- Début de l'entraînement et sauvegarde du modèle factice ---")

    # Création du répertoire de sauvegarde si non existant
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # 1. Préparation d'un petit jeu de données (IMDb pour la simplicité)
    (x_raw, y_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.imdb.load_data(num_words=1000)

    # Convertir les index en texte pour TextVectorization
    word_index = tf.keras.datasets.imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text_indices):
        return " ".join([reverse_word_index.get(i - 3, "?") for i in text_indices])

    # Décodez un petit échantillon pour l'entraînement et l'évaluation
    num_train_samples = 100
    num_eval_samples = 20 # Encore plus petit pour l'évaluation factice

    x_train_text = [decode_review(indices) for indices in x_raw[:num_train_samples]]
    y_train = np.array(y_raw[:num_train_samples])

    x_eval_text = [decode_review(indices) for indices in x_test_raw[:num_eval_samples]]
    y_eval = np.array(y_test_raw[:num_eval_samples])

    print(f"Jeu de données factice préparé : {len(x_train_text)} exemples d'entraînement, {len(x_eval_text)} exemples d'évaluation.")

    # 2. Configuration de la couche TextVectorization
    max_features = 1000 # Seulement 1000 mots pour ce modèle factice
    sequence_length = 50 # Séquence très courte

    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    # Adapter la couche au vocabulaire d'entraînement
    print("Apprentissage du vocabulaire du TextVectorization...")
    text_dataset = tf.data.Dataset.from_tensor_slices(x_train_text).batch(32)
    vectorize_layer.adapt(text_dataset)
    print("Vocabulaire appris.")

    # Sauvegarder le vocabulaire (peut être utile pour l'inspection ou d'autres usages, même si la couche est sauvegardée avec le modèle)
    vocabulary = vectorize_layer.get_vocabulary()
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=4)
    print(f"Vocabulaire sauvegardé dans : {VOCAB_PATH}")

    # 3. Création du modèle Keras très léger
    embedding_dim = 4 # Dimension d'embedding encore plus petite

    print("Construction du modèle TensorFlow/Keras factice...")
    model = Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string), # Input est une chaîne de caractères
        vectorize_layer, # La couche de vectorisation est intégrée au modèle
        Embedding(max_features + 1, embedding_dim),
        GlobalAveragePooling1D(),
        Dense(2, activation='relu'), # Deux neurones dans la couche cachée
        Dense(1, activation='sigmoid') # Classification binaire
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # 4. Entraînement très court du modèle
    train_dataset_tf = tf.data.Dataset.from_tensor_slices((x_train_text, y_train)).batch(32)

    print("Début de l'entraînement factice (1 époque)...")
    model.fit(train_dataset_tf, epochs=1, verbose=0) # Très peu d'époques, pas de logs détaillés
    print("Entraînement factice terminé.")

    # 5. Sauvegarde du modèle
    model.save(MODEL_PATH, save_format='h5') # Sauvegarde au format HDF5
    print(f"Modèle factice sauvegardé dans : {MODEL_PATH}")
    print("--- Fin de l'entraînement et sauvegarde du modèle factice ---")

    # Retourne le modèle entraîné et les données d'évaluation pour usage direct si besoin
    return model, x_eval_text, y_eval

def evaluate_model_and_save_results(model=None, x_eval_text=None, y_eval=None):
    """
    Charge le modèle, évalue sa performance et sauvegarde les résultats.
    Peut recevoir un modèle déjà chargé et des données si appelée après entraînement.
    """
    print("\n--- Début de l'évaluation du modèle ---")

    # Définir custom_objects ici car TextVectorization est une couche personnalisée
    # Cela aide Keras à reconstruire correctement la couche lors du chargement.
    custom_objects = {"TextVectorization": TextVectorization}

    if model is None:
        # Tenter de charger le modèle si non fourni
        if not os.path.exists(MODEL_PATH):
            print(f"Erreur: Le fichier du modèle '{MODEL_PATH}' n'existe pas. Veuillez d'abord entraîner et sauvegarder le modèle.")
            return {"error": "Modèle non trouvé pour évaluation."}
        
        # Le fichier de vocabulaire n'est plus strictement nécessaire pour le chargement
        # du modèle complet si TextVectorization est la première couche et sauvegardée avec le modèle.
        # Donc, nous n'appelons PLUS .set_weights() manuellement ici.
        
        try:
            # Charger le modèle. Keras va gérer le chargement de TextVectorization et son vocabulaire interne.
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle pour évaluation: {e}")
            return {"error": f"Impossible de charger le modèle pour évaluation: {e}"}

    # Préparer le jeu de données d'évaluation si non fourni
    if x_eval_text is None or y_eval is None:
        (x_raw, y_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.imdb.load_data(num_words=1000)
        word_index = tf.keras.datasets.imdb.get_word_index()
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        def decode_review(text_indices):
            return " ".join([reverse_word_index.get(i - 3, "?") for i in text_indices])
        num_eval_samples = 20 # Doit correspondre à la taille utilisée dans train_and_save_model
        x_eval_text = [decode_review(indices) for indices in x_test_raw[:num_eval_samples]]
        y_eval = np.array(y_test_raw[:num_eval_samples])

    eval_dataset_tf = tf.data.Dataset.from_tensor_slices(x_eval_text).batch(32)

    # Effectuer l'évaluation
    loss, accuracy = model.evaluate(eval_dataset_tf, verbose=0)
    print(f"Résultats d'évaluation - Perte: {loss:.4f}, Précision: {accuracy:.4f}")

    # Calcul du F1-score
    y_pred_probs = model.predict(eval_dataset_tf, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    f1 = f1_score(y_eval, y_pred, average="binary")
    print(f"F1-score: {f1:.4f}")

    # Sauvegarde des résultats d'évaluation dans un fichier JSON
    evaluation_results_file = "evaluation_results.json"
    with open(evaluation_results_file, "w") as f:
        json.dump({"accuracy": accuracy, "f1_score": f1}, f)
    print(f"Résultats de l'évaluation sauvegardés dans {evaluation_results_file}")
    print("--- Fin de l'évaluation du modèle ---")
    return {"accuracy": accuracy, "f1_score": f1}


def predict_sentiment(text: str):
    """
    Charge le modèle de sentiment factice sauvegardé et effectue une prédiction.
    """
    print("\n--- Début de la prédiction avec le modèle factice ---")

    # Définir custom_objects ici car TextVectorization est une couche personnalisée
    custom_objects = {"TextVectorization": TextVectorization}

    if not os.path.exists(MODEL_PATH):
        print(f"Erreur: Le fichier du modèle '{MODEL_PATH}' n'existe pas. Veuillez d'abord entraîner et sauvegarder le modèle.")
        return {"error": "Modèle non trouvé. Veuillez exécuter train_and_save_model() d'abord."}

    # Le fichier de vocabulaire n'est plus strictement nécessaire pour le chargement
    # du modèle complet si TextVectorization est la première couche et sauvegardée avec le modèle.
    # Donc, nous n'appelons PLUS .set_weights() manuellement ici.
    
    try:
        # Charger le modèle. Keras va gérer le chargement de TextVectorization et son vocabulaire interne.
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return {"error": f"Impossible de charger le modèle pour prédiction: {e}"}


    # Préparer l'entrée pour la prédiction
    input_data = tf.constant([text])
    
    # Effectuer la prédiction
    prediction_proba = model.predict(input_data)[0][0] # Obtenir la probabilité scalaire

    # Convertir la probabilité en étiquette de sentiment
    sentiment = "positif" if prediction_proba > 0.5 else "négatif"
    
    print(f"Texte d'entrée: '{text}'")
    print(f"Probabilité de prédiction (positif): {prediction_proba:.4f}")
    print(f"Sentiment prédit: {sentiment}")
    print("--- Fin de la prédiction ---")
    
    return {"text": text, "sentiment": sentiment, "probability": float(prediction_proba)}


if __name__ == "__main__":
    # Ce bloc est exécuté si le script est lancé directement depuis la ligne de commande.
    # Nous l'adaptons pour permettre différents modes d'exécution (train, evaluate, predict)
    # via les arguments de ligne de commande.

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "train":
            train_and_save_model()
        elif command == "evaluate":
            evaluate_model_and_save_results()
        elif command == "predict":
            if len(sys.argv) > 2:
                text_to_predict = sys.argv[2]
                predict_sentiment(text_to_predict)
            else:
                print("Usage: python model.py predict 'Your text here'")
        else:
            print("Commande inconnue. Utilisez 'train', 'evaluate' ou 'predict'.")
    else:
        # Comportement par défaut si aucun argument n'est donné (comme avant)
        print("Exécution du script model.py en mode autonome (par défaut: entraînement et exemples de prédiction).")
        # Étape 1: Entraîner et sauvegarder le modèle factice
        trained_model, x_eval_text, y_eval = train_and_save_model()
        
        # Étape 2: Effectuer des prédictions (après avoir sauvegardé et rechargé pour simuler le flux)
        if trained_model:
            print("\n--- Exemples de prédictions après entraînement/sauvegarde ---")
            predict_sentiment("C'était un film fantastique que j'ai adoré !")
            predict_sentiment("Je n'ai vraiment pas aimé ce film, c'était horrible.")
            predict_sentiment("C'est un film correct, rien de spécial.")
            
            # Et une évaluation pour vérifier la nouvelle fonction
            evaluate_model_and_save_results(model=trained_model, x_eval_text=x_eval_text, y_eval=y_eval)
