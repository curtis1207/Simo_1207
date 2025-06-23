import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, TextVectorization
import numpy as np
import json

# Chemin où le modèle sera sauvegardé
# Assurez-vous que ce dossier est dans votre .gitignore
MODEL_DIR = "./models_dummy"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_dummy_model.h5")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json") # Pour sauvegarder le vocabulaire du TextVectorization

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
    # Chargement d'un très petit échantillon pour que le modèle soit vraiment "factice"
    (x_train_raw, y_train_raw), _ = tf.keras.datasets.imdb.load_data(num_words=1000)
    x_train_small = x_train_raw[:100] # Limite à 100 exemples
    y_train_small = y_train_raw[:100]

    # Convertir les index en texte pour TextVectorization
    word_index = tf.keras.datasets.imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text_indices):
        return " ".join([reverse_word_index.get(i - 3, "?") for i in text_indices])

    x_train_text = [decode_review(indices) for indices in x_train_small]
    y_train = np.array(y_train_small)

    print(f"Jeu de données factice préparé : {len(x_train_text)} exemples.")

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

    # Sauvegarder le vocabulaire (important pour la prédiction)
    vocabulary = vectorize_layer.get_vocabulary()
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=4)
    print(f"Vocabulaire sauvegardé dans : {VOCAB_PATH}")


    # 3. Création du modèle Keras très léger
    embedding_dim = 4 # Dimension d'embedding encore plus petite

    print("Construction du modèle TensorFlow/Keras factice...")
    model = Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
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
    return model # Retourne le modèle entraîné (facultatif)

def predict_sentiment(text: str):
    """
    Charge le modèle de sentiment factice sauvegardé et effectue une prédiction.
    """
    print("\n--- Début de la prédiction avec le modèle factice ---")

    if not os.path.exists(MODEL_PATH):
        print(f"Erreur: Le fichier du modèle '{MODEL_PATH}' n'existe pas. Veuillez d'abord entraîner et sauvegarder le modèle.")
        return {"error": "Modèle non trouvé. Veuillez exécuter train_and_save_model() d'abord."}

    if not os.path.exists(VOCAB_PATH):
        print(f"Erreur: Le fichier de vocabulaire '{VOCAB_PATH}' n'existe pas.")
        return {"error": "Vocabulaire non trouvé. Re-entraînez le modèle."}

    # Charger le vocabulaire pour la couche TextVectorization
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)

    # Recréer la couche TextVectorization avec le vocabulaire appris
    # C'est une étape cruciale pour s'assurer que le modèle charge correctement
    max_features = 1000 # Doit correspondre à la valeur utilisée lors de l'entraînement
    sequence_length = 50 # Doit correspondre à la valeur utilisée lors de l'entraînement
    
    # La couche TextVectorization doit être passée aux custom_objects si elle est incluse dans le modèle sauvegardé
    loaded_vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    loaded_vectorize_layer.set_weights([np.array(vocabulary), np.empty(shape=(0,))]) # Charger le vocabulaire

    # Charger le modèle, en spécifiant les objets personnalisés
    # Note: Si TextVectorization était sauvegardée *dans* le modèle, on la passe via custom_objects
    # Sinon, le modèle doit être reconstruit avec la même couche TextVectorization
    # Pour HDF5, custom_objects est souvent nécessaire.
    custom_objects = {"TextVectorization": loaded_vectorize_layer} # Ou juste TextVectorization si elle est définie globalement

    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"TextVectorization": TextVectorization})
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        # Si la TextVectorization a été sauvegardée comme partie du modèle,
        # le mieux est d'avoir une TextVectorization qui charge directement le vocabulaire interne.
        # Pour cet exemple simple, nous faisons une reconstruction plus directe si nécessaire.
        # En production, il est souvent préférable de sauvegarder le TextVectorization séparément ou d'utiliser SavedModel
        # qui gère mieux les couches personnalisées.
        print("Tentative de rechargement en ignorant les custom_objects pour simple Keras model...")
        model = tf.keras.models.load_model(MODEL_PATH)


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
    # Ceci est le bloc qui s'exécute lorsque vous lancez le script directement
    print("Exécution du script model.py en mode autonome.")
    
    # Étape 1: Entraîner et sauvegarder le modèle factice
    trained_model = train_and_save_model()
    
    # Étape 2: Effectuer des prédictions (après avoir sauvegardé et rechargé pour simuler le flux)
    if trained_model:
        print("\n--- Exemples de prédictions après entraînement/sauvegarde ---")
        predict_sentiment("C'était un film fantastique que j'ai adoré !", trained_model)
        predict_sentiment("Je n'ai vraiment pas aimé ce film, c'était horrible.", trained_model)
        predict_sentiment("C'est un film correct, rien de spécial.", trained_model)
    
    # Vous pouvez également tester en lançant le script deux fois:
    # 1. Pour entraîner et sauvegarder
    # 2. Pour commenter l'appel à train_and_save_model() et juste tester la prédiction
    # Cela simule un déploiement où le modèle est déjà là.