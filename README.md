Projet de Mini-IA : Pipeline CI/CD avec GitHub Actions et TensorFlow Keras Léger
Introduction
Ce dépôt contient la solution pour le mini-projet de Git & GitHub, centré sur la mise en place d'un pipeline d'Intégration Continue/Déploiement Continu (CI/CD) pour un modèle d'Intelligence Artificielle "factice" (léger). Le projet démontre l'utilisation de Git pour le contrôle de version, GitHub pour la collaboration et l'hébergement du dépôt, GitHub Actions pour l'automatisation du CI/CD, et Hugging Face Model Hub pour le déploiement du modèle.

Le modèle d'IA utilisé est un classifieur de sentiment très léger, implémenté avec TensorFlow et Keras, utilisant une couche TextVectorization et un réseau de neurones simple. Ce choix vise à démontrer le pipeline CI/CD avec un modèle rapide à entraîner et à déployer, optimisant les ressources.

Contexte du Devoir
L'objectif principal de ce devoir est de :

Mettre en place un dépôt Git/GitHub structuré.

Implémenter un pipeline CI/CD (GitHub Actions) pour entraîner, évaluer et déployer un modèle d'IA.

Déployer le modèle sur Hugging Face Model Hub.

Envoyer des notifications par e-mail sur le statut du pipeline.

Structure du Projet
votre_nom_matricule_GIT-GITHUB/
├── .github/                  # Configuration des workflows GitHub Actions
│   └── workflows/
│       └── ci-cd.yml         # Définition du pipeline CI/CD
├── scripts/                  # Scripts Python du projet
│   ├── model.py              # Fonctions d'entraînement et d'évaluation du modèle TensorFlow
│   └── deploy.py             # Script de déploiement vers Hugging Face Hub
├── models_dummy/             # Dossier où le modèle entraîné (factice) est sauvegardé localement (ignoré par Git)
│   ├── sentiment_dummy_model.h5
│   └── vocab.json
├── .env                      # Variables d'environnement (non versionné)
├── .gitignore                # Fichiers et dossiers à ignorer par Git
├── requirements.txt          # Dépendances Python du projet
└── README.md                 # Ce fichier


Technologies Utilisées
Git & GitHub : Système de contrôle de version et plateforme d'hébergement de dépôts.

GitHub Actions : Plateforme CI/CD pour l'automatisation du workflow.

TensorFlow / Keras : Framework de Machine Learning pour la création et l'entraînement du modèle.

scikit-learn : Pour le calcul des métriques d'évaluation (F1-score).

Hugging Face Model Hub : Plateforme de partage et de déploiement de modèles de ML.

Python-dotenv : Pour la gestion des variables d'environnement.

SMTP (via dawidd6/action-send-mail) : Pour les notifications par e-mail.

Mise en Place du Projet
1. Prérequis
Assurez-vous d'avoir :

Un compte GitHub.

Un compte Hugging Face.

Python 3.8+ installé.

pip et venv (ou conda) pour la gestion des environnements virtuels.

2. Initialisation Locale
Clonez le dépôt :

git clone https://github.com/votre_nom_utilisateur/votre_nom_matricule_GIT-GITHUB.git
cd votre_nom_matricule_GIT-GITHUB

Créez et activez l'environnement virtuel :

python -m venv venv
# Sur Windows: .\venv\Scripts\activate
# Sur macOS/Linux: source venv/bin/activate

Installez les dépendances :
Les dépendances sont listées dans le fichier requirements.txt.

pip install -r requirements.txt

Créez le dossier pour les scripts :

mkdir scripts

Créez le fichier .env :
À la racine de votre projet, créez un fichier nommé .env et ajoutez-y les variables suivantes (remplacez les placeholders) :

HF_API_KEY="hf_VOTRE_JETON_HUGGING_FACE"
THRESHOLD_SCORE="0.5" # Exemple de seuil F1-score pour le déploiement
SMTP_USER="votre.email@gmail.com"
SMTP_PASS="votre_mot_de_passe_application_gmail"

Note importante : Le fichier .env est ignoré par Git pour des raisons de sécurité.

3. Fichiers du Modèle et du Pipeline
scripts/model.py : Contient les fonctions train_and_save_model() et evaluate_model_and_save_results() pour le modèle TensorFlow/Keras léger, et predict_sentiment() pour les prédictions. Le modèle entraîné est sauvegardé dans models_dummy/sentiment_dummy_model.h5.

scripts/deploy.py : Gère la logique de connexion à Hugging Face Hub, vérifie le THRESHOLD_SCORE et pousse le modèle entraîné vers votre dépôt Hugging Face.

Assurez-vous de remplacer your_huggingface_username par votre vrai nom d'utilisateur Hugging Face dans ce fichier.

.github/workflows/ci-cd.yml : Le fichier de workflow GitHub Actions qui orchestre les étapes d'entraînement, d'évaluation et de déploiement. Il utilise les scripts Python mentionnés ci-dessus.

4. Stratégie de Branches
main : Branche principale, réservée aux versions stables et déployées.

dev : Branche de développement, où toutes les nouvelles fonctionnalités et modifications sont implémentées et testées avant d'être fusionnées dans main.

Exécution du Pipeline CI/CD
Le pipeline est configuré pour se déclencher automatiquement sur chaque git push vers la branche dev.

1. Préparation des Secrets GitHub
Pour que le pipeline puisse s'authentifier auprès de Hugging Face et envoyer des e-mails, vous devez configurer des secrets dans votre dépôt GitHub :

Allez sur votre dépôt GitHub (Settings > Secrets and variables > Actions).

Ajoutez les secrets suivants :

HF_API_KEY : Votre jeton d'accès Hugging Face (rôle write).

THRESHOLD_SCORE : Le F1-score minimal requis pour le déploiement (ex: 0.5 ou 0.6).

SMTP_USER : Votre adresse e-mail (ex: votre.email@gmail.com).

SMTP_PASS : Votre mot de passe d'application généré pour votre e-mail (pour Gmail, nécessite la validation en deux étapes).

2. Lancement du Workflow
Assurez-vous que tous les fichiers (scripts Python, ci-cd.yml, requirements.txt, .gitignore) sont à jour.

Ajoutez tous les changements à votre Git et committez-les sur la branche dev :

git add .
git commit -m "feat: Intégration du modèle TensorFlow léger et configuration CI/CD complète pour le TP"

Poussez vos modifications vers la branche dev sur GitHub :

git push origin dev

3. Suivi du Workflow
Rendez-vous sur l'onglet Actions de votre dépôt GitHub.

Cliquez sur le workflow le plus récent pour voir les logs détaillés de chaque étape (Checkout, Setup Python, Install Dependencies, Train Model, Evaluate Model, Deploy Model, Send Email Notification).

4. Vérification du Déploiement
Si le F1-score de votre modèle atteint ou dépasse le THRESHOLD_SCORE configuré, le modèle sera poussé vers votre dépôt sur Hugging Face Model Hub (https://huggingface.co/votre_nom_utilisateur/sentiment-tensorflow-keras-dummy).

5. Notification par Email
Vous devriez recevoir un e-mail à l'adresse configurée dans SMTP_USER vous informant du succès ou de l'échec de votre pipeline CI/CD.

Conclusion
Ce projet démontre une approche pratique de l'intégration des principes de Git, GitHub, et du CI/CD dans un workflow de développement de Machine Learning. L'utilisation d'un modèle léger permet de simuler le pipeline de manière efficace et rapide.
