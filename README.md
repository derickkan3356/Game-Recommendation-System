# A Game Recommendation System

A data‑driven recommender that helps Steam players discover the next game they’ll love.  
It fuses **implicit interactions** (play‑time & achievements) with **rich metadata** from three public APIs ( SteamWeb, SteamSpy, RAWG) learns compact **wALS embeddings**, and ranks unseen games with a **RankNet‑optimized MLP**—achieving **0.85 NDCG@10** on held‑out users.

**Try the live demo:** <https://game-rec-app-team-27-87458525975.us-east1.run.app/>

**Check out the full project report:** <https://drive.google.com/file/d/1wrTZfcbvKhgkW-Cin8upVqxeLIFxYDRU/view?usp=sharing>

## Features
| Stage | Highlights |
|-------|------------|
| **Data ingest** | RAWG, Steam Spy & Steam Web APIs → 3M user–game rows |
| **Pre‑processing** | categorical encodings, missing imputation, Yeo‑Johnson scaling, feature engineering |
| **Collaborative filtering** | weighted ALS → 2‑D user & game embeddings |
| **Supervised ranker** | MLP (+ RankNet loss) + Optuna hyper‑search |
| **Evaluation** | user‑stratified CV, NDCG@K evaluation |
| **Web demo** | Streamlit front‑end |

## Requirements

- Python 3.12.0
- `pip install -r requirements.txt`

## Folder Structure:
For assets, please navigate to this [google drive](https://drive.google.com/drive/folders/1luwRhqwsyz1yvLcEh8MCm3o0WamaMe4g?usp=sharing)
```
📂 data_collection
- 📂 RAWG
  - 📜 data_collection_rawg.ipynb       # -> games_data.json
  - 📜 add_steam_app_id_to_RAWG.ipynb   # games_data.json -> games_data_with_id.pkl
  - 📜 data_cleaning_RAWG.ipynb         # games_data_with_id.pkl -> RAWG_clean.pkl
- 📂 SteamSpy
  - 📜 SteamSpyAPI.ipynb                # -> (SteamSpy_raw.json, SteamSpy_clean.pkl)
- 📂 SteamWeb
  - 📜 crawl_steam_members.py           # -> all_data_users.json
  - 📜 SteamWebAPI.ipynb                # (all_data_users.json, steam_countries.json) -> (steamWeb_raw.csv, steamWeb_processed.pkl)
- 📜 combine_data.ipynb                 # (RAWG_clean.pkl, SteamSpy_clean.pkl, steamWeb_processed.pkl) -> (train_raw.pkl, test_raw.pkl)

📂 data_preprocessing
- 📜 feature_engineering.ipynb          # (train_raw.pkl, test_raw.pkl) -> (train_feature_engineered.pkl, test_feature_engineered.pkl)
- 📜 Missing_value_imputation.ipynb     # (train_feature_engineered.pkl, test_feature_engineered.pkl) -> (train_impute.pkl, test_impute.pkl)
- 📜 scaling_numerical_features.ipynb   # (train_impute.pkl, test_impute.pkl) -> (train_scaled.pkl, test_scaled.pkl)
- 📜 collaborative_filtering.ipynb      # (train_scaled.pkl, test_scaled.pkl) -> (train_ready.pkl, test_ready.pkl, trained_ALS.pkl)
- 📄 trained_ALS.pkl                    # Trained ALS

📂 modeling
- 📜 neural_network.ipynb               # (train_ready.pkl, test_ready.pkl) -> (PyTorch_trained_model.pth, PyTorch_model_weights.pth, PyTorch_model_hyperparameters.json)
- 📜 XGBoost.ipynb                      # (train_raw.pkl, test_raw.pkl) -> fitted_transformers.pkl
- 📄 PyTorch_trained_model.pth          # Trained neural network full model
- 📄 PyTorch_model_weights.pth          # Trained neural network weight only
- 📄 PyTorch_model_hyperparameters.json # Trained neural network hyperparameter only
- 📄 fitted_transformers.pkl            # Trained XGBoost
- 📄 user_country_encoder.pkl           # Saved label encode mapper for country code

📂 visualization
- 📜 Visualization.ipynb

📂 deployment_pipeline
- 📜 prediction_pipeline.py             # Pipeline for predicting games for a user
- 📜 app.py                             # Front-end UI
- 📜 Dockerfile
- 📜 preprocess_data.py

📄 requirements.txt                     # Python library version

📂 assets (stored in Google drive)
- 📂 RAWG
  - 📄 games_data.json                  # Raw data collected from RAWG
  - 📄 games_data_with_id.pkl           # Added Steam app ID
  - 📄 RAWG_clean.pkl                   # Basic cleaning
- 📂 SteamSpy
  - 📄 SteamSpy_raw.json                # Raw data collected from SteamSpy
  - 📄 SteamSpy_clean.pkl               # Basic cleaning
- 📂 SteamWeb
  - 📄 all_data_users.json              # Raw data collected from SteamWeb
  - 📄 steam_countries.json             # To map user location to coordinates
  - 📄 steamWeb_raw.csv                 # all_data_users.json with more user data
  - 📄 steamWeb_processed.pkl           # Basic cleaning
- 📂 combined
  - 📄 train_raw.pkl                    # Non-clean train set
  - 📄 test_raw.pkl                     # Non-clean test set
  - 📄 train_feature_engineered.pkl     # Train set with feature engineering
  - 📄 test_feature_engineered.pkl      # Test set with feature engineering
  - 📄 train_impute.pkl                 # Train set with feature engineering and missing imputation
  - 📄 test_impute.pkl                  # Test set with feature engineering and missing imputation
  - 📄 train_scaled.pkl                 # Train set with feature engineering, missing imputation, and numerical scaling
  - 📄 test_scaled.pkl                  # Test set with feature engineering, missing imputation, and numerical scaling
  - 📄 train_ready.pkl                  # Train set with all pre-processing steps
  - 📄 test_ready.pkl                   # Test set with all pre-processing steps
  - 📄 train_NN_processed.pkl           # Train set with NN additional processing steps
  - 📄 test_NN_processed.pkl            # Test set with NN additional processing steps
- 📂 model
  - 📄 PyTorch_trained_model.pth        # Trained neural network

```
