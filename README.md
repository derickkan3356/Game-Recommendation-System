# A Game Recommendation System

Please run `pip install -r ..\requirements.txt` to synchronize our python library version.

Whenever a new library is installed or updated, please run `pip freeze > ..\requirements.txt` to update the requirements file.

Please store any assets in our google drive, since github have file size limit.

## Plan

- [x] Data collection
    - [x] RAWG
    - [x] SteamSpy
    - [x] SteamWeb
    - [x] Combine data
- [x] Data Preprocessing
    - [x] Feature engineering
    - [x] Missing value imputation
    - [x] Scale numerical features
    - [x] Encode categorical features
    - [x] Get user and game embedding from collaborative filtering
- [ ] Modelling
    - [ ] Neural Network
    - [ ] XGBoost
    - [ ] Evaluation and compare performance
- [ ] Deployment
    - [ ] Set up database
    - [ ] Front-end UI design
    - [ ] Data pipeline between front-end UI, database, and model prediction
    - [ ] Enhance prediction with user feedback
- [ ] Write report

## Folder Structure:
```
📂 assets (store in Google Drive)
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
- 📂 model
  - 📄 PyTorch_trained_model.pth        # Trained neural network

📂 data_collection
- 📂 RAWG
  - 📜 data_collection_rawg.ipynb       # -> games_data.json
  - 📜 add_steam_app_id_to_RAWG.ipynb   # games_data.json -> games_data_with_id.pkl
  - 📜 data_cleaning_RAWG.ipynb         # games_data_with_id.pkl -> RAWG_clean.pkl
- 📂 SteamSpy
  - 📜 SteamSpyAPI.ipynb                # -> (SteamSpy_raw.json, SteamSpy_clean.pkl)
- 📂 SteamWeb
  - 📜 crawl_steam_members.ipynb        # -> all_data_users.json
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

📂 visualization
- 📜 Visualization.ipynb

📄 requirements.txt                     # Python library version
```
