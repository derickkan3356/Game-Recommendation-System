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
- [ ] Data Preprocessing
    - [x] Feature engineering
    - [ ] Missing value imputation
    - [ ] Scale numerical features
    - [ ] Encode categorical features
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
  - 📄 train_CF.pkl                     # Train set with feature engineering and embedding
  - 📄 test_CF.pkl                      # Test set with feature engineering and embedding

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
- 📜 collaborative_filtering.ipynb      # (train_feature_engineered.pkl, test_feature_engineered.pkl) -> (train_CF.pkl, test_CF.pkl)

📄 requirements.txt                     # Python library version
```
