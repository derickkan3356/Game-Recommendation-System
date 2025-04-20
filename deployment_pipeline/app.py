# deployment_pipeline/app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import json
import time
import random
import os # <-- Required for path manipulation

# --- Import Core Logic ---
# Assuming necessary functions/classes are in prediction_pipeline.py (sibling file)
try:
    from prediction_pipeline import (
        NN_model, UserGameDataset,  # Import classes for type hinting or potential use (though NN_model isn't loaded here)
        preprocess_new_user,
        pair_non_played_games,
        NN_forward_pass
        # Add any other necessary functions/classes like impute_user_profile if called directly
    )
    st.sidebar.success("Core logic imported successfully!")
except ImportError as e:
    st.error(f"Fatal Error: Failed to import core logic from prediction_pipeline.py: {e}")
    st.error("Please ensure prediction_pipeline.py exists in the same directory as app.py and contains the required functions/classes.")
    st.stop() # Stop if core logic can't be loaded
except Exception as ex:
    st.error(f"Fatal Error during import: {ex}")
    st.exception(ex)
    st.stop()

# --- Configuration ---

# Calculate paths relative to this script's location
try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(APP_DIR) # Project root is one level up

    st.info(f"Project Root Directory detected as: {PROJECT_ROOT}")

    # Define key directories relative to project root
    MODELING_DIR = os.path.join(PROJECT_ROOT, "modeling")
    DATA_PREPROC_DIR = os.path.join(PROJECT_ROOT, "data_preprocessing")

    # ---> DEFINE WHERE YOUR DATA ACTUALLY LIVES <---
    # Examples - adjust these based on your actual project structure:
    # Option 1: If data lives in a root 'data/' folder
    # DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    # Option 2: If processed data lives within data_preprocessing
    DATA_DIR = DATA_PREPROC_DIR # Example: Assuming train_NN_processed lives here
    # Option 3: If raw data lives within data_collection/SteamWeb
    RAW_DATA_SOURCE_DIR = os.path.join(PROJECT_ROOT, "data_collection", "SteamWeb") # Example

    # --- PATHS TO ASSETS (ADJUST THESE!) ---
    # ---> UPDATE THIS PATH <---  Where is the main training data stored?
    TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train_NN_processed.pkl') # Example: Needs verification!

    # ---> UPDATE THIS PATH <--- Where is the raw score data CSV stored?
    SCORE_RAW_PATH = os.path.join(RAW_DATA_SOURCE_DIR, "steamWeb_raw.csv") # Example: Needs verification!

    # Paths to models/encoders (These should be correct based on your 'find' output)
    COUNTRY_ENCODER_PATH = os.path.join(MODELING_DIR, 'user_country_encoder.pkl')
    ALS_MODEL_PATH = os.path.join(DATA_PREPROC_DIR, 'trained_ALS.pkl')
    # NN Model weights/hyperparams are loaded internally by prediction_pipeline.py

    # --- Feature Column Lists (Copied from prediction_pipeline.py - VERIFY!) ---
    single_cat_cols = ['user_country_code']
    multi_cat_cols = ['game_tags', 'game_available_platform', 'game_developer', 'game_publisher']
    num_cols = ['user_latitude', 'user_longitude', 'user_account_age_months', 'game_RAWG_weighted_avg_rating', 'game_RAWG_ratings_count', 'game_RAWG_reviews_with_text_count', 'game_RAWG_bookmark_count', 'game_metacritic_rating', 'game_RAWG_system_suggest_count', 'game_RAWG_reviews_count', 'game_released_month', 'game_released_day', 'game_RAWG_rating_5_percent', 'game_RAWG_rating_4_percent', 'game_RAWG_rating_3_percent', 'game_RAWG_rating_1_percent', 'game_RAWG_bookmark_type_yet_count', 'game_RAWG_bookmark_type_owned_count', 'game_RAWG_bookmark_type_beaten_count', 'game_RAWG_bookmark_type_toplay_count', 'game_RAWG_bookmark_type_dropped_count', 'game_RAWG_bookmark_type_playing_count', 'game_positive_review_count', 'game_negative_review_count', 'game_avg_playtime_forever', 'game_median_playtime_forever', 'game_current_price', 'game_initial_price', 'game_concurrent_user', 'game_estimate_owners_lower', 'game_estimate_owners_upper', 'game_popularity', 'user_preference_game_popularity', 'user_preference_game_duration', 'user_preference_new_game', 'user_preference_avg_spent', 'user_preference_game_esrb_rating_Rating Pending', 'user_preference_game_esrb_rating_Missing', 'user_preference_game_esrb_rating_Mature', 'user_preference_game_esrb_rating_Everyone 10+', 'user_preference_game_esrb_rating_Teen', 'user_preference_game_esrb_rating_Everyone', 'user_preference_game_esrb_rating_Adults Only', 'user_preference_game_genres_Action', 'user_preference_game_genres_Adventure', 'user_preference_game_genres_Arcade', 'user_preference_game_genres_Board Games', 'user_preference_game_genres_Card', 'user_preference_game_genres_Casual', 'user_preference_game_genres_Educational', 'user_preference_game_genres_Family', 'user_preference_game_genres_Fighting', 'user_preference_game_genres_Indie', 'user_preference_game_genres_Massively Multiplayer', 'user_preference_game_genres_Platformer', 'user_preference_game_genres_Puzzle', 'user_preference_game_genres_RPG', 'user_preference_game_genres_Racing', 'user_preference_game_genres_Shooter', 'user_preference_game_genres_Simulation', 'user_preference_game_genres_Sports', 'user_preference_game_genres_Strategy', 'user_preference_game_platforms_3DO', 'user_preference_game_platforms_Android', 'user_preference_game_platforms_Apple Macintosh', 'user_preference_game_platforms_Atari', 'user_preference_game_platforms_Commodore / Amiga', 'user_preference_game_platforms_Linux', 'user_preference_game_platforms_Neo Geo', 'user_preference_game_platforms_Nintendo', 'user_preference_game_platforms_PlayStation', 'user_preference_game_platforms_SEGA', 'user_preference_game_platforms_Web', 'user_preference_game_platforms_Xbox', 'user_preference_game_platforms_iOS', 'game_released_year_since_1984.0']
    bool_cols = ['user_has_coordinates', 'game_tba', 'game_current_discount', 'game_esrb_rating_Rating Pending', 'game_esrb_rating_Missing', 'game_esrb_rating_Mature', 'game_esrb_rating_Everyone 10+', 'game_esrb_rating_Teen', 'game_esrb_rating_Everyone', 'game_esrb_rating_Adults Only', 'game_genres_Action', 'game_genres_Adventure', 'game_genres_Arcade', 'game_genres_Board Games', 'game_genres_Card', 'game_genres_Casual', 'game_genres_Educational', 'game_genres_Family', 'game_genres_Fighting', 'game_genres_Indie', 'game_genres_Massively Multiplayer', 'game_genres_Platformer', 'game_genres_Puzzle', 'game_genres_RPG', 'game_genres_Racing', 'game_genres_Shooter', 'game_genres_Simulation', 'game_genres_Sports', 'game_genres_Strategy', 'game_platforms_3DO', 'game_platforms_Android', 'game_platforms_Apple Macintosh', 'game_platforms_Atari', 'game_platforms_Commodore / Amiga', 'game_platforms_Linux', 'game_platforms_Neo Geo', 'game_platforms_Nintendo', 'game_platforms_PC', 'game_platforms_PlayStation', 'game_platforms_SEGA', 'game_platforms_Web', 'game_platforms_Xbox', 'game_platforms_iOS']
    cf_emb_cols = ['user_emb', 'game_emb']

except Exception as e:
    st.error(f"Fatal Error during Configuration or Path Setup: {e}")
    st.exception(e)
    st.stop()


# --- Caching Data Loading ---

@st.cache_resource # Cache models and encoders
def load_models_and_encoders():
    """Loads ML models and encoders needed by app.py."""
    try:
        st.info(f"Loading country encoder from: {COUNTRY_ENCODER_PATH}")
        if not os.path.exists(COUNTRY_ENCODER_PATH): raise FileNotFoundError(f"Encoder not found at {COUNTRY_ENCODER_PATH}")
        user_country_encoder = joblib.load(COUNTRY_ENCODER_PATH)

        st.info(f"Loading ALS model from: {ALS_MODEL_PATH}")
        if not os.path.exists(ALS_MODEL_PATH): raise FileNotFoundError(f"ALS Model not found at {ALS_MODEL_PATH}")
        model_CF = joblib.load(ALS_MODEL_PATH)

        st.success("Loaded encoders and ALS model.")
        return user_country_encoder, model_CF
    except FileNotFoundError as fnf:
        st.error(f"Error loading file: {fnf}. Please check configuration paths.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred loading models/encoders: {e}")
        st.exception(e)
        return None, None

@st.cache_data # Cache dataframes
def load_data():
    """Loads necessary dataframes."""
    try:
        st.info(f"Loading training data from: {TRAIN_DATA_PATH}")
        if not os.path.exists(TRAIN_DATA_PATH): raise FileNotFoundError(f"Training data not found at {TRAIN_DATA_PATH}")
        train_df = joblib.load(TRAIN_DATA_PATH)
        # Perform necessary type conversions after loading pickle if needed
        for col in bool_cols:
             if col in train_df.columns and train_df[col].dtype != bool:
                 try: train_df[col] = train_df[col].astype(bool)
                 except: st.warning(f"Could not convert '{col}' to bool post-load.")

        st.info(f"Loading raw score data from: {SCORE_RAW_PATH}")
        if not os.path.exists(SCORE_RAW_PATH): raise FileNotFoundError(f"Raw score data not found at {SCORE_RAW_PATH}")
        score_raw_df = pd.read_csv(SCORE_RAW_PATH)

        # Derive game_df (unique games with game features) from train_df
        game_cols_to_keep = [col for col in train_df.columns if col.startswith('game_') or col == 'app_id']
        # Ensure 'game_name' is included if it exists and needed for display
        if 'game_name' in train_df.columns and 'game_name' not in game_cols_to_keep:
            game_cols_to_keep.append('game_name')
        elif 'game_name' not in train_df.columns:
             st.warning("'game_name' column not found in training data for display.")

        game_df = train_df[game_cols_to_keep].drop_duplicates(subset='app_id').reset_index(drop=True).copy()
        st.success("Loaded dataframes (train_df, score_raw_df, game_df).")
        return train_df, game_df, score_raw_df
    except FileNotFoundError as fnf:
        st.error(f"Error loading file: {fnf}. Please check configuration paths in app.py, especially TRAIN_DATA_PATH and SCORE_RAW_PATH.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred loading data: {e}")
        st.exception(e)
        return None, None, None

# --- Helper Function for Input Parsing ---
def parse_list_input(input_str, expected_type=int):
    """Parses comma-separated string into a list of numbers or None."""
    if not isinstance(input_str, str) or not input_str.strip():
        return []
    items = []
    error = None
    for item in input_str.split(','):
        item_stripped = item.strip()
        if item_stripped.lower() in ['none', 'na', 'n/a', 'null', '']:
            items.append(None)
        else:
            try:
                items.append(expected_type(item_stripped))
            except ValueError:
                error = f"Invalid input: Could not convert '{item_stripped}' to {expected_type.__name__}. Please enter comma-separated numbers or 'None'."
                break
    if error:
        st.error(error)
        return None # Indicate failure
    return items

# --- Load Assets On Start ---
load_start_time = time.time()
st.markdown("---")
st.subheader("Initializing Application...")
assets_loaded = False
with st.spinner("Loading dataframes, models, and encoders..."):
    train_df, game_df, score_raw_df = load_data()
    user_country_encoder, model_CF = load_models_and_encoders()
    # NN Model is NOT loaded here - prediction_pipeline.py handles it internally

assets_loaded = not(train_df is None or game_df is None or score_raw_df is None or user_country_encoder is None or model_CF is None)

if assets_loaded:
    loading_time = time.time() - load_start_time
    st.info(f"Asset loading complete in {loading_time:.2f} seconds.")
else:
    st.error("Fatal Error: Failed to load necessary assets. Application cannot start.")
    st.stop() # Halt execution if assets are missing

# --- Streamlit App UI ---
st.markdown("---")
st.title("🎮 Game Recommendation Engine")

# --- Input Form ---
st.subheader("Enter Your Game History")
st.caption("Provide the App IDs, playtime (in minutes), and achievement counts for games you've played. Enter 'None' or leave blank if unknown. Ensure lists are comma-separated and items align across the three boxes.")

with st.form("recommendation_form"):
    # Use examples from the original script as defaults
    game_list_str = st.text_area("Game App IDs (comma-separated)", "320, 628770, 838330, 725510, 839560")
    playtime_str = st.text_area("Playtime in Minutes (comma-separated, use 'None' if unknown)", "1000, None, 10, 5555, 10")
    achievements_str = st.text_area("Achievements Count (comma-separated, use 'None' if unknown)", "10, 0, 0, 0, None")

    st.subheader("Optional: Tell Us More About You")
    country_code = st.text_input("Your Country Code (e.g., US, GB)")
    latitude = st.number_input("Your Latitude (optional)", value=None, format="%.4f", placeholder="-122.4194")
    longitude = st.number_input("Your Longitude (optional)", value=None, format="%.4f", placeholder="37.7749")
    account_age = st.number_input("Your Account Age in Months (optional)", value=None, min_value=0, step=1, placeholder="12")

    submitted = st.form_submit_button("✨ Get Recommendations ✨")

# --- Processing and Output ---
if submitted:
    st.markdown("---")
    st.subheader("Processing Request...")
    # --- Input Parsing and Validation ---
    game_list = parse_list_input(game_list_str, int)
    playtime_list = parse_list_input(playtime_str, float) # Use float for playtime
    achievements_list = parse_list_input(achievements_str, int)

    valid_input = True
    if game_list is None or playtime_list is None or achievements_list is None:
        valid_input = False # Error already shown by parser
    elif not (len(game_list) == len(playtime_list) == len(achievements_list)):
        st.error("Error: The number of items in Game IDs, Playtime, and Achievements lists must match.")
        valid_input = False
    elif not game_list:
        st.warning("Please enter at least one game in your history.")
        valid_input = False

    if valid_input:
        # --- Construct User Dictionary ---
        user = {
            'user_id': random.randint(10000, 99999), # Temporary ID
            'game_list': game_list,
            'playtime_forever': playtime_list,
            'achievements': achievements_list,
            'user_country_code': country_code.strip().upper() if country_code else None,
            'user_latitude': latitude,
            'user_longitude': longitude,
            'user_account_age_months': account_age,
        }

        # Log the constructed input
        st.write("Constructed User Input (for backend):")
        col1, col2 = st.columns(2)
        with col1: st.json({k: v for k, v in user.items() if not isinstance(v, list)}, expanded=False)
        with col2: st.json({k: v for k, v in user.items() if isinstance(v, list)}, expanded=False)

        # --- Run Recommendation Pipeline ---
        pipeline_start_time = time.time()
        predictions = None # Initialize predictions variable

        # Filter score_raw_df for the specific user's games needed for imputation/relevance
        # Do this early to ensure it's available for preprocess_new_user
        try:
            user_score_raw_df = score_raw_df[score_raw_df['app_id'].isin(user['game_list'])].copy()
        except Exception as e:
             st.error(f"Error filtering score_raw_df for user games: {e}")
             st.exception(e)
             st.stop()

        # 1. Preprocess User
        with st.spinner("🔧 Preprocessing user..."):
            start_prep_time = time.time()
            try:
                user_df = preprocess_new_user(
                    user, train_df, game_df, user_score_raw_df, model_CF, user_country_encoder
                )
            except Exception as e:
                st.error(f"Error during user preprocessing: {e}")
                st.exception(e)
                st.stop()
            prep_time = time.time() - start_prep_time
            st.info(f"Preprocessing done ({prep_time:.2f}s).")

        # 2. Pair with Candidate Games
        with st.spinner("🔄 Pairing user with candidate games..."):
            start_pair_time = time.time()
            try:
                paired_df = pair_non_played_games(user, user_df, game_df)
            except Exception as e:
                 st.error(f"Error during game pairing: {e}")
                 st.exception(e)
                 st.stop()
            pair_time = time.time() - start_pair_time
            st.info(f"Pairing done ({pair_time:.2f}s). Found {len(paired_df)} candidates.")

            if paired_df.empty:
                st.warning("No candidate games found to recommend (perhaps all games in the dataset were played?).")
                st.stop()

        # 3. Predict using NN (with CWD workaround)
        original_cwd = os.getcwd() # Store original CWD
        try:
            # Temporarily change CWD to project root before calling the function
            # This helps the hardcoded relative paths inside NN_forward_pass resolve correctly
            os.chdir(PROJECT_ROOT)
            st.info(f"Temporarily changed CWD to: {PROJECT_ROOT} for prediction.")

            with st.spinner("🧠 Predicting scores (NN model reloading internally)..."):
                start_nn_time = time.time()
                # Call NN_forward_pass WITHOUT model/device args
                predictions = NN_forward_pass(
                    paired_df,
                    train_df, # Still pass necessary dataframes
                    single_cat_cols, multi_cat_cols, num_cols, bool_cols, cf_emb_cols # Pass feature lists
                )
                nn_time = time.time() - start_nn_time
                st.warning(f"Prediction done ({nn_time:.2f}s). (Note: Includes NN model reload time)")

        except FileNotFoundError as fnf_in_pipeline:
             st.error(f"FileNotFoundError during prediction pipeline execution: {fnf_in_pipeline}")
             st.error(f"This likely means a hardcoded relative path within prediction_pipeline.py (like 'modeling/...') could not be found from the CWD '{PROJECT_ROOT}'. Please check these internal paths or ensure the necessary files exist at the expected relative locations from the project root.")
             st.stop()
        except Exception as e:
             st.error(f"Error during NN prediction step: {e}")
             st.exception(e)
             st.stop()
        finally:
            # --- IMPORTANT: Change CWD back ---
            os.chdir(original_cwd)
            st.info(f"Restored CWD to: {original_cwd}")

        # 4. Process Results
        with st.spinner("📊 Processing and Ranking results..."):
            start_rank_time = time.time()
            try:
                # Ensure predictions tensor is valid before proceeding
                if predictions is None or not isinstance(predictions, torch.Tensor) or predictions.numel() == 0:
                     st.error("Prediction step returned invalid or no results.")
                     st.stop()

                # Detach predictions from graph, move to CPU, convert to numpy
                paired_df['prediction'] = predictions.squeeze().detach().cpu().numpy()
                # Sort by the prediction score
                ranked_df = paired_df.sort_values(by='prediction', ascending=False)

                # Select and rename columns for display
                display_cols = ['app_id']
                if 'game_name' in ranked_df.columns:
                    display_cols.append('game_name')
                display_cols.append('prediction')

                top_recommendations = ranked_df[display_cols].head(10).reset_index(drop=True)
                # Format score for display
                top_recommendations['prediction'] = top_recommendations['prediction'].map('{:.4f}'.format)
                # Rename columns for better presentation
                top_recommendations.rename(columns={'app_id': 'App ID', 'game_name': 'Game Name', 'prediction': 'Score'}, inplace=True)

            except Exception as e:
                 st.error(f"Error during results processing: {e}")
                 st.exception(e)
                 st.stop()
            rank_time = time.time() - start_rank_time
            st.info(f"Ranking done ({rank_time:.2f}s).")

        # --- Display Results ---
        st.success("Recommendations generated!")
        st.subheader("🏆 Top 10 Recommendations")
        st.dataframe(top_recommendations, use_container_width=True, hide_index=True)

        total_time = time.time() - pipeline_start_time
        st.caption(f"Total processing time for this request: {total_time:.2f} seconds.")
        st.caption(f"(Includes NN model reload time: ~{nn_time:.2f}s)")


# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info(
    "This app generates personalized game recommendations based on your game history "
    "and optional profile details, using a hybrid approach combining content features, "
    "collaborative filtering (ALS), and a deep learning model (Neural Network)."
)
st.sidebar.warning("Note: Due to current constraints, the underlying Neural Network model is reloaded on each request, which impacts performance.")
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
    1.  Enter comma-separated **App IDs** of games you've played.
    2.  Enter corresponding **Playtime** (minutes) and **Achievements**. Use 'None' if unknown. *Lists must match in length!*
    3.  (Optional) Provide country, location, and account age.
    4.  Click 'Get Recommendations'.
    """
)
st.sidebar.header("Configuration Info")
st.sidebar.json({
    "Project Root": PROJECT_ROOT,
    "Train Data Path": TRAIN_DATA_PATH,
    "Raw Score Path": SCORE_RAW_PATH,
    "ALS Model Path": ALS_MODEL_PATH,
    "Encoder Path": COUNTRY_ENCODER_PATH
}, expanded=False)