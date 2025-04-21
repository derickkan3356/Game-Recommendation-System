import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import json
import time
import random
import os
import io
from google.cloud import storage # Use Google Cloud Storage
import pycountry
import tempfile

# --- Core Logic Import ---
# Assumes prediction_pipeline.py is in the same directory (/app in the container)
try:
    from prediction_pipeline import (
        preprocess_new_user,
        pair_non_played_games,
        NN_forward_pass
    )
except ImportError as e:
    st.error(f"Fatal Error: Failed to import from prediction_pipeline.py: {e}")
    st.stop()
except Exception as ex:
    st.error(f"Fatal Error during import: {ex}")
    st.exception(ex)
    st.stop()

# ==============================================================================
# Configuration
# ==============================================================================
try:
    # --- GCS Configuration ---
    GCS_BUCKET_NAME = "game-rec-team-27-data"
    GCS_LOOKUP_BLOB = "game_lookup.csv"
    GCS_ENCODER_BLOB = "user_country_encoder.pkl"
    GCS_ALS_BLOB = "trained_ALS.pkl"
    GCS_TRAIN_BLOB = "train_NN_processed.parquet" # Using Parquet format
    GCS_SCORE_BLOB = "steamWeb_raw.csv"
    GCS_NN_HYPERPARAMS_BLOB = "PyTorch_model_hyperparameters.json"
    GCS_NN_WEIGHTS_BLOB = "PyTorch_model_weights.pth"

    # --- Local Path Configuration (Inside Container) ---
    # NN models are downloaded here to match prediction_pipeline.py's expected path
    LOCAL_MODELING_DIR = "/app/modeling"
    LOCAL_NN_HYPERPARAMS_PATH = os.path.join(LOCAL_MODELING_DIR, GCS_NN_HYPERPARAMS_BLOB)
    LOCAL_NN_WEIGHTS_PATH = os.path.join(LOCAL_MODELING_DIR, GCS_NN_WEIGHTS_BLOB)

    # --- Feature Column Lists ---
    # (Assuming these lists are correct and complete)
    single_cat_cols = ['user_country_code']
    multi_cat_cols = ['game_tags', 'game_available_platform', 'game_developer', 'game_publisher']
    num_cols = ['user_latitude', 'user_longitude', 'user_account_age_months', 'game_RAWG_weighted_avg_rating', 'game_RAWG_ratings_count', 'game_RAWG_reviews_with_text_count', 'game_RAWG_bookmark_count', 'game_metacritic_rating', 'game_RAWG_system_suggest_count', 'game_RAWG_reviews_count', 'game_released_month', 'game_released_day', 'game_RAWG_rating_5_percent', 'game_RAWG_rating_4_percent', 'game_RAWG_rating_3_percent', 'game_RAWG_rating_1_percent', 'game_RAWG_bookmark_type_yet_count', 'game_RAWG_bookmark_type_owned_count', 'game_RAWG_bookmark_type_beaten_count', 'game_RAWG_bookmark_type_toplay_count', 'game_RAWG_bookmark_type_dropped_count', 'game_RAWG_bookmark_type_playing_count', 'game_positive_review_count', 'game_negative_review_count', 'game_avg_playtime_forever', 'game_median_playtime_forever', 'game_current_price', 'game_initial_price', 'game_concurrent_user', 'game_estimate_owners_lower', 'game_estimate_owners_upper', 'game_popularity', 'user_preference_game_popularity', 'user_preference_game_duration', 'user_preference_new_game', 'user_preference_avg_spent', 'user_preference_game_esrb_rating_Rating Pending', 'user_preference_game_esrb_rating_Missing', 'user_preference_game_esrb_rating_Mature', 'user_preference_game_esrb_rating_Everyone 10+', 'user_preference_game_esrb_rating_Teen', 'user_preference_game_esrb_rating_Everyone', 'user_preference_game_esrb_rating_Adults Only', 'user_preference_game_genres_Action', 'user_preference_game_genres_Adventure', 'user_preference_game_genres_Arcade', 'user_preference_game_genres_Board Games', 'user_preference_game_genres_Card', 'user_preference_game_genres_Casual', 'user_preference_game_genres_Educational', 'user_preference_game_genres_Family', 'user_preference_game_genres_Fighting', 'user_preference_game_genres_Indie', 'user_preference_game_genres_Massively Multiplayer', 'user_preference_game_genres_Platformer', 'user_preference_game_genres_Puzzle', 'user_preference_game_genres_RPG', 'user_preference_game_genres_Racing', 'user_preference_game_genres_Shooter', 'user_preference_game_genres_Simulation', 'user_preference_game_genres_Sports', 'user_preference_game_genres_Strategy', 'user_preference_game_platforms_3DO', 'user_preference_game_platforms_Android', 'user_preference_game_platforms_Apple Macintosh', 'user_preference_game_platforms_Atari', 'user_preference_game_platforms_Commodore / Amiga', 'user_preference_game_platforms_Linux', 'user_preference_game_platforms_Neo Geo', 'user_preference_game_platforms_Nintendo', 'user_preference_game_platforms_PlayStation', 'user_preference_game_platforms_SEGA', 'user_preference_game_platforms_Web', 'user_preference_game_platforms_Xbox', 'user_preference_game_platforms_iOS', 'game_released_year_since_1984.0']
    bool_cols = ['user_has_coordinates', 'game_tba', 'game_current_discount', 'game_esrb_rating_Rating Pending', 'game_esrb_rating_Missing', 'game_esrb_rating_Mature', 'game_esrb_rating_Everyone 10+', 'game_esrb_rating_Teen', 'game_esrb_rating_Everyone', 'game_esrb_rating_Adults Only', 'game_genres_Action', 'game_genres_Adventure', 'game_genres_Arcade', 'game_genres_Board Games', 'game_genres_Card', 'game_genres_Casual', 'game_genres_Educational', 'game_genres_Family', 'game_genres_Fighting', 'game_genres_Indie', 'game_genres_Massively Multiplayer', 'game_genres_Platformer', 'game_genres_Puzzle', 'game_genres_RPG', 'game_genres_Racing', 'game_genres_Shooter', 'game_genres_Simulation', 'game_genres_Sports', 'game_genres_Strategy', 'game_platforms_3DO', 'game_platforms_Android', 'game_platforms_Apple Macintosh', 'game_platforms_Atari', 'game_platforms_Commodore / Amiga', 'game_platforms_Linux', 'game_platforms_Neo Geo', 'game_platforms_Nintendo', 'game_platforms_PC', 'game_platforms_PlayStation', 'game_platforms_SEGA', 'game_platforms_Web', 'game_platforms_Xbox', 'game_platforms_iOS']
    cf_emb_cols = ['user_emb', 'game_emb']

except Exception as e:
    # Catch potential errors during configuration setup
    st.error(f"Configuration Error: {e}")
    st.exception(e)
    st.stop()

# ==============================================================================
# Data Loading Functions (with Caching)
# ==============================================================================

@st.cache_resource
def load_models_and_encoders(bucket_name: str, encoder_blob_name: str, als_blob_name: str):
    """Loads user country encoder and ALS model from GCS."""
    user_country_encoder = None
    model_CF = None
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Load Encoder
        st.caption(f"Loading encoder: gs://{bucket_name}/{encoder_blob_name}")
        encoder_blob = bucket.blob(encoder_blob_name)
        encoder_bytes = encoder_blob.download_as_bytes()
        user_country_encoder = joblib.load(io.BytesIO(encoder_bytes))

        # Load ALS Model
        st.caption(f"Loading ALS model: gs://{bucket_name}/{als_blob_name}")
        als_blob = bucket.blob(als_blob_name)
        als_bytes = als_blob.download_as_bytes()
        model_CF = joblib.load(io.BytesIO(als_bytes))

        # Basic validation
        if not isinstance(user_country_encoder, dict):
            st.warning("Loaded country encoder is not a dictionary.")
        if not hasattr(model_CF, 'recalculate_user'): # Check for an expected method
            st.warning("Loaded ALS model might be incorrect or incomplete.")

    except Exception as e:
        st.error(f"Error loading models/encoders from GCS: {e}")
        st.exception(e)
    return user_country_encoder, model_CF

@st.cache_data
def load_game_lookup(bucket_name: str, lookup_blob_name: str):
    """Loads the game lookup dataframe (app_id, game_name) from GCS."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        st.caption(f"Loading lookup: gs://{bucket_name}/{lookup_blob_name}")
        lookup_blob = bucket.blob(lookup_blob_name)

        if lookup_blob_name.endswith(".csv"):
            lookup_bytes = lookup_blob.download_as_bytes()
            lookup_df = pd.read_csv(io.BytesIO(lookup_bytes), usecols=['app_id', 'game_name'])
        else:
            st.error(f"Unsupported lookup file format: {lookup_blob_name}")
            return None

        # Data cleaning
        lookup_df['app_id'] = lookup_df['app_id'].astype(int)
        lookup_df.dropna(subset=['app_id', 'game_name'], inplace=True)
        lookup_df.drop_duplicates(subset=['app_id'], inplace=True)
        return lookup_df

    except Exception as e:
        st.error(f"Error loading game lookup from GCS: {e}")
        st.exception(e)
        return None

@st.cache_data
def prepare_ui_options(_lookup_df: pd.DataFrame, _user_country_encoder: dict):
    """Prepares dropdown options and mappings for the Streamlit UI."""
    game_options = ["- Error loading games -"]
    game_map = {}
    country_options = ["- Error loading countries -"]
    country_display_map = {}
    g_select_disabled = True
    c_select_disabled = True

    try:
        # Prepare Game Options
        if _lookup_df is not None and 'app_id' in _lookup_df.columns and 'game_name' in _lookup_df.columns:
            lookup_df_valid = _lookup_df.dropna(subset=['game_name']).copy()
            lookup_df_valid['app_id'] = lookup_df_valid['app_id'].astype(int)
            # Create a display name like "Game Title (ID: 12345)"
            lookup_df_valid['display_name'] = lookup_df_valid.apply(
                lambda row: f"{row['game_name']} (ID: {row['app_id']})", axis=1
            )
            game_options = sorted(lookup_df_valid['display_name'].unique().tolist())
            game_map = pd.Series(lookup_df_valid.app_id.values, index=lookup_df_valid.display_name).to_dict()
            game_options.insert(0, "- Select a game to add -")
            g_select_disabled = False
        else:
            st.error("Game selection cannot be prepared: Lookup data is invalid.")

        # Prepare Country Options
        country_options_list = ["- Select Country (Optional) -"]
        if _user_country_encoder and isinstance(_user_country_encoder, dict):
            temp_countries = {}
            # Use pycountry to get full country names from codes
            for code in sorted(list(_user_country_encoder.keys())):
                display_name = f"{code}" # Default if pycountry fails
                try:
                    country = pycountry.countries.get(alpha_2=code)
                    if country:
                        display_name = f"{country.name} ({code})"
                except Exception:
                    pass # Keep default display name if lookup fails
                temp_countries[display_name] = code
            country_options_list.extend(sorted(temp_countries.keys()))
            country_display_map = temp_countries
            c_select_disabled = False
        else:
            st.warning("Could not load country options from encoder.")
        country_options = country_options_list

    except Exception as e:
        st.error(f"Error during UI option preparation: {e}")

    return game_options, game_map, country_options, country_display_map, g_select_disabled, c_select_disabled

def load_heavy_data(bucket_name: str, train_blob_name: str, score_blob_name: str):
    """
    Downloads large train_df (parquet) and score_raw_df (csv) from GCS
    to temporary local files and then loads them into memory.
    Called only when the user submits the form.
    """
    st.info("Loading large datasets from GCS...")
    load_heavy_start = time.time()
    train_df, game_df_backend, score_raw_df = (None, None, None)
    temp_train_path = None
    temp_score_path = None

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # --- Load train_df (Parquet) ---
        train_blob = bucket.blob(train_blob_name)
        st.info(f"Checking: gs://{bucket_name}/{train_blob_name}...")
        train_blob.reload() # Get metadata like size
        total_size_train = train_blob.size if train_blob.size else 0

        if total_size_train > 0:
            st.info(f"File size: {total_size_train / 1e6:.1f} MB (Parquet)")
            # Download to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as temp_file:
                temp_train_path = temp_file.name
                st.info(f"Downloading {train_blob_name} to {temp_train_path}...")
                train_blob.download_to_filename(temp_train_path)
            st.info("Download complete. Loading DataFrame from Parquet...")
            train_df = pd.read_parquet(temp_train_path)
            st.info("train_df loaded.")
        else:
             st.error(f"{train_blob_name} not found or empty on GCS. Cannot proceed.")
             return None, None, None # Exit if essential training data is missing

        # --- Load score_raw_df (CSV) ---
        score_blob = bucket.blob(score_blob_name)
        st.info(f"Checking: gs://{bucket_name}/{score_blob_name}...")
        score_blob.reload()
        if score_blob.size > 0:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                temp_score_path = temp_file.name
                st.info(f"Downloading {score_blob_name} to {temp_score_path}...")
                score_blob.download_to_filename(temp_score_path)
            st.info("Download complete. Loading score_raw_df...")
            score_raw_df = pd.read_csv(temp_score_path, dtype={'user_id': int, 'app_id': int})
            st.info("score_raw_df loaded.")
        else:
            st.error(f"{score_blob_name} not found or empty on GCS. Cannot proceed.")
            # Depending on logic, maybe allow proceeding without score_raw_df? For now, exit.
            # If it's optional, handle the None case downstream.
            return None, None, None

        # --- Post-Load Processing & Validation ---
        if train_df is None or score_raw_df is None:
             raise ValueError("Failed to load one or more required dataframes.")
        if 'app_id' not in train_df.columns:
             raise ValueError("'app_id' column missing in the loaded train_df.")

        # Derive game features dataframe (used later)
        game_cols_to_keep = [col for col in train_df.columns if col.startswith('game_') or col == 'app_id']
        if 'game_name' in train_df.columns and 'game_name' not in game_cols_to_keep:
             game_cols_to_keep.append('game_name') # Keep game name if available
        game_df_backend = train_df[game_cols_to_keep].drop_duplicates(subset='app_id').reset_index(drop=True).copy()

        load_heavy_end = time.time()
        st.info(f"Heavy data loaded in {load_heavy_end - load_heavy_start:.2f} seconds.")
        return train_df, game_df_backend, score_raw_df

    except FileNotFoundError as fnf_ex:
        st.error(f"Error loading heavy data: File not found on GCS. Check blob names in config.")
        st.error(f"Details: {fnf_ex}")
        st.exception(fnf_ex)
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred loading heavy data from GCS: {e}")
        st.exception(e)
        return None, None, None
    finally:
        # Clean up temporary files
        if temp_train_path and os.path.exists(temp_train_path):
            try: os.remove(temp_train_path)
            except OSError as e_rem: st.warning(f"Could not delete temp train file {temp_train_path}: {e_rem}")
        if temp_score_path and os.path.exists(temp_score_path):
            try: os.remove(temp_score_path)
            except OSError as e_rem: st.warning(f"Could not delete temp score file {temp_score_path}: {e_rem}")


def download_nn_models(bucket_name: str, hyperparams_blob: str, weights_blob: str, target_dir: str, hyperparams_path: str, weights_path: str):
    """Downloads NN model files (hyperparams, weights) from GCS to a local directory."""
    nn_files_downloaded = False
    try:
        with st.spinner("Downloading NN model files..."):
            os.makedirs(target_dir, exist_ok=True) # Create target dir if needed
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)

            # Download Hyperparameters
            st.caption(f"Downloading {hyperparams_blob} to {hyperparams_path}...")
            blob_h = bucket.blob(hyperparams_blob)
            blob_h.download_to_filename(hyperparams_path)

            # Download Weights
            st.caption(f"Downloading {weights_blob} to {weights_path}...")
            blob_w = bucket.blob(weights_blob)
            blob_w.download_to_filename(weights_path)

            nn_files_downloaded = True
            st.caption("NN model files downloaded.")
    except Exception as e:
        st.error(f"Failed to download NN model files from GCS: {e}")
        st.exception(e)
    return nn_files_downloaded

def cleanup_nn_files(hyperparams_path: str, weights_path: str, target_dir: str):
    """Removes downloaded NN model files and the directory if empty."""
    try:
        if os.path.exists(hyperparams_path):
             os.remove(hyperparams_path)
        if os.path.exists(weights_path):
             os.remove(weights_path)
        # Attempt to remove the directory if it's empty
        if os.path.exists(target_dir) and not os.listdir(target_dir):
             os.rmdir(target_dir)
    except Exception as cleanup_e:
        st.warning(f"Could not clean up temp NN files/dir in {target_dir}: {cleanup_e}")

# ==============================================================================
# Streamlit UI Setup
# ==============================================================================

st.set_page_config(page_title="Game Recommendation Engine", layout="wide")
st.title("🎮 Game Recommendation Engine")

# --- Load Initial Assets ---
# Load non-heavy assets required for the UI immediately
# Uses caching defined in the functions above
init_load_start = time.time()
st.caption("Loading initial UI assets...")
user_country_encoder_init, model_CF_init = load_models_and_encoders(
    GCS_BUCKET_NAME, GCS_ENCODER_BLOB, GCS_ALS_BLOB
)
game_lookup_df_init = load_game_lookup(GCS_BUCKET_NAME, GCS_LOOKUP_BLOB)

# Prepare UI options based on loaded assets
GAME_OPTIONS_LIST, GAME_DISPLAY_TO_ID_MAP, COUNTRY_OPTIONS, COUNTRY_DISPLAY_MAP, game_select_disabled, country_select_disabled = prepare_ui_options(
    game_lookup_df_init, user_country_encoder_init
)
init_load_end = time.time()

# Stop if essential UI assets failed to load
if model_CF_init is None or game_lookup_df_init is None or game_select_disabled:
    st.error("Application cannot start: Failed loading essential UI assets from GCS.")
    st.stop()
else:
    st.caption(f"Initial assets loaded ({init_load_end - init_load_start:.2f}s). Ready for input.")


# --- Initialize Session State ---
if 'user_games' not in st.session_state:
    st.session_state.user_games = []

# --- UI Helper Functions ---
def add_game_to_list():
    """Adds the selected game and playtime to the session state list."""
    selected_game_display = st.session_state.get('new_game_select', None)
    playtime_hours = st.session_state.get('new_game_playtime_hrs', None)

    if not selected_game_display or selected_game_display == GAME_OPTIONS_LIST[0]:
        st.warning("Please select a game.")
        return

    app_id = GAME_DISPLAY_TO_ID_MAP.get(selected_game_display)
    if not app_id:
        st.error("Could not find App ID for the selected game.")
        return

    game_name = selected_game_display.split(" (ID:")[0]

    if len(st.session_state.user_games) >= 5:
        st.warning("Maximum of 5 games already added.")
        return

    if any(game['id'] == app_id for game in st.session_state.user_games):
        st.warning(f"'{game_name}' has already been added.")
        return

    # Store playtime, handle None if input is empty or invalid
    playtime_val_hrs = None
    if playtime_hours is not None:
       try:
           playtime_val_hrs = float(playtime_hours)
           if playtime_val_hrs < 0: playtime_val_hrs = None # Treat negative as None
       except ValueError:
           playtime_val_hrs = None # Treat non-numeric as None

    st.session_state.user_games.append({
        'name': game_name,
        'id': int(app_id),
        'playtime_hours': playtime_val_hrs
    })

    # Reset input fields after adding
    try:
        st.session_state.new_game_select = GAME_OPTIONS_LIST[0]
        st.session_state.new_game_playtime_hrs = None
    except Exception as e:
        print(f"Error resetting state after adding game: {e}") # Log error

    st.rerun() # Rerun script to update the displayed list

def remove_last_game():
    """Removes the most recently added game from the session state list."""
    if st.session_state.user_games:
        st.session_state.user_games.pop()
        st.rerun() # Rerun script to update the displayed list

# --- Main UI Layout ---

# Section 1: Build Game History
st.subheader("1. Add Games You've Played (Up to 5)")

if st.session_state.user_games:
    st.write("**Your Added Games:**")
    cols_header = st.columns([4, 2])
    cols_header[0].write("**Game**")
    cols_header[1].write("**Playtime (Hours)**")
    for game in st.session_state.user_games:
        cols_game = st.columns([4, 2])
        cols_game[0].text(f"{game['name']} (ID: {game['id']})")
        playtime_display = f"{game['playtime_hours']:.1f}" if game['playtime_hours'] is not None else "N/A"
        cols_game[1].text(playtime_display)
    st.button("Remove Last Game", on_click=remove_last_game, key="remove_last_game_btn")
    st.markdown("---")

if len(st.session_state.user_games) < 5:
    st.write(f"**Add Game {len(st.session_state.user_games) + 1} of 5:**")
    sel_col, play_col = st.columns([4, 2])
    with sel_col:
        st.selectbox(
            "Select Game",
            options=GAME_OPTIONS_LIST,
            key='new_game_select',
            index=0, # Default to placeholder
            disabled=game_select_disabled,
            help="Start typing game name to filter"
        )
    with play_col:
        st.number_input(
            "Playtime (Hours, optional)",
            min_value=0.0,
            step=0.5, # Allow half hours
            format="%.1f",
            value=st.session_state.get('new_game_playtime_hrs', None),
            key='new_game_playtime_hrs',
            placeholder="e.g., 10.5",
            disabled=game_select_disabled
        )
    st.button("Add Game to List", on_click=add_game_to_list, key="add_game_btn", disabled=game_select_disabled)
else:
    st.success("You have added the maximum of 5 games.")
st.markdown("---")


# Section 2: Optional User Details & Submit Form
st.subheader("2. Optional: Tell Us More About You")
with st.form("recommendation_form"):
    selected_country_display = st.selectbox(
        "Your Country",
        options=COUNTRY_OPTIONS,
        key='country_select',
        index=0, # Default to placeholder
        disabled=country_select_disabled,
        help="Select your country"
    )
    account_age_years = st.number_input(
        "Your Account Age (Years, optional)",
        value=None,
        min_value=0.0,
        step=0.5,
        format="%.1f",
        placeholder="e.g., 2.5"
    )
    st.markdown("---")
    submitted = st.form_submit_button("✨ Get Recommendations ✨")


# ==============================================================================
# Processing Logic (When Form Submitted)
# ==============================================================================
if submitted:
    if not st.session_state.user_games:
        st.error("Please add at least one game you've played before getting recommendations.")
        st.stop()

    st.markdown("---")
    st.subheader("⏳ Processing Request...")
    processing_start_time = time.time()

    # --- Load Heavy Data (On Demand) ---
    # Pass already loaded initial models to avoid reloading if possible
    # (Note: Caching should handle this, but passing explicitly is clearer)
    train_df, game_df_backend, score_raw_df = load_heavy_data(
        GCS_BUCKET_NAME, GCS_TRAIN_BLOB, GCS_SCORE_BLOB
    )
    if train_df is None or score_raw_df is None or game_df_backend is None:
        st.error("Failed to load necessary datasets for calculation. Please check logs.")
        st.stop() # Stop if heavy data loading failed

    # --- Filter User Games ---
    # Ensure entered games exist in the scoring data for valid calculations
    st.write("Checking entered games against available data...")
    original_game_count = len(st.session_state.user_games)
    valid_app_ids_in_scores = set(score_raw_df['app_id'].unique())
    filtered_user_games = [
        game for game in st.session_state.user_games if game['id'] in valid_app_ids_in_scores
    ]

    if len(filtered_user_games) < original_game_count:
        removed_count = original_game_count - len(filtered_user_games)
        st.warning(f"Note: {removed_count} game(s) were removed from your input as they lack necessary data for calculations.")
        if not filtered_user_games:
            st.error("None of your entered games could be found in the available data. Please try adding different games.")
            st.stop() # Stop if no valid games remain

    # --- Prepare User Input Dictionary for Pipeline ---
    game_list = [game['id'] for game in filtered_user_games]
    playtime_hours_list = [game['playtime_hours'] for game in filtered_user_games]
    # Convert hours to minutes, handle None
    playtime_minutes_list = [(h * 60.0) if h is not None else None for h in playtime_hours_list]
    achievements_list = [None] * len(game_list) # Placeholder if achievements aren't used

    # Map selected country display name back to code
    country_code = None
    if selected_country_display != COUNTRY_OPTIONS[0]:
         country_code = COUNTRY_DISPLAY_MAP.get(selected_country_display)
         if country_code: country_code = country_code.upper()

    # Convert account age years to months, handle None
    account_age_months = None
    if account_age_years is not None and account_age_years >= 0:
         account_age_months = float(account_age_years * 12.0)

    # Construct user dictionary expected by prediction_pipeline functions
    user_input_dict = {
        'user_id': random.randint(1_000_000, 9_999_999), # Generate a random temporary ID
        'game_list': game_list,
        'playtime_forever': playtime_minutes_list,
        'achievements': achievements_list,
        'user_country_code': country_code,
        'user_latitude': None, # Placeholder
        'user_longitude': None, # Placeholder
        'user_account_age_months': account_age_months
    }

    # --- Download NN Model Files ---
    # Downloads to /app/modeling/ to match prediction_pipeline.py expectation
    nn_files_downloaded = download_nn_models(
        GCS_BUCKET_NAME,
        GCS_NN_HYPERPARAMS_BLOB,
        GCS_NN_WEIGHTS_BLOB,
        LOCAL_MODELING_DIR,
        LOCAL_NN_HYPERPARAMS_PATH,
        LOCAL_NN_WEIGHTS_PATH
    )
    if not nn_files_downloaded:
        st.error("Failed to download NN model files needed for prediction.")
        st.stop()

    # --- Execute Recommendation Pipeline ---
    pipeline_start_time = time.time()
    predictions = None
    nn_time = 0.0
    top_recommendations = pd.DataFrame() # Initialize empty dataframe

    try:
        # Check if NN files exist at the expected path before proceeding
        if not os.path.exists(LOCAL_NN_HYPERPARAMS_PATH) or not os.path.exists(LOCAL_NN_WEIGHTS_PATH):
             st.error(f"NN Model files missing from expected path ({LOCAL_MODELING_DIR}). Cannot run prediction.")
             st.stop()

        # 1. Preprocess User Data
        with st.spinner("🔧 Preprocessing user..."):
            user_df = preprocess_new_user(
                user_input_dict, train_df, game_df_backend, score_raw_df,
                model_CF_init, # Use cached ALS model
                user_country_encoder_init # Use cached encoder
            )

        # 2. Pair with Candidate Games
        with st.spinner("🔄 Pairing candidates..."):
            paired_df = pair_non_played_games(user_input_dict, user_df, game_df_backend)

        if paired_df.empty:
            st.warning("No suitable candidate games found to recommend based on your input.")
            st.stop()

        # 3. Predict Scores with NN
        # Calls the function from prediction_pipeline.py
        # Assumes it finds models in the hardcoded '/app/modeling/' path
        with st.spinner("🧠 Predicting scores (NN)..."):
            nn_start = time.time()
            predictions = NN_forward_pass(
                paired_df, train_df, single_cat_cols, multi_cat_cols,
                num_cols, bool_cols, cf_emb_cols
            )
            nn_time = time.time() - nn_start

        # 4. Rank and Format Results
        with st.spinner("📊 Ranking results..."):
            if predictions is None or not isinstance(predictions, torch.Tensor):
                raise ValueError("Prediction step failed to return valid tensor.")
            if predictions.shape[0] != len(paired_df):
                raise ValueError("Prediction shape mismatch with candidate games.")

            paired_df['prediction'] = predictions.squeeze().detach().cpu().numpy()
            ranked_df = paired_df.sort_values(by='prediction', ascending=False)

            # Prepare columns for display
            display_cols = ['app_id']
            if 'game_name' in ranked_df.columns:
                display_cols.append('game_name')
            display_cols.append('prediction')

            top_recommendations = ranked_df[display_cols].head(10).reset_index(drop=True)
            top_recommendations['prediction'] = top_recommendations['prediction'].map('{:.4f}'.format) # Format score

            # Rename columns for better display
            rename_map = {'app_id': 'App ID', 'prediction': 'Score'}
            if 'game_name' in display_cols:
                rename_map['game_name'] = 'Game Name'
            top_recommendations.rename(columns=rename_map, inplace=True)

        # --- Display Results ---
        st.success("Recommendations generated!")
        st.subheader("🏆 Top 10 Recommendations")
        st.dataframe(top_recommendations, use_container_width=True, hide_index=True)

        total_pipeline_time = time.time() - pipeline_start_time
        total_processing_time = time.time() - processing_start_time
        st.caption(f"Processing Time: {total_processing_time:.2f}s (Pipeline: {total_pipeline_time:.2f}s, NN: ~{nn_time:.2f}s)")

    except Exception as pipeline_ex:
         st.error(f"An error occurred during the recommendation pipeline: {pipeline_ex}")
         st.exception(pipeline_ex) # Show traceback in logs/UI
         st.stop()
    finally:
        # --- Cleanup Downloaded NN Files ---
        if nn_files_downloaded:
            cleanup_nn_files(LOCAL_NN_HYPERPARAMS_PATH, LOCAL_NN_WEIGHTS_PATH, LOCAL_MODELING_DIR)


# ==============================================================================
# Sidebar Information
# ==============================================================================
st.sidebar.header("About")
st.sidebar.info("This application generates personalized game recommendations based on user input and machine learning models.")
st.sidebar.warning("Note: Large datasets and models are loaded on demand after submitting input, which may cause a delay.")

st.sidebar.header("How to Use")
st.sidebar.markdown("""
1.  Add up to 5 games you have played using the selection box. Optionally add playtime.
2.  (Optional) Provide your country and account age for potentially better recommendations.
3.  Click 'Get Recommendations'.
""")

st.sidebar.header("Data Info")
try:
    games_count = len(GAME_DISPLAY_TO_ID_MAP)
    countries_count = len(COUNTRY_OPTIONS) - 1 # Exclude placeholder
    st.sidebar.caption(f"Games available for input: {games_count}")
    st.sidebar.caption(f"Countries available for input: {countries_count}")
except NameError:
    st.sidebar.warning("Could not display data info.")
except Exception as e:
     st.sidebar.error(f"Error displaying sidebar info: {e}")

