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
import boto3
import pycountry
import tempfile


# --- Try Importing Core Logic ---
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

# --- Configuration ---
try:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(APP_DIR) 

    # --- S3 Configuration ---
    S3_BUCKET_NAME = "s3-bucket-team-27" 
    S3_LOOKUP_KEY = "game_lookup.csv"
    S3_ENCODER_KEY = "user_country_encoder.pkl"
    S3_ALS_KEY = "trained_ALS.pkl"
    S3_TRAIN_KEY = "train_NN_processed.pkl"
    S3_SCORE_KEY = "steamWeb_raw.csv"
    S3_NN_HYPERPARAMS_KEY = "PyTorch_model_hyperparameters.json"
    S3_NN_WEIGHTS_KEY = "PyTorch_model_weights.pth"

    LOCAL_MODELING_DIR = os.path.join(PROJECT_ROOT, "modeling")
    LOCAL_NN_HYPERPARAMS_PATH = os.path.join(LOCAL_MODELING_DIR, 'PyTorch_model_hyperparameters.json')
    LOCAL_NN_WEIGHTS_PATH = os.path.join(LOCAL_MODELING_DIR, 'PyTorch_model_weights.pth')

    # --- Feature Column Lists ---

    single_cat_cols = ['user_country_code']
    multi_cat_cols = ['game_tags', 'game_available_platform', 'game_developer', 'game_publisher']
    num_cols = ['user_latitude', 'user_longitude', 'user_account_age_months', 'game_RAWG_weighted_avg_rating', 'game_RAWG_ratings_count', 'game_RAWG_reviews_with_text_count', 'game_RAWG_bookmark_count', 'game_metacritic_rating', 'game_RAWG_system_suggest_count', 'game_RAWG_reviews_count', 'game_released_month', 'game_released_day', 'game_RAWG_rating_5_percent', 'game_RAWG_rating_4_percent', 'game_RAWG_rating_3_percent', 'game_RAWG_rating_1_percent', 'game_RAWG_bookmark_type_yet_count', 'game_RAWG_bookmark_type_owned_count', 'game_RAWG_bookmark_type_beaten_count', 'game_RAWG_bookmark_type_toplay_count', 'game_RAWG_bookmark_type_dropped_count', 'game_RAWG_bookmark_type_playing_count', 'game_positive_review_count', 'game_negative_review_count', 'game_avg_playtime_forever', 'game_median_playtime_forever', 'game_current_price', 'game_initial_price', 'game_concurrent_user', 'game_estimate_owners_lower', 'game_estimate_owners_upper', 'game_popularity', 'user_preference_game_popularity', 'user_preference_game_duration', 'user_preference_new_game', 'user_preference_avg_spent', 'user_preference_game_esrb_rating_Rating Pending', 'user_preference_game_esrb_rating_Missing', 'user_preference_game_esrb_rating_Mature', 'user_preference_game_esrb_rating_Everyone 10+', 'user_preference_game_esrb_rating_Teen', 'user_preference_game_esrb_rating_Everyone', 'user_preference_game_esrb_rating_Adults Only', 'user_preference_game_genres_Action', 'user_preference_game_genres_Adventure', 'user_preference_game_genres_Arcade', 'user_preference_game_genres_Board Games', 'user_preference_game_genres_Card', 'user_preference_game_genres_Casual', 'user_preference_game_genres_Educational', 'user_preference_game_genres_Family', 'user_preference_game_genres_Fighting', 'user_preference_game_genres_Indie', 'user_preference_game_genres_Massively Multiplayer', 'user_preference_game_genres_Platformer', 'user_preference_game_genres_Puzzle', 'user_preference_game_genres_RPG', 'user_preference_game_genres_Racing', 'user_preference_game_genres_Shooter', 'user_preference_game_genres_Simulation', 'user_preference_game_genres_Sports', 'user_preference_game_genres_Strategy', 'user_preference_game_platforms_3DO', 'user_preference_game_platforms_Android', 'user_preference_game_platforms_Apple Macintosh', 'user_preference_game_platforms_Atari', 'user_preference_game_platforms_Commodore / Amiga', 'user_preference_game_platforms_Linux', 'user_preference_game_platforms_Neo Geo', 'user_preference_game_platforms_Nintendo', 'user_preference_game_platforms_PlayStation', 'user_preference_game_platforms_SEGA', 'user_preference_game_platforms_Web', 'user_preference_game_platforms_Xbox', 'user_preference_game_platforms_iOS', 'game_released_year_since_1984.0']
    bool_cols = ['user_has_coordinates', 'game_tba', 'game_current_discount', 'game_esrb_rating_Rating Pending', 'game_esrb_rating_Missing', 'game_esrb_rating_Mature', 'game_esrb_rating_Everyone 10+', 'game_esrb_rating_Teen', 'game_esrb_rating_Everyone', 'game_esrb_rating_Adults Only', 'game_genres_Action', 'game_genres_Adventure', 'game_genres_Arcade', 'game_genres_Board Games', 'game_genres_Card', 'game_genres_Casual', 'game_genres_Educational', 'game_genres_Family', 'game_genres_Fighting', 'game_genres_Indie', 'game_genres_Massively Multiplayer', 'game_genres_Platformer', 'game_genres_Puzzle', 'game_genres_RPG', 'game_genres_Racing', 'game_genres_Shooter', 'game_genres_Simulation', 'game_genres_Sports', 'game_genres_Strategy', 'game_platforms_3DO', 'game_platforms_Android', 'game_platforms_Apple Macintosh', 'game_platforms_Atari', 'game_platforms_Commodore / Amiga', 'game_platforms_Linux', 'game_platforms_Neo Geo', 'game_platforms_Nintendo', 'game_platforms_PC', 'game_platforms_PlayStation', 'game_platforms_SEGA', 'game_platforms_Web', 'game_platforms_Xbox', 'game_platforms_iOS']
    cf_emb_cols = ['user_emb', 'game_emb']

except Exception as e: st.error(f"Config Error: {e}"); st.exception(e); st.stop()

# --- Caching Functions ---

@st.cache_resource # Cache models and encoders loaded from S3
def load_models_and_encoders():
    """Loads models/encoders from S3."""
    user_country_encoder = None; model_CF = None
    try:
        s3_client = boto3.client('s3')
        st.caption(f"Loading encoder from s3://{S3_BUCKET_NAME}/{S3_ENCODER_KEY}")
        enc_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_ENCODER_KEY)
        user_country_encoder = joblib.load(io.BytesIO(enc_obj['Body'].read()))

        st.caption(f"Loading ALS model from s3://{S3_BUCKET_NAME}/{S3_ALS_KEY}")
        als_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_ALS_KEY)
        model_CF = joblib.load(io.BytesIO(als_obj['Body'].read()))

        if not isinstance(user_country_encoder, dict): st.warning(f"Country encoder not dict.")
        if not hasattr(model_CF, 'recalculate_user'): st.warning("ALS model might be incorrect.")
    except Exception as e: st.error(f"Error loading models/encoders from S3: {e}"); st.exception(e)
    return user_country_encoder, model_CF

@st.cache_data # Cache the small game lookup dataframe loaded from S3
def load_game_lookup():
    """Loads the game lookup file from S3."""
    try:
        s3_client = boto3.client('s3')
        st.caption(f"Loading lookup from s3://{S3_BUCKET_NAME}/{S3_LOOKUP_KEY}")
        lookup_obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_LOOKUP_KEY)

        if S3_LOOKUP_KEY.endswith(".csv"):
            lookup_df = pd.read_csv(io.BytesIO(lookup_obj['Body'].read()), usecols=['app_id', 'game_name'])
        else: st.error(f"Unsupported lookup format: {S3_LOOKUP_KEY}"); return None

        lookup_df['app_id'] = lookup_df['app_id'].astype(int)
        lookup_df.dropna(subset=['app_id', 'game_name'], inplace=True)
        lookup_df.drop_duplicates(subset=['app_id'], inplace=True)
        return lookup_df
    except Exception as e: st.error(f"Error loading game lookup from S3: {e}"); st.exception(e); return None

@st.cache_data # Cache the prepared UI options
def prepare_ui_options(_lookup_df, _user_country_encoder):
    """Prepares the options lists and mappings needed for UI selectboxes."""

    game_options = ["- Error loading games -"]; game_map = {}; country_options = ["- Error loading countries -"]
    country_display_map = {}
    g_select_disabled = True; c_select_disabled = True
    try:
        game_name_col = 'game_name'
        if _lookup_df is not None and game_name_col in _lookup_df.columns and 'app_id' in _lookup_df.columns:
            lookup_df_valid = _lookup_df.dropna(subset=[game_name_col]).copy()
            lookup_df_valid['app_id'] = lookup_df_valid['app_id'].astype(int)
            lookup_df_valid['display_name'] = lookup_df_valid.apply(lambda row: f"{row[game_name_col]} (ID: {row['app_id']})", axis=1)
            game_options = sorted(lookup_df_valid['display_name'].unique().tolist())
            game_map = pd.Series(lookup_df_valid.app_id.values, index=lookup_df_valid.display_name).to_dict()
            game_options.insert(0, "- Select a game to add -")
            g_select_disabled = False
        else: st.error("Game selection cannot be prepared: Lookup data invalid.")

        country_options_list = ["- Select Country (Optional) -"]
        if _user_country_encoder and isinstance(_user_country_encoder, dict):
            temp_countries = {}
            for code in sorted(list(_user_country_encoder.keys())):
                try:
                    country = pycountry.countries.get(alpha_2=code)
                    display_name = f"{country.name} ({code})" if country else f"{code}"
                    temp_countries[display_name] = code
                except Exception: display_name = f"{code}"; temp_countries[display_name] = code
            country_options_list.extend(sorted(temp_countries.keys()))
            country_display_map = temp_countries
            c_select_disabled = False
        else: st.warning("Could not load country options from encoder.")
        country_options = country_options_list
    except Exception as e: st.error(f"Error during UI option preparation: {e}")
    return game_options, game_map, country_options, country_display_map, g_select_disabled, c_select_disabled

# --- Load UPFRONT Assets (Fast Ones from S3) ---
load_start = time.time()
st.caption("Loading UI assets from S3...")
user_country_encoder, model_CF = load_models_and_encoders()
game_lookup_df = load_game_lookup()
GAME_OPTIONS_LIST, GAME_DISPLAY_TO_ID_MAP, COUNTRY_OPTIONS, COUNTRY_DISPLAY_MAP, game_select_disabled, country_select_disabled = prepare_ui_options(game_lookup_df, user_country_encoder)
load_end = time.time()

if model_CF is None or game_lookup_df is None or game_select_disabled:
    st.error("App cannot start: Failed loading upfront assets/UI options from S3."); st.stop()
else: st.caption(f"UI assets loaded ({load_end - load_start:.2f}s).")

# --- Load Heavy Data Function (ONLY called on submit from S3) ---
def load_heavy_data():
    """
    Downloads large train_df and score_raw_df from S3 to temporary
    local files (with progress calculated in main thread) and then loads them.
    """
    st.info(f"Loading large datasets from S3...")
    load_heavy_start = time.time()
    train_df, game_df_backend, score_raw_df = (None, None, None)
    temp_train_path = None
    temp_score_path = None
    s3_client = boto3.client('s3') # Initialize client once

    try:
        # --- Download train_df (Large File with Manual Progress) ---
        st.info(f"Getting size for train_df from s3://{S3_BUCKET_NAME}/{S3_TRAIN_KEY}...")
        s3_object_meta = s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=S3_TRAIN_KEY)
        total_size_train = int(s3_object_meta.get('ContentLength', 0))

        if total_size_train > 0:
            st.info(f"File size: {total_size_train / 1e6:.1f} MB")
            progress_bar_train = st.progress(0)
            progress_text_train = st.caption(f"Downloading train_df: 0.0 / {total_size_train / 1e6:.1f} MB (0%)")
            bytes_downloaded_train = 0
            chunk_size = 10 * 1024 * 1024 # Download in 10MB chunks (adjust if needed)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_train_file:
                temp_train_path = temp_train_file.name
                st.info(f"Downloading train_df to temp file {temp_train_path}...")
                s3_response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_TRAIN_KEY)
                s3_stream = s3_response['Body']

                while True:
                    chunk = s3_stream.read(chunk_size)
                    if not chunk:
                        break # Download finished
                    temp_train_file.write(chunk)
                    bytes_downloaded_train += len(chunk)
                    percentage = (bytes_downloaded_train / total_size_train) * 100
                    # Update progress bar and text IN MAIN THREAD
                    progress_bar_train.progress(min(int(percentage), 100))
                    progress_text_train.caption(f"Downloading train_df: {bytes_downloaded_train / 1e6:.1f} / {total_size_train / 1e6:.1f} MB ({min(int(percentage), 100)}%)")

            progress_text_train.caption("Download complete. Loading train_df from temp file (this may take time)...")
            progress_bar_train.progress(100) # Ensure it shows 100%
        else:
             st.warning("train_df has size 0 on S3. Creating empty temp file.")
             # Create empty file path for joblib.load
             with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_train_file:
                  temp_train_path = temp_train_file.name

        # Load from the temporary file PATH (still potentially slow for large file)
        train_df = joblib.load(temp_train_path)
        st.info("train_df loaded.")

        # --- Download score_raw_df (Smaller File - No progress shown, using download_file is fine) ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_score_file:
             temp_score_path = temp_score_file.name
             st.info(f"Downloading score_raw_df from s3://{S3_BUCKET_NAME}/{S3_SCORE_KEY} to temp file...")
             s3_client.download_file(S3_BUCKET_NAME, S3_SCORE_KEY, temp_score_path)
             st.info("Download complete. Loading score_raw_df from temp file...")
             score_raw_df = pd.read_csv(temp_score_path, dtype={'user_id': int, 'app_id': int})
             st.info("score_raw_df loaded.")

        # --- Post-Load Processing ---
        if train_df is None or score_raw_df is None: raise ValueError("Loaded dataframes are None")
        if 'app_id' not in train_df.columns: raise ValueError("'app_id' missing in train_df")

        # Derive game_df_backend
        game_name_col = 'game_name'; game_cols = [col for col in train_df.columns if col.startswith('game_') or col == 'app_id']
        if game_name_col in train_df.columns and game_name_col not in game_cols: game_cols.append(game_name_col)
        game_df_backend = train_df[game_cols].drop_duplicates(subset='app_id').reset_index(drop=True).copy()

        load_heavy_end = time.time()
        st.info(f"Heavy data loaded via temp files in {load_heavy_end - load_heavy_start:.2f} seconds.")
        return train_df, game_df_backend, score_raw_df

    except Exception as e:
        st.error(f"An error occurred loading heavy data via temp files: {e}")
        st.exception(e)
        return None, None, None
    finally:
        # --- Clean up temporary files ---
        if temp_train_path and os.path.exists(temp_train_path):
            try: os.remove(temp_train_path)
            except OSError as e: st.warning(f"Could not delete temp train file {temp_train_path}: {e}")
        if temp_score_path and os.path.exists(temp_score_path):
            try: os.remove(temp_score_path)
            except OSError as e: st.warning(f"Could not delete temp score file {temp_score_path}: {e}")

# --- Initialize Session State ---
if 'user_games' not in st.session_state:
    st.session_state.user_games = []

# --- UI Helper Functions ---
def add_game_to_list():
    selected_game_display = st.session_state.get('new_game_select', None)
    playtime_hours = st.session_state.get('new_game_playtime_hrs', None)
    if not selected_game_display or selected_game_display == GAME_OPTIONS_LIST[0]: st.warning("Please select a game."); return
    playtime_val_hrs = float(playtime_hours) if playtime_hours is not None and playtime_hours >= 0 else None
    app_id = GAME_DISPLAY_TO_ID_MAP.get(selected_game_display)
    if not app_id: st.error("Could not find App ID."); return
    game_name = selected_game_display.split(" (ID:")[0]
    if len(st.session_state.user_games) >= 5: st.warning("Max 5 games."); return
    if any(game['id'] == app_id for game in st.session_state.user_games): st.warning(f"'{game_name}' already added."); return
    st.session_state.user_games.append({'name': game_name, 'id': int(app_id), 'playtime_hours': playtime_val_hrs})
    try: st.session_state.new_game_select = GAME_OPTIONS_LIST[0]; st.session_state.new_game_playtime_hrs = None
    except Exception as e: print(f"Error resetting state: {e}")
    st.rerun()

def remove_last_game():
    if st.session_state.user_games: st.session_state.user_games.pop()


# --- Streamlit App UI ---
st.title("🎮 Game Recommendation Engine")

# --- Section 1: Build Your Game History (Max 5) ---
st.subheader("1. Add Games You've Played (Up to 5)")
if st.session_state.user_games:
    st.write("**Your Added Games:**"); cols_header = st.columns([4, 2]); cols_header[0].write("**Game**"); cols_header[1].write("**Playtime (Hours)**")
    for game in st.session_state.user_games:
        cols_game = st.columns([4, 2]); cols_game[0].text(f"{game['name']} (ID: {game['id']})"); cols_game[1].text(f"{game['playtime_hours'] if game['playtime_hours'] is not None else 'N/A'}")
    st.button("Remove Last Game", on_click=remove_last_game, key="remove_last_game_btn")
    st.markdown("---")
if len(st.session_state.user_games) < 5:
    st.write(f"**Add Game {len(st.session_state.user_games) + 1} of 5:**")
    sel_col, play_col = st.columns([4, 2])
    with sel_col: st.selectbox("Select Game", options=GAME_OPTIONS_LIST, key='new_game_select', index=0, disabled=game_select_disabled, help="Start typing game name")
    with play_col: st.number_input("Playtime (Hours)", min_value=0.0, step=None, format="%.1f", value=st.session_state.get('new_game_playtime_hrs', None), key='new_game_playtime_hrs', placeholder="e.g., 10.5", disabled=game_select_disabled)
    st.button("Add Game to List", on_click=add_game_to_list, key="add_game_btn", disabled=game_select_disabled)
else: st.success("You have added the maximum of 5 games.")
st.markdown("---")

# --- Section 2: Optional User Details ---
st.subheader("2. Optional: Tell Us More About You")
with st.form("recommendation_form"):
    # *** Use COUNTRY_OPTIONS with full names ***
    selected_country_display = st.selectbox("Your Country", options=COUNTRY_OPTIONS, key='country_select', index=0, disabled=country_select_disabled, help="Select your country")
    # *** Input Account Age in YEARS ***
    account_age_years = st.number_input("Your Account Age (Years, optional)", value=None, min_value=0.0, step=0.5, format="%.1f", placeholder="e.g., 2.5")
    st.markdown("---")
    submitted = st.form_submit_button("✨ Get Recommendations ✨")


# --- Processing and Output ---
if submitted:
    if not st.session_state.user_games: st.error("Please add at least one game first."); st.stop()
    st.markdown("---"); st.subheader("Processing Request...")
    processing_start_time = time.time()

    # --- Load HEAVY data only NOW ---
    train_df, game_df_backend, score_raw_df = load_heavy_data()
    if train_df is None: st.error("Failed to load data needed for calculation."); st.stop()

    # --- Filter user games based on score_raw_df ---
    st.write("Checking entered games against score data...")
    original_game_count = len(st.session_state.user_games)
    valid_app_ids_in_scores = set(score_raw_df['app_id'].unique())
    filtered_user_games = [game for game in st.session_state.user_games if game['id'] in valid_app_ids_in_scores]
    if len(filtered_user_games) < original_game_count:
        removed_count = original_game_count - len(filtered_user_games)
        st.warning(f"Note: {removed_count} game(s) were removed from input as they lack score data for calculations.")
        if not filtered_user_games: st.error("None of your entered games could be found in score data."); st.stop()

    # --- Prepare data for the backend using FILTERED list ---
    game_list = [game['id'] for game in filtered_user_games]
    playtime_hours_list = [game['playtime_hours'] for game in filtered_user_games]
    playtime_minutes_list = [(h * 60.0) if h is not None else None for h in playtime_hours_list]
    achievements_list = [None] * len(game_list) # Dummy list

    # Handle country selection - map display name back to code
    country_code = None
    if selected_country_display != COUNTRY_OPTIONS[0]:
         country_code = COUNTRY_DISPLAY_MAP.get(selected_country_display) # Map back
         if country_code: country_code = country_code.upper()

    # Convert Account Age Years to Months
    account_age_months = float(account_age_years * 12.0) if account_age_years is not None and account_age_years >= 0 else None

    # --- Construct User Dictionary for Backend ---
    user = { 'user_id': random.randint(1_000_000, 9_999_999), 'game_list': game_list, 'playtime_forever': playtime_minutes_list, 'achievements': achievements_list, 'user_country_code': country_code, 'user_latitude': None, 'user_longitude': None, 'user_account_age_months': account_age_months }

    # *** Download NN Model Files Temporarily from S3 ***
    nn_files_downloaded = False
    try:
        with st.spinner("Downloading NN model files..."):
            os.makedirs(LOCAL_MODELING_DIR, exist_ok=True) # Ensure local modeling dir exists
            s3_client = boto3.client('s3')
            st.caption(f"Downloading {S3_NN_HYPERPARAMS_KEY}...")
            s3_client.download_file(S3_BUCKET_NAME, S3_NN_HYPERPARAMS_KEY, LOCAL_NN_HYPERPARAMS_PATH)
            st.caption(f"Downloading {S3_NN_WEIGHTS_KEY}...")
            s3_client.download_file(S3_BUCKET_NAME, S3_NN_WEIGHTS_KEY, LOCAL_NN_WEIGHTS_PATH)
            nn_files_downloaded = True
    except Exception as e:
        st.error(f"Failed to download NN model files from S3: {e}"); st.exception(e); st.stop()


    # --- Run Recommendation Pipeline ---
    pipeline_start_time = time.time(); predictions = None; nn_time = 0
    try:
        # Check if NN files were actually downloaded locally before CWD change
        if not os.path.exists(LOCAL_NN_HYPERPARAMS_PATH) or not os.path.exists(LOCAL_NN_WEIGHTS_PATH):
             st.error("NN Model files failed to download locally. Cannot run prediction."); st.stop()

        with st.spinner("🔧 Preprocessing user..."):
            user_df = preprocess_new_user(user, train_df, game_df_backend, score_raw_df, model_CF, user_country_encoder)

        with st.spinner("🔄 Pairing candidates..."):
            paired_df = pair_non_played_games(user, user_df, game_df_backend)
        if paired_df.empty: st.warning("No candidates found."); st.stop()

        original_cwd = os.getcwd()
        try:
            os.chdir(PROJECT_ROOT) # Change CWD
            with st.spinner("🧠 Predicting scores (NN)..."):
                nn_start = time.time()
                # NN_forward_pass loads from the *local* files downloaded earlier
                predictions = NN_forward_pass(paired_df, train_df, single_cat_cols, multi_cat_cols, num_cols, bool_cols, cf_emb_cols)
                nn_time = time.time()-nn_start
        finally:
            os.chdir(original_cwd) # Change CWD back

        with st.spinner("📊 Ranking results..."):
            # ... ranking logic ...
            if predictions is None or not isinstance(predictions, torch.Tensor): st.error("Prediction failed."); st.stop()
            if predictions.shape[0] != len(paired_df): st.error("Prediction shape mismatch."); st.stop()
            paired_df['prediction'] = predictions.squeeze().detach().cpu().numpy()
            ranked_df = paired_df.sort_values(by='prediction', ascending=False)
            display_cols = ['app_id']; game_name_col = 'game_name'
            if game_name_col in ranked_df.columns: display_cols.append(game_name_col)
            display_cols.append('prediction')
            top_recommendations = ranked_df[display_cols].head(10).reset_index(drop=True)
            top_recommendations['prediction'] = top_recommendations['prediction'].map('{:.4f}'.format)
            rename_map = {'app_id': 'App ID', 'prediction': 'Score'}
            if game_name_col in display_cols: rename_map[game_name_col] = 'Game Name'
            top_recommendations.rename(columns=rename_map, inplace=True)

        # --- Display Results ---
        st.success("Recommendations generated!")
        st.subheader("🏆 Top 10 Recommendations"); st.dataframe(top_recommendations, use_container_width=True, hide_index=True)
        total_pipeline_time = time.time() - pipeline_start_time
        total_processing_time = time.time() - processing_start_time
        st.caption(f"Processing Time: {total_processing_time:.2f}s (Pipeline: {total_pipeline_time:.2f}s, NN: ~{nn_time:.2f}s)")

    except Exception as pipeline_ex:
         st.error(f"Pipeline Error: {pipeline_ex}"); st.exception(pipeline_ex)
         if 'original_cwd' in locals() and os.getcwd() != original_cwd: # Restore CWD
             try: os.chdir(original_cwd)
             except: pass
         st.stop()
    # --- Cleanup Temporary NN Files ---
    finally:
        if nn_files_downloaded:
            try:
                if os.path.exists(LOCAL_NN_HYPERPARAMS_PATH): os.remove(LOCAL_NN_HYPERPARAMS_PATH)
                if os.path.exists(LOCAL_NN_WEIGHTS_PATH): os.remove(LOCAL_NN_WEIGHTS_PATH)
            except Exception as cleanup_e: st.warning(f"Could not clean up temp NN files: {cleanup_e}")


# --- Sidebar Info ---
st.sidebar.header("About"); st.sidebar.info("Generates game recommendations using ML.")
st.sidebar.warning("Note: NN model reloads on each request.")
st.sidebar.header("How to Use"); st.sidebar.markdown("1. Add up to 5 games.\n2. (Optional) Add details.\n3. Click 'Get Recommendations'.")
st.sidebar.header("Data Info")
try:
    games_count = len(GAME_DISPLAY_TO_ID_MAP); countries_count = len(COUNTRY_OPTIONS) - 1
    st.sidebar.caption(f"Games in Lookup: {games_count}"); st.sidebar.caption(f"Countries in Enc: {countries_count}")
except NameError: st.sidebar.warning("Could not display config info.")
