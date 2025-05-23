import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler 
from collections import Counter # Used within preprocess_test_data
import os
import joblib
import pickle # Used for loading/saving processed data/transformers
import time
import traceback
import gc 

# ==============================================================================
# Configuration / File Paths
# ==============================================================================

ASSETS_DIR = '../assets/combined/'
RAW_TRAIN_PATH = os.path.join(ASSETS_DIR, 'train_raw.pkl')
RAW_TEST_PATH = os.path.join(ASSETS_DIR, 'test_raw.pkl')
READY_TRAIN_PATH = os.path.join(ASSETS_DIR, 'train_ready.pkl') # Used for test embeddings
READY_TEST_PATH = os.path.join(ASSETS_DIR, 'test_ready.pkl') # Used for test embeddings

PROCESSED_X_TRAIN_PATH = 'X_train_processed.pkl'
PROCESSED_Y_TRAIN_PATH = 'y_train.pkl'
FITTED_TRANSFORMERS_PATH = 'fitted_transformers.pkl'

MODEL_SAVE_PATH = 'xgboost_recommender_model.joblib'

# Define TARGET_COLUMN consistently
TARGET_COLUMN = 'relevance_score'

# ==============================================================================
# Preprocessing Function for Test Data
# ==============================================================================

def preprocess_test_data(df_raw, df_ready_for_emb, fitted_transformers, x_train_columns):
    """
    Applies preprocessing steps to the test data using fitted transformers.
    Includes fixes for category handling errors.

    Args:
        df_raw (pd.DataFrame): Raw test data (like df_test_raw).
        df_ready_for_emb (pd.DataFrame): DataFrame containing test embeddings
                                         (like df_test_ready[['user_id', 'app_id', 'user_emb', 'game_emb']]).
        fitted_transformers (dict): Dictionary containing fitted objects and values from training.
        x_train_columns (pd.Index): The columns from the processed X_train dataset for final reindexing.

    Returns:
        pd.DataFrame: Processed X_test DataFrame, ready for prediction.
        None: If a critical error occurs.
    """
    print("[INFO] Starting preprocessing of test data...")
    print("-" * 70)

    if not isinstance(df_raw, pd.DataFrame) or df_raw.empty:
        print("[ERROR] Input df_raw is invalid or empty.")
        return None
    if not isinstance(df_ready_for_emb, pd.DataFrame) or df_ready_for_emb.empty:
        print("[ERROR] Input df_ready_for_emb is invalid or empty.")
        return None

    # Define column lists 
    embedding_cols = ['user_emb', 'game_emb']
    game_tags_col = 'game_tags'
    game_genres_col = 'game_genres'
    platform_col = 'game_available_platform'
    single_item_list_cols = ['game_developer', 'game_publisher']
    low_card_categorical_cols = ['user_country_code', 'game_esrb_rating']
    numerical_cols_to_keep = [
        'user_account_age_months', 'game_RAWG_weighted_avg_rating',
        'game_RAWG_ratings_count', 'game_RAWG_bookmark_count',
        'game_positive_review_count', 'game_negative_review_count',
        'game_avg_playtime_forever', 'game_median_playtime_forever',
        'game_current_price', 'game_initial_price', 'game_concurrent_user',
        'game_estimate_owners_lower', 'game_estimate_owners_upper',
    ]
    date_col = 'game_released_year'
    REF_YEAR = 1984

    try:
        # Create a copy
        X_test = df_raw.copy()
        print(f"[INFO] Initial test data shape: {X_test.shape}")

        # --- Initial Column Drop (Keep merge keys temporarily) ---
        merge_keys = ['user_id', 'app_id']
        if not all(key in X_test.columns for key in merge_keys):
             raise ValueError(f"Merge keys {merge_keys} not found in df_test_raw.")

        cols_to_drop = [
             'game_name', 'user_has_coordinates', 'user_latitude', 'user_longitude',
             'game_tba', 'game_metacritic_rating', 'game_RAWG_reviews_with_text_count',
             'game_RAWG_system_suggest_count', 'game_RAWG_reviews_count',
             'game_released_month', 'game_released_day',
             'game_RAWG_rating_5_percent', 'game_RAWG_rating_4_percent',
             'game_RAWG_rating_3_percent', 'game_RAWG_rating_1_percent',
             'game_RAWG_bookmark_type_yet_count','game_RAWG_bookmark_type_owned_count',
             'game_RAWG_bookmark_type_beaten_count', 'game_RAWG_bookmark_type_toplay_count',
             'game_RAWG_bookmark_type_dropped_count','game_RAWG_bookmark_type_playing_count',
             'game_available_parent_platforms', 'game_avg_user_score',
             'game_avg_playtime_last_2weeks', 'game_median_last_2weeks',
             'game_current_discount', TARGET_COLUMN # Ensure target is dropped if present
        ]
        temp_keys_to_keep = [key for key in merge_keys if key in cols_to_drop]
        cols_to_drop_now = [col for col in cols_to_drop if col not in temp_keys_to_keep]
        X_test.drop(columns=[col for col in cols_to_drop_now if col in X_test.columns], inplace=True, errors='ignore')
        print(f"[INFO] Columns remaining after initial drop: {len(X_test.columns)}")
        gc.collect()

        # ==============================\n",
        print("\n[PHASE 0] Merging Test Embeddings...")
        # ==============================\n",
        emb_cols_to_borrow = ['user_id', 'app_id', 'user_emb', 'game_emb']
        if not all(col in df_ready_for_emb.columns for col in emb_cols_to_borrow):
            raise ValueError(f"df_ready_for_emb must contain columns: {emb_cols_to_borrow}")

        print(f"  Extracting required columns from embedding source...")
        emb_df_test = df_ready_for_emb[emb_cols_to_borrow].drop_duplicates(subset=['user_id', 'app_id'], keep='first').copy()

        # --- Get Embedding Dim & Fill Value ---
        zero_emb_fill = fitted_transformers['embedding_fill_value']
        if zero_emb_fill is None:
             raise ValueError("Embedding fill value not found in fitted_transformers.")
        emb_dim = fitted_transformers['embedding_dim']
        if emb_dim is None or emb_dim <=0:
             raise ValueError("Embedding dimension not found or invalid in fitted_transformers.")
        print(f"    Using embedding dimension: {emb_dim}, Fill value: {zero_emb_fill}")

        # --- Merge ---
        print("  Merging embeddings onto X_test...")
        if X_test['user_id'].dtype != emb_df_test['user_id'].dtype:
            try: X_test['user_id'] = X_test['user_id'].astype(emb_df_test['user_id'].dtype)
            except Exception: raise TypeError("Could not cast user_id for merge.")
        if X_test['app_id'].dtype != emb_df_test['app_id'].dtype:
             try: X_test['app_id'] = X_test['app_id'].astype(emb_df_test['app_id'].dtype)
             except Exception: raise TypeError("Could not cast app_id for merge.")

        X_test = pd.merge(X_test, emb_df_test, on=['user_id', 'app_id'], how='left')

        # --- Handle NaNs from Merge ---
        for emb_col in embedding_cols:
            missing_embs = X_test[emb_col].isnull().sum()
            if missing_embs > 0:
                 print(f"  [WARN] Found {missing_embs} rows with missing {emb_col} post-merge. Filling with {zero_emb_fill}.")
                 nan_indices = X_test.index[X_test[emb_col].isnull()]
                 # Create list of lists to assign correctly
                 fill_values = [list(zero_emb_fill) for _ in range(len(nan_indices))]
                 X_test.loc[nan_indices, emb_col] = pd.Series(fill_values, index=nan_indices)


        X_test.drop(columns=['user_id', 'app_id'], inplace=True, errors='ignore')
        print("  [SUCCESS] Embeddings merged.")
        del emb_df_test
        gc.collect()
        print("[PHASE 0] Completed.")

        # ==============================\n",
        print("\n[PHASE 1] Flattening Embeddings...")
        # ==============================\n",
        emb_dim = fitted_transformers['embedding_dim']
        if emb_dim is None or emb_dim <= 0:
             print("  [WARN] Invalid embedding dimension found in transformers. Skipping flattening.")
             X_test.drop(columns=embedding_cols, inplace=True, errors='ignore')
        else:
            zero_emb_fill = fitted_transformers['embedding_fill_value']
            for col in embedding_cols:
                 if col in X_test.columns:
                     print(f"  [INFO] Flattening column: {col}...")
                     def check_and_coerce(item):
                         if isinstance(item, (list, np.ndarray)) and len(item) == emb_dim: return item
                         else: return list(zero_emb_fill)
                     X_test[col] = X_test[col].apply(check_and_coerce)

                     emb_cols_names = [f'{col}_{i}' for i in range(emb_dim)]
                     emb_data = np.array(X_test[col].tolist())
                     if emb_data.ndim != 2 or emb_data.shape[1] != emb_dim:
                          raise ValueError(f"Shape mismatch after converting '{col}' to array in test set.")

                     emb_df = pd.DataFrame(emb_data, columns=emb_cols_names, index=X_test.index).astype(float)
                     X_test = X_test.join(emb_df)
                     X_test = X_test.drop(columns=[col])
                     print(f"  [SUCCESS] Flattened '{col}'.")
                 else:
                     print(f"  [WARN] Embedding column '{col}' not found for flattening.")
            gc.collect()
        print("[PHASE 1] Completed.")


        # ==============================\n",
        print("\n[PHASE 2] Processing Game Tags (Top N)...")
        # ==============================\n",
        top_n_tags = fitted_transformers['top_n_tags']
        if top_n_tags is None:
             print("  [WARN] Top N tags list not found in transformers. Skipping tag processing.")
        elif game_tags_col not in X_test.columns:
             print(f"  [WARN] Tags column '{game_tags_col}' not found in test set.")
        else:
             try:
                 print(f"  Applying Binarizer using Top {len(top_n_tags)} tags from training...")
                 tag_lists_test = X_test[game_tags_col].apply(lambda x: x if isinstance(x, list) else [])
                 top_n_tags_set = set(top_n_tags)
                 filtered_tag_lists_test = tag_lists_test.apply(lambda lst: [tag for tag in lst if tag in top_n_tags_set])

                 mlb_top_tags = MultiLabelBinarizer(classes=top_n_tags)
                 # Use fit_transform here because transform alone might fail if test set has tags not in classes,
                 # but fit_transform handles it by ignoring them when classes are provided.
                 top_tag_features_test = mlb_top_tags.fit_transform(filtered_tag_lists_test)
                 top_tag_df_test = pd.DataFrame(
                     top_tag_features_test, columns=mlb_top_tags.classes_, index=X_test.index
                 ).add_prefix('tag_').astype(bool)

                 X_test = X_test.join(top_tag_df_test)
                 X_test = X_test.drop(columns=[game_tags_col])
                 print(f"  [SUCCESS] Processed '{game_tags_col}'. Added/Aligned {top_tag_df_test.shape[1]} tag features.")
                 del tag_lists_test, filtered_tag_lists_test, top_tag_features_test, top_tag_df_test
                 gc.collect()
             except Exception as e:
                 print(f"  [ERROR] Failed processing '{game_tags_col}' in test set: {e}")
                 print(traceback.format_exc())
        print("[PHASE 2] Completed.")


        # ==============================\n",
        print("\n[PHASE 3] Processing Game Genres (Binarize)...")
        # ==============================\n",
        mlb_genres = fitted_transformers['mlb_genres']
        if mlb_genres is None:
             print("  [WARN] Genres MLB not found in transformers. Skipping genres.")
        elif game_genres_col not in X_test.columns:
             print(f"  [WARN] Genres column '{game_genres_col}' not found in test set.")
        else:
             try:
                 print(f"  Applying fitted genres MLB...")
                 genre_lists_test = X_test[game_genres_col].apply(lambda x: x if isinstance(x, list) else [])
                 genre_features_test = mlb_genres.transform(genre_lists_test) # Use TRANSFORM
                 genre_df_test = pd.DataFrame(
                     genre_features_test, columns=mlb_genres.classes_, index=X_test.index
                 ).add_prefix('genre_').astype(bool)
                 X_test = X_test.join(genre_df_test)
                 X_test = X_test.drop(columns=[game_genres_col])
                 print(f"  [SUCCESS] Processed '{game_genres_col}'. Added/Aligned {genre_df_test.shape[1]} genre features.")
                 del genre_lists_test, genre_features_test, genre_df_test
                 gc.collect()
             except Exception as e:
                 print(f"  [ERROR] Failed processing '{game_genres_col}' in test set: {e}")
                 print(traceback.format_exc())
        print("[PHASE 3] Completed.")


        # ==============================\n",
        print("\n[PHASE 4] Processing Platforms (Binarize)...")
        # ==============================\n",
        mlb_platform = fitted_transformers['mlb_platform']
        if mlb_platform is None:
             print("  [WARN] Platform MLB not found in transformers. Skipping platforms.")
        elif platform_col not in X_test.columns:
              print(f"  [WARN] Platform column '{platform_col}' not found in test set.")
        else:
             try:
                 print(f"  Applying fitted platform MLB...")
                 platform_lists_test = X_test[platform_col].apply(lambda x: x if isinstance(x, list) else [])
                 platform_features_test = mlb_platform.transform(platform_lists_test) # Use TRANSFORM
                 platform_df_test = pd.DataFrame(
                     platform_features_test, columns=mlb_platform.classes_, index=X_test.index
                 ).add_prefix('platform_').astype(bool)
                 X_test = X_test.join(platform_df_test)
                 X_test = X_test.drop(columns=[platform_col])
                 print(f"  [SUCCESS] Processed '{platform_col}'. Added/Aligned {platform_df_test.shape[1]} platform features.")
                 del platform_lists_test, platform_features_test, platform_df_test
                 gc.collect()
             except Exception as e:
                 print(f"  [ERROR] Failed processing '{platform_col}' in test set: {e}")
                 print(traceback.format_exc())
        print("[PHASE 4] Completed.")


        # ==============================\n",
        print("\n[PHASE 5] Processing Dev/Publisher (Top N + Category)...")
        # ==============================\n",
        try:
            # 5a. Extract strings
            for col in single_item_list_cols:
                 if col in X_test.columns:
                     print(f"  [INFO] Extracting string from '{col}'...")
                     X_test[col] = X_test[col].apply(lambda x: str(x[0]) if isinstance(x, list) and len(x) > 0 else 'Unknown')
                     print(f"  [SUCCESS] Extracted strings for '{col}'.")
                 else:
                      print(f"  [WARN] Column '{col}' not found in test set for string extraction.")

            # 5b. Apply Top N + 'Other' using stored lists
            for col in single_item_list_cols:
                 if col in X_test.columns:
                     transformer_key = f'top_n_{col}s'
                     top_n = fitted_transformers.get(transformer_key)
                     if top_n is None:
                          print(f"  [WARN] Top N list '{transformer_key}' not found. Skipping Top N for '{col}'.")
                          continue
                     print(f"  [INFO] Applying Top {len(top_n)} / Other to '{col}' using stored list...")
                     top_n_set = set(top_n)
                     fill_other_cat = 'Other'
                     X_test[col] = X_test[col].apply(lambda x: x if x in top_n_set else fill_other_cat)
                     print(f"  [SUCCESS] Applied Top N / Other to '{col}'.")
                 else:
                      print(f"  [WARN] Column '{col}' not found for Top N processing.")

            # 5c. Convert Dev/Pub to category - This step is deferred to final enforcement
            print("  [INFO] Deferring category dtype conversion for Dev/Pub to final enforcement step.")
            gc.collect()
        except Exception as e:
            print(f"  [ERROR] Failed during Dev/Publisher processing in test set: {e}")
            print(traceback.format_exc())
        print("[PHASE 5] Completed.")


        # ==============================\n",
        print("\n[PHASE 6] Processing Other Categoricals (Category)...")
        # ==============================\n",
        try:
            for col in low_card_categorical_cols:
                 if col in X_test.columns:
                     print(f"  [INFO] Processing '{col}'...")
                     fill_val = 'Missing'
                     if X_test[col].isnull().any():
                          X_test.loc[X_test[col].isnull(), col] = fill_val
                          print(f"  [INFO] Filled NaNs in '{col}' with '{fill_val}'.")
                     X_test[col] = X_test[col].astype(str) # Ensure string before category enforcement
                     print(f"  [INFO] Deferring category dtype conversion for '{col}' to final enforcement step.")
                 else:
                     print(f"  [WARN] Column '{col}' not found in test set.")
            gc.collect()
        except Exception as e:
             print(f"  [ERROR] Failed processing other categoricals in test set: {e}")
             print(traceback.format_exc())
        print("[PHASE 6] Completed.")


        # ==============================\n",
        print("\n[PHASE 7] Processing Numerical Features (Imputation)...")
        # ==============================\n",
        try:
            print(f"  [INFO] Imputing selected numerical columns using stored medians...")
            numerical_cols_present_test = [col for col in numerical_cols_to_keep if col in X_test.columns]
            cols_actually_imputed_test = []

            for col in numerical_cols_present_test:
                 X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
                 if X_test[col].isnull().any():
                     median_val = fitted_transformers['numerical_medians'].get(col)
                     if median_val is None:
                          print(f"  [WARN] Median for '{col}' not found in transformers. Filling with 0.")
                          median_val = 0.0
                     X_test.loc[X_test[col].isnull(), col] = median_val
                     cols_actually_imputed_test.append(col)
                 X_test[col] = X_test[col].astype(float) # Ensure float after imputation

            if cols_actually_imputed_test:
                 print(f"    Imputed NaNs in test set: {cols_actually_imputed_test}")
            else:
                 print("    No NaNs found or needing imputation in selected numerical columns in test set.")
            gc.collect()
        except Exception as e:
             print(f"  [ERROR] Failed processing numerical features in test set: {e}")
             print(traceback.format_exc())
        print("[PHASE 7] Completed.")


        # ==============================\n",
        print("\n[PHASE 8] Processing Date Feature...")
        # ==============================\n",
        try:
            if date_col in X_test.columns:
                 print(f"  [INFO] Processing '{date_col}'...")
                 X_test[date_col] = pd.to_numeric(X_test[date_col], errors='coerce')
                 if X_test[date_col].isnull().any():
                     median_year = fitted_transformers['numerical_medians'].get(date_col)
                     if median_year is None:
                          print(f"  [WARN] Median year for '{date_col}' not found. Filling with {REF_YEAR}.")
                          median_year = float(REF_YEAR)
                     X_test.loc[X_test[date_col].isnull(), date_col] = median_year
                     print(f"    Imputed NaNs in '{date_col}' with median ({median_year:.0f}).")

                 new_date_col_name = f'{date_col}_since_{REF_YEAR}'
                 X_test[new_date_col_name] = X_test[date_col] - REF_YEAR
                 X_test[new_date_col_name] = X_test[new_date_col_name].astype(float)
                 X_test.drop(columns=[date_col], inplace=True)
                 print(f"  [SUCCESS] Created '{new_date_col_name}' and dropped original.")
            else:
                 print(f"  [WARN] Date column '{date_col}' not found in test set.")
            gc.collect()
        except Exception as e:
            print(f"  [ERROR] Failed processing date feature in test set: {e}")
            print(traceback.format_exc())
        print("[PHASE 8] Completed.")


        # ==============================\n",
        print("\n[PHASE 9] Final Alignment (Reindex)...\n")
        # ==============================\n",
        print(f"  [INFO] Aligning test columns to training columns order and presence ({len(x_train_columns)} columns)...")
        print(f"  Test columns BEFORE reindex: {len(X_test.columns)}")

        missing_cols = x_train_columns.difference(X_test.columns)
        extra_cols = X_test.columns.difference(x_train_columns)

        if not missing_cols.empty:
            print(f"  [INFO] Columns missing in test set (will be added and filled): {missing_cols.tolist()}")
        if not extra_cols.empty:
            print(f"  [INFO] Columns present in test but not train (will be removed): {extra_cols.tolist()}")

        # Reindex to match training columns exactly
        X_test = X_test.reindex(columns=x_train_columns, fill_value=0) # Fill new columns with 0 initially

        print(f"  Test columns AFTER reindex: {len(X_test.columns)}")
        if len(X_test.columns) != len(x_train_columns):
             print("  [ERROR] Column count mismatch after reindex!")

        # Final check and fill for specific types if needed (e.g., boolean)
        # This step might be less necessary if category enforcement below works well
        # but can catch columns added solely by reindex
        for col in missing_cols:
             if pd.api.types.is_bool_dtype(x_train_columns[col]): # Check original intended type
                 print(f"   Ensuring boolean column '{col}' is filled with False.")
                 X_test[col] = X_test[col].astype(bool)

        final_nan_check = X_test.isnull().sum().sum()
        if final_nan_check > 0:
            print(f"  [WARN] {final_nan_check} NaNs remain after reindex before final dtype enforcement!")
        else:
            print("  [INFO] No NaNs found after reindex.")
        print("[PHASE 9] Completed.")

        print("\n[INFO] Test data preprocessing finished.")
        print(f"[INFO] Final X_test shape: {X_test.shape}")
        return X_test

    except Exception as e_main:
        print(f"\n[FATAL ERROR] An unexpected error occurred during test data preprocessing.")
        print(traceback.format_exc())
        return None


def enforce_test_categories(X_test_processed, X_train):
    """
    Enforces category dtypes on X_test based on X_train categories.
    This is crucial for XGBoost's enable_categorical=True.
    """
    print("\n--- Applying final category dtype enforcement on X_test_processed ---")
    cols_to_enforce = ['user_country_code', 'game_esrb_rating', 'game_developer', 'game_publisher']
    if X_test_processed is None or X_train is None:
        print("[ERROR] X_test_processed or X_train is None. Cannot enforce dtypes.")
        return None

    X_test_enforced = X_test_processed.copy() # Work on a copy

    for col in cols_to_enforce:
        if col in X_test_enforced.columns and col in X_train.columns:
            if isinstance(X_train[col].dtype, pd.CategoricalDtype):
                print(f"  Enforcing category dtype for: {col}")
                train_categories = X_train[col].cat.categories
                test_values_original = X_test_enforced[col].astype(str) # Ensure string comparison

                # Check for values in test not present in train categories
                unknown_values = set(test_values_original.unique()) - set(train_categories.astype(str))
                fill_cat = 'Other' if 'Other' in train_categories else 'Missing' if 'Missing' in train_categories else None

                if unknown_values and fill_cat:
                    print(f"    Found unknown values in test for '{col}': {list(unknown_values)[:5]}... Mapping to '{fill_cat}'.")
                    # Map unknown values to the fill category BEFORE converting to categorical
                    X_test_enforced[col] = test_values_original.apply(lambda x: x if x in train_categories.astype(str) else fill_cat)
                    # Ensure the fill category is in the target categories if it wasn't
                    if fill_cat not in train_categories:
                         train_categories = train_categories.add_categories([fill_cat])
                elif unknown_values:
                     print(f"    [WARN] Found unknown values in test for '{col}' but no 'Other'/'Missing' category defined in train. Unknowns will become NaN.")

                # Convert using the (potentially updated) training categories
                X_test_enforced[col] = pd.Categorical(X_test_enforced[col], categories=train_categories, ordered=False)

                # Check for NaNs again after conversion (should only be if fill_cat was None and unknowns existed)
                if X_test_enforced[col].isnull().any():
                    print(f"    [WARN] NaNs detected in '{col}' after category conversion. This might indicate issues.")
                    # Attempt to fill NaNs if a fill category was identified
                    if fill_cat:
                         X_test_enforced[col] = X_test_enforced[col].cat.add_categories([fill_cat]) # Ensure category exists
                         X_test_enforced[col].fillna(fill_cat, inplace=True)
                         print(f"    Filled final NaNs in '{col}' with '{fill_cat}'.")


            else:
                 print(f"  [WARN] Cannot enforce category for '{col}' as it's not categorical in X_train.")
        else:
             print(f"  [WARN] Column '{col}' not found in X_test_processed or X_train for final dtype enforcement.")

    print("\n--- Verifying Dtypes of X_test_processed AFTER enforcement ---")
    cols_present = [col for col in cols_to_enforce if col in X_test_enforced.columns]
    if cols_present:
        print(X_test_enforced[cols_present].info())
    else:
         print("Columns to check not found in X_test_enforced.")
    print("-" * 70)
    return X_test_enforced


# ==============================================================================
# Main Script Logic
# ==============================================================================
if __name__ == "__main__":

    print("--- Loading Data ---")
    try:
        # Load pre-processed training data and fitted transformers
        print(f"Loading processed training data: {PROCESSED_X_TRAIN_PATH}, {PROCESSED_Y_TRAIN_PATH}")
        X_train = pd.read_pickle(PROCESSED_X_TRAIN_PATH)
        y_train = pd.read_pickle(PROCESSED_Y_TRAIN_PATH)
        print(f"Loading fitted transformers: {FITTED_TRANSFORMERS_PATH}")
        with open(FITTED_TRANSFORMERS_PATH, 'rb') as f:
            fitted_transformers = pickle.load(f)

        # Load raw test data and 'ready' test data (for embeddings)
        print(f"Loading raw test data: {RAW_TEST_PATH}")
        df_test_raw = joblib.load(RAW_TEST_PATH)
        print(f"Loading ready test data (for embeddings): {READY_TEST_PATH}")
        df_test_ready = joblib.load(READY_TEST_PATH) # Contains test embeddings

        print("Data loading complete.")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  df_test_raw shape: {df_test_raw.shape}")
        print(f"  df_test_ready shape: {df_test_ready.shape}")

    except FileNotFoundError as e:
        print(f"\n[ERROR] Required data file not found: {e}")
        print("Please ensure pre-processed files (X_train_processed.pkl, y_train.pkl, fitted_transformers.pkl)")
        print("and raw/ready test files (test_raw.pkl, test_ready.pkl) exist in the expected locations.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Failed to load necessary data: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n--- Preprocessing Test Data ---")
    # Ensure df_test_ready has the necessary columns before passing
    required_emb_cols = ['user_id', 'app_id', 'user_emb', 'game_emb']
    if all(c in df_test_ready.columns for c in required_emb_cols):
        X_test_processed = preprocess_test_data(
            df_test_raw,
            df_test_ready[required_emb_cols], # Pass only needed columns
            fitted_transformers,
            X_train.columns
        )
    else:
        print(f"[ERROR] df_test_ready missing required embedding columns: {required_emb_cols}")
        X_test_processed = None

    if X_test_processed is None:
        print("\n[ERROR] Test data preprocessing failed. Exiting.")
        sys.exit(1)

    print("\n--- Enforcing Final Category Dtypes for Test Data ---")
    # This step is crucial to fix the dtype mismatch for XGBoost
    X_test_final = enforce_test_categories(X_test_processed, X_train)

    if X_test_final is None:
        print("\n[ERROR] Final category dtype enforcement failed. Exiting.")
        sys.exit(1)

    # Define y_test from raw test data (ensure index alignment)
    print("\n--- Defining y_test ---")
    if TARGET_COLUMN in df_test_raw.columns:
        # Assume index aligns if X_test_final originates from df_test_raw copy
        y_test = df_test_raw[TARGET_COLUMN].loc[X_test_final.index].copy()
        print(f"Defined y_test from '{TARGET_COLUMN}'. Length: {len(y_test)}")
        if len(y_test) != len(X_test_final):
             print("[WARNING] Length mismatch between final X_test and y_test!")
    else:
        print(f"[ERROR] Target column '{TARGET_COLUMN}' not found in df_test_raw. Cannot create y_test.")
        y_test = None # Set y_test to None if it cannot be created

    print("\n--- Training XGBoost Model ---")
    # Define model parameters
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        enable_categorical=True, # Use built-in category handling
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1 # Use all cores
    )

    print("Starting model training...")
    start_time = time.time()
    model.fit(X_train, y_train) # Train on the loaded processed training data
    end_time = time.time()
    print(f"[SUCCESS] Model training completed in {(end_time - start_time):.2f} seconds.")

    # ave the trained model
    try:
        joblib.dump(model, MODEL_SAVE_PATH)
        print(f"Trained model saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to save trained model: {e}")


    print("\n--- Making Predictions on Test Set ---")
    predictions = None
    if X_test_final is not None:
        try:
            start_time = time.time()
            predictions = model.predict(X_test_final) # Predict on the final processed test data
            end_time = time.time()
            print(f"[SUCCESS] Predictions made in {(end_time - start_time):.2f} seconds.")
            print(f"Sample predictions: {predictions[:10]}")
        except Exception as e:
            print(f"[ERROR] Failed to make predictions: {e}")
            print(traceback.format_exc())
    else:
        print("[ERROR] Final processed test data 'X_test_final' not available.")


    print("\n--- Evaluating Model Performance ---")
    if y_test is not None and predictions is not None:
        if len(y_test) == len(predictions):
            try:
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, predictions)

                print(f"Evaluation Metrics:")
                print(f"  Mean Squared Error (MSE):      {mse:.4f}")
                print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
                print(f"  Mean Absolute Error (MAE):     {mae:.4f}")
            except Exception as e:
                 print(f"[ERROR] Failed during evaluation calculation: {e}")
        else:
             print(f"[ERROR] Length mismatch! y_test ({len(y_test)}) vs predictions ({len(predictions)}). Cannot evaluate.")
    elif y_test is None:
         print("[ERROR] `y_test` is not available. Cannot evaluate.")
    elif predictions is None:
         print("[ERROR] `predictions` are not available. Cannot evaluate.")

    print("-" * 70)
    print("Script finished.")

