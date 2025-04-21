import pandas as pd
import joblib
import os
import sys
import argparse
import traceback

# ==============================================================================
# Configuration & Path Setup
# ==============================================================================

# Assume this script lives in a subdirectory (e.g., 'scripts') one level below the project root
# Calculate paths relative to the project root directory
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    print(f"Detected Script Directory: {SCRIPT_DIR}")
    print(f"Assuming Project Root: {PROJECT_ROOT}")
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive environment)
    PROJECT_ROOT = os.getcwd() # Use current directory as root
    print(f"Warning: Could not determine script directory, using current CWD as Project Root: {PROJECT_ROOT}")


# --- Input File Paths (Relative to Project Root) ---
RAWG_CLEAN_PKL_PATH = os.path.join(PROJECT_ROOT, "data_collection", "RAWG", "RAWG_clean.pkl")
TRAIN_NN_PKL_PATH = os.path.join(PROJECT_ROOT, "data_preprocessing", "train_NN_processed.pkl")

# --- Output File Paths (Relative to Project Root) ---
# Lookup Output
LOOKUP_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "assets", "combined")
LOOKUP_CSV_PATH = os.path.join(LOOKUP_OUTPUT_DIR, "game_lookup.csv")

# Parquet Conversion Output (Save in the same directory as input PKL)
TRAIN_NN_PARQUET_PATH = os.path.splitext(TRAIN_NN_PKL_PATH)[0] + ".parquet"

# --- Parquet Conversion Settings ---
PARQUET_ENGINE = 'pyarrow'
PARQUET_COMPRESSION = 'snappy' # Options: 'snappy', 'gzip', 'brotli', None

# ==============================================================================
# Helper Functions
# ==============================================================================

def _ensure_dir_exists(directory_path: str):
    """Creates a directory if it doesn't exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Checked/Created directory: {directory_path}")
    except OSError as e:
        print(f"ERROR: Failed to create directory '{directory_path}': {e}")
        sys.exit(1)

def _check_file_exists(file_path: str, file_desc: str):
    """Checks if a file exists, prints error and exits if not."""
    if not os.path.exists(file_path):
        print(f"\nERROR: Input file for {file_desc} not found at '{file_path}'")
        sys.exit(1)

# ==============================================================================
# Core Processing Functions
# ==============================================================================

def create_game_lookup(input_path: str, output_dir: str, output_path: str):
    """
    Loads RAWG data, extracts game IDs and names, cleans them,
    and saves the lookup table as a CSV.
    """
    print("\n--- Task: Creating Game Lookup Table ---")
    _check_file_exists(input_path, "RAWG Clean Data")
    _ensure_dir_exists(output_dir)

    # --- Load Input ---
    print(f"Loading RAWG data: {input_path}...")
    try:
        df_rawg = joblib.load(input_path)
        print(f"Loaded successfully. Shape: {df_rawg.shape}")
    except Exception as e:
        print(f"ERROR: Failed to load input file '{input_path}': {e}")
        sys.exit(1)

    # --- Process ---
    print("Processing data...")
    try:
        id_col = 'steam_app_id'
        name_col = 'name'

        if id_col not in df_rawg.columns or name_col not in df_rawg.columns:
            print(f"ERROR: Input DataFrame missing required columns '{id_col}' or '{name_col}'.")
            print(f"Available columns: {df_rawg.columns.tolist()}")
            sys.exit(1)

        lookup_df = df_rawg[[id_col, name_col]].copy()
        lookup_df.dropna(subset=[id_col, name_col], inplace=True)
        lookup_df.drop_duplicates(subset=[id_col], keep='first', inplace=True)
        lookup_df.rename(columns={id_col: 'app_id', name_col: 'game_name'}, inplace=True)
        lookup_df['app_id'] = lookup_df['app_id'].astype(int)
        print("Data processed successfully.")

    except Exception as e:
        print(f"ERROR: Failed during data processing: {e}")
        sys.exit(1)

    # --- Save Output ---
    print(f"Saving lookup table to: {output_path}...")
    try:
        lookup_df.to_csv(output_path, index=False)
        print(f"Successfully saved {os.path.basename(output_path)}")
        print(f"Final lookup table shape: {lookup_df.shape}")
    except Exception as e:
        print(f"ERROR: Failed to save CSV output file '{output_path}': {e}")
        sys.exit(1)

    print("--- Game Lookup Table Creation Complete ---")


def convert_train_data_to_parquet(pkl_path: str, parquet_path: str, engine: str, compression: str):
    """
    Loads training data from a Pickle file, checks if it's a Pandas DataFrame,
    and saves it to Parquet format. Requires significant RAM.
    """
    print("\n--- Task: Converting Training Data Pickle to Parquet ---")
    _check_file_exists(pkl_path, "Training Data Pickle")

    try:
        # --- Load Pickle ---
        print(f"Loading data from Pickle file: {pkl_path}")
        print("(This requires enough RAM to hold the entire dataset in memory)")
        loaded_data = joblib.load(pkl_path)
        print("Pickle file loaded.")

        # --- Validate Type ---
        if not isinstance(loaded_data, pd.DataFrame):
            print(f"ERROR: Loaded data from '{pkl_path}' is type {type(loaded_data)}, not DataFrame.")
            sys.exit(1)
        df = loaded_data

        # --- Save Parquet ---
        print(f"Saving DataFrame to Parquet: {parquet_path}")
        print(f"(Engine: {engine}, Compression: {compression})")
        df.to_parquet(parquet_path, engine=engine, compression=compression, index=False)
        print("DataFrame saved successfully to Parquet.")

        # --- Verification ---
        pkl_size_mb = os.path.getsize(pkl_path) / (1024*1024)
        parquet_size_mb = os.path.getsize(parquet_path) / (1024*1024)
        print(f"\nOriginal Pickle size: {pkl_size_mb:.2f} MB")
        print(f"New Parquet size: {parquet_size_mb:.2f} MB")
        if parquet_size_mb < pkl_size_mb and pkl_size_mb > 0:
             print(f"Parquet file is {(1 - parquet_size_mb/pkl_size_mb)*100:.1f}% smaller.")

    except MemoryError:
        print("\nERROR: Ran out of memory loading the Pickle file.")
        print("Please run this on a machine with sufficient RAM.")
        sys.exit(1)
    except ImportError:
         print(f"\nERROR: Missing library. Is '{engine}' installed? (Requires 'pyarrow' or 'fastparquet')")
         sys.exit(1)
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during Parquet conversion: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("--- Parquet Conversion Complete ---")


# ==============================================================================
# Main Execution & Argument Parsing
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data preprocessing tasks.")
    parser.add_argument(
        '--task',
        type=str,
        choices=['lookup', 'convert_parquet', 'all'],
        required=True,
        help="Specify the task to run: 'lookup' (create game lookup), 'convert_parquet' (convert training data), or 'all' (run both)."
    )

    args = parser.parse_args()

    print("=============================================")
    print(f"      Running Preprocessing Task: {args.task.upper()}")
    print("=============================================")

    if args.task == 'lookup' or args.task == 'all':
        create_game_lookup(
            input_path=RAWG_CLEAN_PKL_PATH,
            output_dir=LOOKUP_OUTPUT_DIR,
            output_path=LOOKUP_CSV_PATH
        )

    if args.task == 'convert_parquet' or args.task == 'all':
        # Crucial Memory Warning
        print("\nWARNING: The Parquet conversion task requires significant RAM to load the large Pickle file.")
        # input("Press Enter to continue if you have enough RAM, or Ctrl+C to cancel...") # Optional confirmation prompt

        convert_train_data_to_parquet(
            pkl_path=TRAIN_NN_PKL_PATH,
            parquet_path=TRAIN_NN_PARQUET_PATH,
            engine=PARQUET_ENGINE,
            compression=PARQUET_COMPRESSION
        )

    print("\n=============================================")
    print("      Preprocessing Script Finished")
    print("=============================================")
