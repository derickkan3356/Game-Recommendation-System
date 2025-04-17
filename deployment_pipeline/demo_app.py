# deployment_pipeline/app.py (DEMO VERSION - UI Only)

import streamlit as st
import pandas as pd # Keep for st.dataframe if displaying inputs/dummy output
import random       # Keep for dummy user ID if displaying inputs

# --- NO Backend Logic Import ---
# We are not importing from prediction_pipeline in this demo version

# --- NO Configuration for File Paths Needed ---
# No data or models are loaded

# --- NO Asset Loading Needed ---
# The load_data() and load_models_and_encoders() functions are removed

# --- Helper Function for Input Parsing (Still useful for demo) ---
def parse_list_input(input_str, expected_type=int):
    """Parses comma-separated string into a list of numbers or None."""
    # Simplified parsing for demo - detects errors but doesn't stop execution immediately
    if not isinstance(input_str, str) or not input_str.strip():
        return [], None # Return empty list and no error
    items = []
    error_msg = None
    for item in input_str.split(','):
        item_stripped = item.strip()
        if item_stripped.lower() in ['none', 'na', 'n/a', 'null', '']:
            items.append(None)
        else:
            try:
                items.append(expected_type(item_stripped))
            except ValueError:
                # Store the error message but continue parsing other items
                error_msg = f"Invalid input: Could not convert '{item_stripped}' to {expected_type.__name__}."
                items.append(f"ERROR: {item_stripped}") # Include error marker
    if error_msg:
        st.warning(error_msg + " Please check comma-separated lists for errors.")
    return items, error_msg # Return list and potential error message

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🎮 Game Recommendation Engine (UI Demo)")
st.info("ℹ️ This is a demonstration of the user interface only. No actual recommendations are generated.")
st.markdown("---")


# --- Input Form (UI Definition) ---
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

    # --- Submit Button ---
    submitted = st.form_submit_button("✨ Get Recommendations ✨")

# --- Processing and Output (DEMO - Show Inputs or Placeholder) ---
if submitted:
    st.markdown("---")
    st.subheader("Processing Request... (Demo Mode)")

    # --- Input Parsing ---
    # Capture potential errors during parsing
    game_list, game_err = parse_list_input(game_list_str, int)
    playtime_list, playtime_err = parse_list_input(playtime_str, float)
    achievements_list, ach_err = parse_list_input(achievements_str, int)

    # --- Validation ---
    valid_input = True
    if game_err or playtime_err or ach_err:
        st.error("Please correct errors in the input lists above.")
        valid_input = False
    elif not (len(game_list) == len(playtime_list) == len(achievements_list)):
        st.error("Error: The number of items in Game IDs, Playtime, and Achievements lists must match.")
        valid_input = False
    elif not game_list:
        st.warning("Please enter at least one game.")
        valid_input = False

    if valid_input:
        # --- Display Captured Input ---
        user_input_data = {
            'user_id': random.randint(10000, 99999), # Example temporary ID
            'game_list': game_list,
            'playtime_forever': playtime_list,
            'achievements': achievements_list,
            'user_country_code': country_code.strip().upper() if country_code else None,
            'user_latitude': latitude,
            'user_longitude': longitude,
            'user_account_age_months': account_age,
        }

        st.success("Input captured successfully!")
        st.write("Captured Input Data:")
        st.json(user_input_data, expanded=True) # Display the dictionary

        # --- Placeholder for Actual Processing ---
        st.markdown("---")
        st.info("🚀 **In the full application, the backend prediction pipeline would run now.**")

        # --- Optional: Display Dummy Output Table ---
        st.subheader("🏆 Dummy Recommendations (Example Output)")
        dummy_data = {
            'App ID': [101, 202, 303, 404, 505],
            'Game Name': ['Demo Game A', 'Placeholder Quest', 'Example Adventure', 'Test Runner', 'Simulated Success'],
            'Score': [0.9512, 0.8834, 0.7500, 0.6891, 0.5523]
        }
        dummy_df = pd.DataFrame(dummy_data)
        dummy_df['Score'] = dummy_df['Score'].map('{:.4f}'.format) # Format score
        st.dataframe(dummy_df, hide_index=True, use_container_width=True)

# --- Sidebar Info ---
st.sidebar.header("About")
st.sidebar.info(
    "This is a **DEMO UI** for the game recommendation engine. "
    "It shows the input interface and captures data, but **does not** run the actual prediction pipeline."
)
st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
    1.  Enter comma-separated **App IDs**.
    2.  Enter corresponding **Playtime** and **Achievements** (use 'None' if unknown). *Lists must match length.*
    3.  (Optional) Provide profile details.
    4.  Click 'Get Recommendations' to see the captured input and dummy output.
    """
)