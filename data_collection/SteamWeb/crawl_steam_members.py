import requests
import re
import time
import json
import os
import sys
import argparse
import traceback
from tqdm import tqdm # Progress bar

# ==============================================================================
# API Interaction Functions
# ==============================================================================

def get_steam_id_from_vanity(api_key: str, vanity_url: str) -> str | None:
    """Resolves a Steam vanity URL to a Steam64 ID using the Steam Web API."""
    api_endpoint = "http://api.steampowered.com/ISteamUser/ResolveVanityURL/v0001/"
    params = {
        "key": api_key,
        "vanityurl": vanity_url
    }
    try:
        response = requests.get(api_endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("response", {}).get("success") == 1:
            return data["response"]["steamid"]
        else:
            message = data.get("response", {}).get("message", "Unknown reason")
            print(f"Warning: Could not resolve vanity URL '{vanity_url}': {message}", file=sys.stderr)
            return None
    except requests.exceptions.Timeout:
        print(f"Warning: Timeout resolving vanity URL '{vanity_url}'", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"Warning: Network error resolving vanity URL '{vanity_url}': {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Warning: Error decoding JSON response for vanity URL '{vanity_url}'. Response text: {response.text[:100]}... Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Unexpected error resolving vanity URL '{vanity_url}': {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def get_player_summaries(api_key: str, steam64_id: str) -> dict | None:
    """Fetches player summary data using a Steam64 ID."""
    api_endpoint = "https://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
    params = {
        "key": api_key,
        "steamids": steam64_id
    }
    try:
        response = requests.get(api_endpoint, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "response" in data and "players" in data["response"] and len(data["response"]["players"]) > 0:
            return data["response"]["players"][0]
        else:
            print(f"Warning: No player summary data found for Steam64 ID: {steam64_id}", file=sys.stderr)
            return None
    except requests.exceptions.Timeout:
         print(f"Warning: Timeout fetching player summary for Steam64 ID: {steam64_id}", file=sys.stderr)
         return None
    except requests.exceptions.RequestException as e:
        print(f"Warning: Network error fetching player summary for Steam64 ID {steam64_id}: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Warning: Error decoding JSON response for player summary {steam64_id}. Response text: {response.text[:100]}... Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Unexpected error fetching player summary for Steam64 ID {steam64_id}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


def get_owned_games(api_key: str, steam64_id: str) -> list | None:
    """Fetches owned games (including playtime and app info) using a Steam64 ID."""
    api_endpoint = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": api_key,
        "steamid": steam64_id,
        "include_appinfo": "1",
        "include_played_free_games": "1"
    }
    try:
        response = requests.get(api_endpoint, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        if response_data := data.get("response"):
            if games := response_data.get("games"):
                 return games
            else:
                 print(f"Info: No games found or profile is private for Steam64 ID: {steam64_id}", file=sys.stderr)
                 return []
        else:
             print(f"Warning: Unexpected response structure for owned games for Steam64 ID: {steam64_id}. Response: {data}", file=sys.stderr)
             return None

    except requests.exceptions.Timeout:
         print(f"Warning: Timeout fetching owned games for Steam64 ID: {steam64_id}", file=sys.stderr)
         return None
    except requests.exceptions.RequestException as e:
        print(f"Warning: Network error fetching owned games for Steam64 ID {steam64_id}: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"Warning: Error decoding JSON response for owned games {steam64_id}. Response text: {response.text[:100]}... Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Warning: Unexpected error fetching owned games for Steam64 ID {steam64_id}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


# ==============================================================================
# Helper Function
# ==============================================================================

def _ensure_dir_exists(directory_path: str):
    """Creates a directory if it doesn't exist."""
    try:
        os.makedirs(directory_path, exist_ok=True)
        # print(f"Checked/Created directory: {directory_path}") # Less verbose
    except OSError as e:
        print(f"ERROR: Failed to create directory '{directory_path}': {e}", file=sys.stderr)
        sys.exit(1)

# ==============================================================================
# Main Crawler Function
# ==============================================================================

def crawl_steam_group_members(api_key: str, base_group_url: str, num_pages: int, output_file: str):
    """
    Crawls Steam group member pages, extracts profiles, fetches user summaries
    and owned games via Steam API, and saves the combined data to a JSON file.
    """
    print(f"Starting crawl for group: {base_group_url}")
    print(f"Processing up to {num_pages} pages.")
    print(f"Output will be saved to: {output_file}")
    print("---")

    all_user_data = {}
    all_profile_urls = set()
    processed_steam_ids = set()

    # --- Step 1: Collect Profile URLs ---
    print("Step 1: Collecting profile URLs...")
    try:
        for page_num in tqdm(range(1, num_pages + 1), desc="Processing Pages", unit="page"):
            page_url = f"{base_group_url}?p={page_num}"
            try:
                response = requests.get(page_url, timeout=15)
                response.raise_for_status()
                page_content = response.text
                found_urls = re.findall(r"(https://steamcommunity\.com/(id|profiles)/([a-zA-Z0-9_-]+|[0-9]{17}))", page_content)
                if not found_urls and page_num > 1:
                     print(f"\nNo profile URLs found on page {page_num}, stopping URL collection.")
                     break
                all_profile_urls.update(url[0] for url in found_urls)

            except requests.exceptions.Timeout:
                 print(f"\nWarning: Timeout fetching group page: {page_url}", file=sys.stderr)
            except requests.exceptions.RequestException as e:
                 print(f"\nWarning: Error fetching group page {page_url}: {e}", file=sys.stderr)
            except Exception as e:
                 print(f"\nWarning: Unexpected error processing page {page_url}: {e}", file=sys.stderr)
                 traceback.print_exc(file=sys.stderr)
            finally:
                time.sleep(0.5) # Rate limit page requests

    except KeyboardInterrupt:
        print("\nUser interrupted URL collection.")
    finally:
        print(f"\nCollected {len(all_profile_urls)} unique profile URLs.")
        print("---")

    # --- Step 2: Fetch Data for Profiles ---
    if not all_profile_urls:
        print("No profile URLs collected, exiting.")
        return

    print("Step 2: Fetching user info via Steam API...")
    try:
        for profile_url in tqdm(all_profile_urls, desc="Fetching User Info", unit="user"):
            steam64_id = None
            identifier_type = None
            match = re.search(r"steamcommunity\.com/(id|profiles)/([^/?]+)", profile_url)

            if not match:
                print(f"Warning: Could not parse identifier from URL '{profile_url}', skipping.", file=sys.stderr)
                continue

            identifier_type = match.group(1)
            identifier = match.group(2)

            if identifier_type == "profiles":
                if identifier.isdigit() and len(identifier) == 17:
                     steam64_id = identifier
                else:
                     print(f"Warning: Invalid Steam64 ID found in URL '{profile_url}', skipping.", file=sys.stderr)
                     continue
            elif identifier_type == "id":
                steam64_id = get_steam_id_from_vanity(api_key, identifier)
                time.sleep(0.1) # Delay after vanity resolution attempt
                if not steam64_id: continue # Skip if resolution failed
            else:
                 print(f"Warning: Unknown profile URL format '{profile_url}', skipping.", file=sys.stderr)
                 continue

            if steam64_id in processed_steam_ids: continue

            if steam64_id:
                player_info = get_player_summaries(api_key, steam64_id)
                time.sleep(0.1)

                if player_info:
                    owned_games = get_owned_games(api_key, steam64_id)
                    time.sleep(0.1)

                    if owned_games is not None:
                        player_info["owned_games"] = owned_games
                        all_user_data[steam64_id] = player_info
                        processed_steam_ids.add(steam64_id)
                    else:
                         print(f"Warning: Failed to retrieve owned games for Steam64 ID: {steam64_id} (URL: {profile_url}), skipping user.", file=sys.stderr)
                else:
                    print(f"Warning: Failed to retrieve player summary for Steam64 ID: {steam64_id} (URL: {profile_url}), skipping user.", file=sys.stderr)

    except KeyboardInterrupt:
         print("\nUser interrupted data fetching.")
    finally:
        print("\nFinished fetching user info.")
        print(f"Total unique users processed: {len(all_user_data)}")
        print("---")

    # --- Step 3: Save Collected Data ---
    print(f"Step 3: Saving collected data to '{output_file}'...")
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir:
             _ensure_dir_exists(output_dir)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_user_data, f, indent=4, ensure_ascii=False)
        print("Data saved successfully.")
    except IOError as e:
        print(f"ERROR: Could not write to output file '{output_file}': {e}", file=sys.stderr)
    except Exception as e:
        print(f"ERROR: Unexpected error saving data to JSON: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

    print("--- Crawl Complete ---")


# ==============================================================================
# Command-Line Interface Setup
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl Steam group members and fetch their data via Steam API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )

    # REMOVED API Key Argument
    # parser.add_argument(
    #     "--api_key",
    #     type=str,
    #     required=True, # No longer required here
    #     help="Your Steam Web API key."
    # )
    parser.add_argument(
        "--group_url",
        type=str,
        default="https://steamcommunity.com/games/steam/members",
        help="The base URL of the Steam group members page."
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Maximum number of member pages to crawl."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="steam_member_data.json",
        help="Path to save the output JSON file."
    )

    args = parser.parse_args()

    # --- Get API Key from Environment Variable ---
    api_key = os.environ.get("STEAM_API_KEY")
    if not api_key:
        print("ERROR: Steam API Key not found.")
        print("Please set the STEAM_API_KEY environment variable before running the script.")
        print("Example (Bash/Zsh): export STEAM_API_KEY='YourKeyHere'")
        print("Example (Windows CMD): set STEAM_API_KEY=YourKeyHere")
        print("Example (PowerShell): $env:STEAM_API_KEY='YourKeyHere'")
        sys.exit(1) # Exit if key is missing

    print("--- Steam API Key loaded from environment variable ---")

    # Run the main crawling function
    crawl_steam_group_members(
        api_key=api_key,
        base_group_url=args.group_url,
        num_pages=args.pages,
        output_file=args.output
    )
