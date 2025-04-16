import pandas as pd
import numpy as np
import joblib
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cProfile
import pstats
import json

# ---------------------------
# Helper Functions
# ---------------------------
def impute_user_profile(user, train_df, user_features, score_raw_df):
    """
    Fill in missing user profile:
      - For country, use 'Missing'
      - For coordinates, compute a boolean flag
      - For latitude/longitude, use country medians (if country exists) or global medians.
      - For account age, simply use the global median.
      - For playtime and achievements, use mean of the game
    """
    # Impute user features
    for feature in user_features:
        if user.get(feature) is None:
            if feature == 'user_country_code':
                user[feature] = 'Missing'
            elif feature == 'user_has_coordinates':
                user[feature] = bool(user.get('user_latitude') and user.get('user_longitude'))
            elif feature in ['user_latitude', 'user_longitude']:
                if user.get('user_country_code') and user.get('user_country_code') != 'Missing':
                    # Impute using country median if available
                    median = train_df.groupby("user_country_code")[feature].median().to_dict().get(user.get('user_country_code'))
                else:
                    median = train_df[feature].median()
                user[feature] = median
            elif feature == 'user_account_age_months':
                user[feature] = train_df[feature].median()

    # Impute playtime_forever and achievements
    for feature in ['playtime_forever', 'achievements']:
        for i, (val, app_id) in enumerate(zip(user[feature], user['game_list'])):
            if val == None:  # Not input by user
                impute = score_raw_df[score_raw_df['app_id']==app_id][['app_id', feature]].groupby('app_id').agg('mean').values[0][0]
                user[feature][i] = float(impute) if not np.isnan(impute) else 0

    return user

def compute_user_preferences(user_game_df):
    """
    Derive aggregate user preferences based on games they have played.
    """
    # Identify game feature groups
    esrb_cols = [col for col in user_game_df.columns if 'esrb' in col]
    genres_cols = [col for col in user_game_df.columns if 'genres' in col]
    platforms_cols = [col for col in user_game_df.columns if 'platforms' in col]

    preferences = {}
    if esrb_cols:
        preferences.update(user_game_df[esrb_cols].mean().add_prefix("user_preference_").to_dict())
    if genres_cols:
        preferences.update(user_game_df[genres_cols].mean().add_prefix("user_preference_").to_dict())
    if platforms_cols:
        preferences.update(user_game_df[platforms_cols].mean().add_prefix("user_preference_").to_dict())

    # Numeric preferences: you can adjust these fields as needed.
    numeric_fields = {
       'game_popularity': 'user_preference_game_popularity',
       'game_avg_playtime_forever': 'user_preference_game_duration',
       'game_released_year_since_1984.0': 'user_preference_new_game',
       'game_initial_price': 'user_preference_avg_spent'
    }
    for game_field, pref_field in numeric_fields.items():
        if game_field in user_game_df.columns:
            preferences[pref_field] = user_game_df[game_field].mean()

    return preferences

def compute_relevance_score(
    df,
    score_raw_df,  # Historical data for scaling
    w1=0.7,  # Weight for playtime
    w2=0.3,  # Weight for achievements
    min_users_per_game=5  # Min users per game for normalization
):
    """
    Computes a per-game normalized 'relevance' score using log-transform and Min-Max scaling.
    If a game has no achievements, only playtime is used.
    """

    # Keep user id for later extraction
    user_id = df['user_id'].unique()[0]

    # Concat with pre-transform data
    df = pd.concat([df, score_raw_df])

    # Identify if a game actually has achievements available
    df['game_has_ach'] = df.groupby('app_id')['achievements'].transform(lambda x: x.max() > 0).astype(int)

    # Log-Transform Playtime & Achievements to Reduce Skew
    df['log_playtime'] = np.log1p(df['playtime_forever'])
    df['log_achievements'] = np.log1p(df['achievements'])

    # Per-Game Min-Max Scaling
    def min_max_scale(series):
        """Min-Max scales per game (0 to 1)."""
        min_val = series.min()
        max_val = series.max()
        return (series - min_val) / (max_val - min_val + 1e-6)  # Prevent division by zero

    df['minmax_playtime'] = df.groupby('app_id')['log_playtime'].transform(
        lambda x: min_max_scale(x) if len(x) >= min_users_per_game else 0
    )
    df['minmax_achievements'] = df.groupby('app_id')['log_achievements'].transform(
        lambda x: min_max_scale(x) if len(x) >= min_users_per_game else 0
    )

    # Compute final relevance score
    df['relevance_score'] = np.where(
        df['game_has_ach'] == 1,
        w1 * df['minmax_playtime'] + w2 * df['minmax_achievements'],  # Weighted sum if game has achievements
        df['minmax_playtime']  # Use only playtime otherwise
    )

    # Get back the user data
    df = df[df['user_id'] == user_id].copy()
    
    return dict(zip(df['app_id'], df['relevance_score']))

def build_user_item_matrix(user_relevance_scores, game2idx, n_games):
    """
    Build a CSR matrix for one user using the relevance scores,
    which is needed for recalculating the user embedding.
    """
    app_ids = list(user_relevance_scores.keys())
    # Filter only games that are in game2idx
    valid_games = [(app, user_relevance_scores[app]) for app in app_ids if app in game2idx]
    if not valid_games:
        return None  # or raise an error if no game is valid
    scores = [score for (_, score) in valid_games]
    game_indices = [game2idx[app] for (app, _) in valid_games]
    # Only one user, so row index is all zeros.
    user_item_coo = coo_matrix((scores, ([0]*len(game_indices), game_indices)), shape=(1, n_games))
    return user_item_coo.tocsr()

class UserGameDataset(Dataset):
    def __init__(self, single_cat_inputs, multi_cat_inputs, num_inputs, bool_inputs, CF_inputs):
        self.single_cat_inputs = single_cat_inputs
        self.multi_cat_inputs = multi_cat_inputs
        self.num_inputs = num_inputs
        self.bool_inputs = bool_inputs
        self.CF_inputs = CF_inputs

    def __len__(self):
        return len(self.num_inputs)

    def __getitem__(self, idx):
        # For each sample, return the inputs and the corresponding target (score)
        single_cat_input = self.single_cat_inputs[idx]
        multi_cat_input = [multi_cat[idx] for multi_cat in self.multi_cat_inputs]  # Extract each multi-cat feature
        numeric_input = self.num_inputs[idx]
        boolean_input = self.bool_inputs[idx]
        CF_input = [cf[idx] for cf in self.CF_inputs]

        # Combine input into single tensor
        all_input = torch.cat([single_cat_input, *multi_cat_input, numeric_input, boolean_input, *CF_input])
        
        return all_input
    
class NN_model(nn.Module):
    def __init__(self, single_cat_dims, multi_cat_dims, emb_dim, num_dim, bool_dim, CF_dim, hidden_sizes, dropout_rate,
                 single_cat_n_columns, multi_cat_n_columns, num_n_columns, bool_n_columns, CF_n_columns):
        super(NN_model, self).__init__()

        self.single_cat_n_columns = single_cat_n_columns
        self.multi_cat_n_columns = multi_cat_n_columns
        self.num_n_columns = num_n_columns
        self.bool_n_columns = bool_n_columns
        self.CF_n_columns = CF_n_columns

        # Embedding layers for single categorical columns
        self.single_cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim) for cat_dim in single_cat_dims
        ])

        # Embedding layers for multi-categorical columns with padding
        self.multi_cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim, padding_idx=0) for cat_dim in multi_cat_dims
        ])

        # Layer normalization for each single categorical embedding
        self.single_cat_layer_norms = nn.ModuleList([
            nn.LayerNorm(emb_dim) for _ in single_cat_dims
        ])

        # Layer normalization for each multi-categorical embedding
        self.multi_cat_layer_norms = nn.ModuleList([
            nn.LayerNorm(emb_dim) for _ in multi_cat_dims
        ])

        # Dense layer for numeric and boolean inputs
        self.num_dense = nn.Linear(num_dim, emb_dim)
        self.bool_dense = nn.Linear(bool_dim, emb_dim)
        self.CF_dense = nn.Linear(CF_dim, emb_dim)

        # Fully connected layers setup
        self.fc_layers = nn.ModuleList()

        # Batch normalization setup
        self.batch_norm_layers = nn.ModuleList()

        # Drop out
        self.dropout = nn.Dropout(dropout_rate)

        input_size = emb_dim * (len(single_cat_dims) + len(multi_cat_dims) + 3)

        for hidden_size in hidden_sizes:
            self.fc_layers.append(nn.Linear(input_size, hidden_size)) # Fully connected layers
            self.batch_norm_layers.append(nn.BatchNorm1d(hidden_size)) # Batch normalization
            input_size = hidden_size
        self.output = nn.Linear(input_size, 1)  # Output for regression

    def forward(self, all_inputs):
        # Extract single categorical columns
        single_cat_inputs = all_inputs[:, :self.single_cat_n_columns]

        # Extract multi-categorical columns
        i = 0
        multi_cat_inputs = []
        for n_col in self.multi_cat_n_columns:
            multi_cat_inputs.append(all_inputs[:, self.single_cat_n_columns + i : self.single_cat_n_columns + i + n_col])
            i += n_col

        # Extract numerical columns
        num_inputs = all_inputs[:, self.single_cat_n_columns + sum(self.multi_cat_n_columns) : self.single_cat_n_columns + sum(self.multi_cat_n_columns) + self.num_n_columns]

        # Extract Boolean columns
        bool_inputs = all_inputs[:, self.single_cat_n_columns + sum(self.multi_cat_n_columns) + self.num_n_columns : self.single_cat_n_columns + sum(self.multi_cat_n_columns) + self.num_n_columns + self.bool_n_columns]

        # Extract user and game embedding from CF
        CF_inputs = all_inputs[:, self.single_cat_n_columns + sum(self.multi_cat_n_columns) + self.num_n_columns + self.bool_n_columns : self.single_cat_n_columns + sum(self.multi_cat_n_columns) + self.num_n_columns + self.bool_n_columns + sum(self.CF_n_columns)]

        # Process single categorical columns through embeddings and normalize
        single_cat_emb = [
            self.single_cat_layer_norms[i](embed(single_cat_inputs[:, i].type(torch.long))) 
            for i, embed in enumerate(self.single_cat_embeddings)
        ]

        # Process multi-categorical columns through embeddings with max pooling the most representative category for each feature and then normalize
        multi_cat_emb = [
            self.multi_cat_layer_norms[i](torch.max(embed(multi_cat_inputs[i].type(torch.long)), dim=1)[0])
            for i, embed in enumerate(self.multi_cat_embeddings)
        ]

        # Process numeric, boolean and CF features through dense layers
        num_out = torch.relu(self.num_dense(num_inputs))
        bool_out = torch.relu(self.bool_dense(bool_inputs.float()))
        CF_out = torch.relu(self.CF_dense(CF_inputs.float()))

        # Concatenate all features together
        combined = torch.cat(single_cat_emb + multi_cat_emb + [num_out, bool_out, CF_out], dim=1)

        # Pass through fully connected layers
        x = combined
        for fc_layer, batch_norm_layer in zip(self.fc_layers, self.batch_norm_layers):
            x = torch.relu(fc_layer(x))
            x = batch_norm_layer(x)
            x = self.dropout(x)
        output = self.output(x)  # Regression output
        
        return output
    
def ranknet_loss(scores, targets):
    if targets.dim() == 1:
        targets = targets.unsqueeze(1)
        
    pairwise_diff = scores - scores.t()
    target_diff = targets - targets.t()
    
    valid_pairs = (target_diff != 0).float()
    
    Sij = torch.sign(target_diff)
    
    P_ij = torch.sigmoid(pairwise_diff)
    
    target_prob = (Sij + 1) / 2.0
    
    loss = F.binary_cross_entropy(P_ij, target_prob, reduction='none')
    
    loss = (loss * valid_pairs).sum() / valid_pairs.sum().clamp(min=1)
    return loss

def grouped_ranknet_loss(outputs, targets, user_ids):
    if not torch.is_tensor(user_ids):
        user_ids = torch.tensor(user_ids)
    unique_users = torch.unique(user_ids)
    group_loss_sum = 0.0
    group_count = 0
    for uid in unique_users:
        indices = (user_ids == uid).nonzero(as_tuple=True)[0]
        if indices.numel() < 2:
            continue
        group_output = outputs[indices]
        group_targets = targets[indices]
        group_loss = ranknet_loss(group_output, group_targets)
        group_loss_sum += group_loss
        group_count += 1
    if group_count > 0:
        return group_loss_sum / group_count
    else:
        return torch.tensor(0.0, device=outputs.device)
    
# ---------------------------
# Main Preprocessing Function
# ---------------------------
def preprocess_new_user(user, train_df, game_df, score_raw_df, model_CF, user_country_encoder):
    """
    Process a single new user's data to compute their aggregated features and embedding.
    Returns a DataFrame (one row) with all user features to be later joined with candidate games.
    """
    # Identify user feature columns used in train data (except for embedded features)
    user_features = [col for col in train_df.columns if col.startswith('user_') and col not in ['user_emb', 'user_id']]

    # Impute missing values in the user's profile.
    user = impute_user_profile(user, train_df, user_features, score_raw_df)
    
    # Encode country code.
    user['user_country_code_encoded'] = user_country_encoder.get(user['user_country_code'], -1)
    
    # Build a DataFrame for the games the user has played, merging with game metadata.
    user_games_df = pd.DataFrame({'app_id': user['game_list']})
    for key, value in user.items():
        if key != 'game_list':
            user_games_df[key] = value
    # Merge with game metadata from train_df (assumed to include columns with prefix 'game_')
    user_game_df = user_games_df.merge(game_df, on='app_id', how='inner')
    
    # Compute aggregate game preferences.
    game_preferences = compute_user_preferences(user_game_df)
    
    # Compute relevance scores for games played
    user_relevance_scores = compute_relevance_score(user_game_df[['user_id', 'app_id', 'playtime_forever', 'achievements']], score_raw_df)
    
    # Build a game index mapping from train data
    unique_games = train_df['app_id'].unique()
    game2idx = {g: i for i, g in enumerate(unique_games)}
    
    # Build user-item matrix for CF
    user_item_matrix = build_user_item_matrix(user_relevance_scores, game2idx, len(unique_games))
    
    # Recalculate user embedding from the CF model.
    user_embedding = model_CF.recalculate_user(user['user_id'], user_item_matrix)
    
    # Assemble final user features.
    user_features_dict = {key: user.get(key) for key in user_features}
    user_features_dict.update(game_preferences)
    user_features_dict['user_emb'] = user_embedding
    user_features_dict['user_id'] = user['user_id']
    
    # Return a one-row DataFrame of user features
    user_df = pd.DataFrame([user_features_dict])
    return user_df

def pair_non_played_games(user, user_df, game_df):
    # Filter games that are not yet played by user
    played_games = set(user['game_list'])
    candidate_games_df = game_df[~game_df['app_id'].isin(played_games)].copy()

    user_df['join_key'] = 1  # Dummy column for joining
    candidate_games_df['join_key'] = 1
    paired_df = candidate_games_df.merge(user_df, on='join_key').drop('join_key', axis=1)

    return paired_df

def NN_forward_pass(paired_df, train_df, single_cat_cols, multi_cat_cols, num_cols, bool_cols, cf_emb_cols):
    # Convert into a tensor and build Pytorch dataset
    single_cat_inputs = torch.tensor(paired_df[[col + '_encoded' for col in single_cat_cols]].values)
    multi_cat_inputs = [
        torch.tensor(np.stack(paired_df[col + '_encoded_padded'].values), dtype=torch.float32)
        for col in multi_cat_cols
    ]
    num_inputs = torch.tensor(paired_df[num_cols].values, dtype=torch.float32)
    bool_inputs = torch.tensor(paired_df[bool_cols].values)
    CF_inputs = [
        torch.tensor(np.stack(paired_df[col].values), dtype=torch.float32)
        for col in cf_emb_cols
    ]

    dataset = UserGameDataset(
        single_cat_inputs,
        multi_cat_inputs,
        num_inputs,
        bool_inputs,
        CF_inputs
    )

    loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    # Model configure setup
    single_cat_n_columns = single_cat_inputs.shape[1]
    multi_cat_n_columns = [tensor.shape[1] for tensor in multi_cat_inputs]
    num_n_columns = num_inputs.shape[1]
    bool_n_columns = bool_inputs.shape[1]
    CF_n_columns = [tensor.shape[1] for tensor in CF_inputs]

    single_cat_dims = [train_df[col + '_encoded'].nunique() + 1 for col in single_cat_cols]
    multi_cat_dims = [train_df[col + '_encoded_padded'].apply(lambda x: max(x)).max() + 2 for col in multi_cat_cols]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open('modeling/PyTorch_model_hyperparameters.json', 'r') as f:
        hyperparameters = json.load(f)

    model = NN_model(
        single_cat_dims=single_cat_dims,
        multi_cat_dims=multi_cat_dims,
        emb_dim=hyperparameters['embedding_dim'],
        num_dim=len(num_cols),
        bool_dim=len(bool_cols),
        CF_dim=sum(CF_n_columns),
        hidden_sizes=hyperparameters['hidden_sizes'],
        dropout_rate=hyperparameters['dropout_rate'],
        single_cat_n_columns=single_cat_n_columns,
        multi_cat_n_columns=multi_cat_n_columns,
        num_n_columns=num_n_columns,
        bool_n_columns=bool_n_columns,
        CF_n_columns=CF_n_columns
    ).to(device)

    model.load_state_dict(torch.load("modeling/PyTorch_model_weights.pth", map_location=device))

    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_output = model(batch)
            all_preds.append(batch_output.detach().cpu())

    predictions = torch.cat(all_preds, dim=0)
    return predictions

def main():
    # Sample user
    # These are the features we can prompt the user. Missing inputs will be imputed within the pipeline.
    user = {
        'user_id': 1,
        'game_list': [320, 628770, 838330, 725510, 839560],
        'playtime_forever': [1000, None, 10, 5555, 10],
        'achievements': [10, 0, 0, 0, None],
        'user_country_code': None,
        'user_has_coordinates': None,
        'user_latitude': None,
        'user_longitude': None,
        'user_account_age_months': None
    }

    # Load pre-trained and pre-processed objects.
    train_df = joblib.load('assets/combined/train_NN_processed.pkl')
    game_df = train_df.drop_duplicates(subset='app_id')[
        [col for col in train_df.columns if col.startswith('game_') or col == 'app_id']
    ].copy()
    score_raw_df = pd.read_csv("assets/SteamWeb/steamWeb_raw.csv")
    score_raw_df = score_raw_df[score_raw_df['app_id'].isin(user.get('game_list'))][['user_id', 'app_id', 'playtime_forever', 'achievements']].copy()
    user_country_encoder = joblib.load('modeling/user_country_encoder.pkl')
    model_CF = joblib.load('data_preprocessing/trained_ALS.pkl')

    single_cat_cols = ['user_country_code']
    multi_cat_cols = ['game_tags', 'game_available_platform', 'game_developer', 'game_publisher']
    num_cols = ['user_latitude', 'user_longitude', 'user_account_age_months', 'game_RAWG_weighted_avg_rating', 'game_RAWG_ratings_count', 'game_RAWG_reviews_with_text_count', 'game_RAWG_bookmark_count', 'game_metacritic_rating', 'game_RAWG_system_suggest_count', 'game_RAWG_reviews_count', 'game_released_month', 'game_released_day', 'game_RAWG_rating_5_percent', 'game_RAWG_rating_4_percent', 'game_RAWG_rating_3_percent', 'game_RAWG_rating_1_percent', 'game_RAWG_bookmark_type_yet_count', 'game_RAWG_bookmark_type_owned_count', 'game_RAWG_bookmark_type_beaten_count', 'game_RAWG_bookmark_type_toplay_count', 'game_RAWG_bookmark_type_dropped_count', 'game_RAWG_bookmark_type_playing_count', 'game_positive_review_count', 'game_negative_review_count', 'game_avg_playtime_forever', 'game_median_playtime_forever', 'game_current_price', 'game_initial_price', 'game_concurrent_user', 'game_estimate_owners_lower', 'game_estimate_owners_upper', 'game_popularity', 'user_preference_game_popularity', 'user_preference_game_duration', 'user_preference_new_game', 'user_preference_avg_spent', 'user_preference_game_esrb_rating_Rating Pending', 'user_preference_game_esrb_rating_Missing', 'user_preference_game_esrb_rating_Mature', 'user_preference_game_esrb_rating_Everyone 10+', 'user_preference_game_esrb_rating_Teen', 'user_preference_game_esrb_rating_Everyone', 'user_preference_game_esrb_rating_Adults Only', 'user_preference_game_genres_Action', 'user_preference_game_genres_Adventure', 'user_preference_game_genres_Arcade', 'user_preference_game_genres_Board Games', 'user_preference_game_genres_Card', 'user_preference_game_genres_Casual', 'user_preference_game_genres_Educational', 'user_preference_game_genres_Family', 'user_preference_game_genres_Fighting', 'user_preference_game_genres_Indie', 'user_preference_game_genres_Massively Multiplayer', 'user_preference_game_genres_Platformer', 'user_preference_game_genres_Puzzle', 'user_preference_game_genres_RPG', 'user_preference_game_genres_Racing', 'user_preference_game_genres_Shooter', 'user_preference_game_genres_Simulation', 'user_preference_game_genres_Sports', 'user_preference_game_genres_Strategy', 'user_preference_game_platforms_3DO', 'user_preference_game_platforms_Android', 'user_preference_game_platforms_Apple Macintosh', 'user_preference_game_platforms_Atari', 'user_preference_game_platforms_Commodore / Amiga', 'user_preference_game_platforms_Linux', 'user_preference_game_platforms_Neo Geo', 'user_preference_game_platforms_Nintendo', 'user_preference_game_platforms_PlayStation', 'user_preference_game_platforms_SEGA', 'user_preference_game_platforms_Web', 'user_preference_game_platforms_Xbox', 'user_preference_game_platforms_iOS', 'game_released_year_since_1984.0']
    bool_cols = ['user_has_coordinates', 'game_tba', 'game_current_discount', 'game_esrb_rating_Rating Pending', 'game_esrb_rating_Missing', 'game_esrb_rating_Mature', 'game_esrb_rating_Everyone 10+', 'game_esrb_rating_Teen', 'game_esrb_rating_Everyone', 'game_esrb_rating_Adults Only', 'game_genres_Action', 'game_genres_Adventure', 'game_genres_Arcade', 'game_genres_Board Games', 'game_genres_Card', 'game_genres_Casual', 'game_genres_Educational', 'game_genres_Family', 'game_genres_Fighting', 'game_genres_Indie', 'game_genres_Massively Multiplayer', 'game_genres_Platformer', 'game_genres_Puzzle', 'game_genres_RPG', 'game_genres_Racing', 'game_genres_Shooter', 'game_genres_Simulation', 'game_genres_Sports', 'game_genres_Strategy', 'game_platforms_3DO', 'game_platforms_Android', 'game_platforms_Apple Macintosh', 'game_platforms_Atari', 'game_platforms_Commodore / Amiga', 'game_platforms_Linux', 'game_platforms_Neo Geo', 'game_platforms_Nintendo', 'game_platforms_PC', 'game_platforms_PlayStation', 'game_platforms_SEGA', 'game_platforms_Web', 'game_platforms_Xbox', 'game_platforms_iOS']
    cf_emb_cols = ['user_emb', 'game_emb']
    
    # Process the user to get the final user DataFrame.
    user_df = preprocess_new_user(user, train_df, game_df, score_raw_df, model_CF, user_country_encoder)

    # Pair every games in training set that are not yet played by the user
    paired_df = pair_non_played_games(user, user_df, game_df)

    # Make prediction using trained NN model
    predictions = NN_forward_pass(paired_df, train_df, single_cat_cols, multi_cat_cols, num_cols, bool_cols, cf_emb_cols)

    # Attach predictions to the paired DataFrame
    paired_df['prediction'] = predictions.squeeze().detach().cpu().numpy()
    
    # Sort descending by prediction score
    ranked_df = paired_df.sort_values(by='prediction', ascending=False)
    
    print("Top 10 games based on predicted score:")
    print(ranked_df[['app_id', 'game_name', 'prediction']].head(10))

if __name__ == '__main__':
    main()

    #cProfile.run('main()', 'profile_output.prof')
    #stats = pstats.Stats('profile_output.prof')
    #stats.sort_stats('tottime').print_stats('prediction_pipeline.py')