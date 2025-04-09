# --- START OF FILE app.py ---

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import io
import base64
import fsspec
import pyarrow.dataset as ds # Needed for load_data_filtered
import pyarrow as pa       # Potentially needed
import gc                  # For garbage collection
import gdown

# --- Function Definitions ---

@st.cache_data(show_spinner="Loading data...")
def load_data_filtered(data_path: str, league: str, season_internal: str, columns=None):
    """Loads data filtered by league and season directly from the source."""
    # Use season_internal which should match the Parquet data format (e.g., '2324')
    file_id = "/1tKjuZ-bDM7NYe-C8hWmIcCNkdqqt8Hfp"
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        output = "Top_5_Leagues_23_24.parquet"
        gdown.download(url, output, quiet=False)

        df = pd.read_parquet(output, columns=columns)
        df = df[(df["league"] == league) & (df["season"] == season_internal)]

        return df

    except Exception as e:
        st.error(f"Failed to load data from Google Drive: {e}")
        return pd.DataFrame()

@st.cache_data
def add_carries(_game_df):
    """Adds Carry events based on consecutive actions."""
    if _game_df.empty:
        return _game_df # Return immediately if input is empty

    game_df = _game_df.copy()
    # Ensure required columns exist before proceeding
    required_cols = ['minute', 'second', 'gameId', 'teamId', 'endX', 'endY', 'x', 'y', 'period', 'playerId', 'team', 'player', 'expandedMinute']
    if not all(col in game_df.columns for col in required_cols):
        st.error("Cannot add carries: Input data is missing required columns.")
        return _game_df # Return original data

    game_df['time_seconds'] = game_df['minute']*60+game_df['second']
    next_actions = game_df.shift(-1)

    same_game = game_df.gameId == next_actions.gameId
    same_team = game_df.teamId == next_actions.teamId
    dx = game_df.endX - next_actions.x
    dy = game_df.endY - next_actions.y
    far_enough = (dx**2 + dy**2) >= 0.0**2
    not_too_far = (dx**2 + dy**2) <= 100.0**2
    dt = next_actions.time_seconds - game_df.time_seconds
    same_phase = dt < 20.0
    same_period = game_df.period == next_actions.period

    dribble_idx = same_team & far_enough & not_too_far & same_phase & same_period & same_game
    valid_dribble_idx = dribble_idx.fillna(False)

    if not valid_dribble_idx.any():
         return game_df # No carries found

    prev = game_df.loc[valid_dribble_idx] # Use .loc for boolean indexing safety
    nex = next_actions.loc[valid_dribble_idx]

    if nex.empty:
        return game_df

    dribbles = pd.DataFrame({
        'gameId': nex['gameId'].values,
        'period': nex['period'].values,
        'expandedMinute': nex['expandedMinute'].values,
        'passKey': False,
        'assist': False,
        'isTouch': True,
        'playerId': nex['playerId'].values,
        'team': nex['team'].values,
        'player': nex['player'].values,
        'time_seconds': (prev['time_seconds'].values + nex['time_seconds'].values) / 2,
        'teamId': nex['teamId'].values,
        'x': prev['endX'].values,
        'y': prev['endY'].values,
        'endX': nex['x'].values,
        'endY': nex['y'].values,
        'type': 'Carry',
        'outcomeType': 'Successful'
    })

    game_df_with_carries = pd.concat([game_df, dribbles], ignore_index=True, sort=False)
    # Drop temporary time_seconds if not needed later, or recalculate minute/second
    # For sorting, time_seconds is reliable
    game_df_with_carries = game_df_with_carries.sort_values(
        ['gameId', 'period', 'time_seconds']
    ).reset_index(drop=True)
    # Optional: Re-calculate minute/second from time_seconds if needed downstream
    # game_df_with_carries['minute'] = (game_df_with_carries['time_seconds'] // 60).astype(int)
    # game_df_with_carries['second'] = (game_df_with_carries['time_seconds'] % 60).astype(int)
    game_df_with_carries['action_id'] = game_df_with_carries.index # Simpler action_id

    return game_df_with_carries

@st.cache_data
def prepare_data(data):
    """Scales coordinates and identifies half-space actions."""
    if data.empty:
        return pd.DataFrame(), pd.DataFrame() # Return empty dfs

    data = data.copy()
    # Check for coordinate columns
    coord_cols = ['x', 'y', 'endX', 'endY']
    if not all(col in data.columns for col in coord_cols):
        st.error("Cannot prepare data: Coordinate columns (x, y, endX, endY) missing.")
        return pd.DataFrame(), pd.DataFrame()

    data['x'] = data['x']*1.2
    data['y'] = data['y']*.8
    data['endX'] = data['endX']*1.2
    data['endY'] = data['endY']*.8

    left_halfspace_x_min, left_halfspace_x_max = 60, 102
    left_halfspace_y_min, left_halfspace_y_max = 50, 62
    right_halfspace_x_min, right_halfspace_x_max = 60, 102
    right_halfspace_y_min, right_halfspace_y_max = 18, 30

    data_carries = data[data['type'] == 'Carry'].copy()
    data_passes = data[(data['type'] == 'Pass') & (data['outcomeType'] == 'Successful')].copy()

    # Flag actions IN half-spaces
    for df in [data_passes, data_carries]:
        if not df.empty:
            df['in_rhs'] = (df['x'].between(right_halfspace_x_min, right_halfspace_x_max) &
                            df['y'].between(right_halfspace_y_min, right_halfspace_y_max))
            df['in_lhs'] = (df['x'].between(left_halfspace_x_min, left_halfspace_x_max) &
                            df['y'].between(left_halfspace_y_min, left_halfspace_y_max))
            # Initialize 'into' columns as False - keep if needed, remove otherwise
            df['into_rhs'] = False
            df['into_lhs'] = False

    return data_passes, data_carries

@st.cache_data
def calculate_progressive_actions(df):
    """Identifies progressive actions based on distance to goal center."""
    if df.empty or not all(col in df.columns for col in ['x', 'y', 'endX', 'endY']):
        return pd.DataFrame()

    df_prog = df.copy()
    goal_x, goal_y = 120, 40

    df_prog['beginning'] = np.sqrt(np.square(goal_x - df_prog['x']) + np.square(goal_y - df_prog['y']))
    df_prog['end'] = np.sqrt(np.square(goal_x - df_prog['endX']) + np.square(goal_y - df_prog['endY']))

    # Calculate progression ratio safely
    df_prog['progressive'] = np.where(
        (df_prog['beginning'].notna()) & (df_prog['beginning'] > 1), # Avoid division by zero/small numbers
        (df_prog['end'] / df_prog['beginning']) < 0.75,
        False
    )
    # Consider adding other progressive criteria if needed (e.g., end in opp half)
    # df_prog['progressive'] = df_prog['progressive'] & (df_prog['endX'] >= 60)

    return df_prog[df_prog['progressive']].copy()


@st.cache_data
def process_halfspace_data(data_passes, data_carries, mins_data):
    """Groups actions, merges, and calculates p90 stats."""
    # --- Calculate Progressive Actions ---
    prog_rhs_passes = calculate_progressive_actions(data_passes[data_passes['in_rhs']])
    prog_lhs_passes = calculate_progressive_actions(data_passes[data_passes['in_lhs']])
    prog_rhs_carries = calculate_progressive_actions(data_carries[data_carries['in_rhs']])
    prog_lhs_carries = calculate_progressive_actions(data_carries[data_carries['in_lhs']])

    # --- Group by Player ---
    group_cols = ['playerId', 'player', 'team']
    def safe_group(df, name):
        # Check if grouping columns exist before grouping
        if df.empty or not all(col in df.columns for col in group_cols):
             return pd.DataFrame(columns=group_cols + [name])
        try:
            return df.groupby(group_cols).size().reset_index(name=name)
        except KeyError as e:
             st.warning(f"Grouping error for '{name}': Missing column {e}. Returning empty group.")
             return pd.DataFrame(columns=group_cols + [name])

    prog_rhs_passes_grouped = safe_group(prog_rhs_passes, 'prog_rhs_passes')
    prog_lhs_passes_grouped = safe_group(prog_lhs_passes, 'prog_lhs_passes')
    prog_rhs_carries_grouped = safe_group(prog_rhs_carries, 'prog_rhs_carries')
    prog_lhs_carries_grouped = safe_group(prog_lhs_carries, 'prog_lhs_carries')

    # --- Merge Grouped Data (Targeted fillna) ---
    def merge_and_fill_counts(df1, df2, on_cols, count_cols_df1, count_cols_df2):
        # Check keys exist before merge
        valid_merge = True
        for df in [df1, df2]:
            if not all(col in df.columns for col in on_cols):
                valid_merge = False
                break
        if not valid_merge:
             st.warning(f"Cannot merge: Missing key columns {on_cols} in one of the dataframes.")
             # Decide how to handle: return df1? return empty? Here we return df1 for partial result.
             # Ensure df1 has the expected count columns, filled with 0 if they should exist
             for col in count_cols_df1:
                 if col not in df1.columns: df1[col] = 0
             return df1

        try:
            merged_df = pd.merge(df1, df2, on=on_cols, how='outer')
        except Exception as merge_error:
             st.error(f"Error during merge: {merge_error}")
             # Return df1 as fallback, ensure its count cols exist
             for col in count_cols_df1:
                 if col not in df1.columns: df1[col] = 0
             return df1

        all_count_cols = count_cols_df1 + count_cols_df2
        for col in all_count_cols:
            if col not in merged_df.columns:
                merged_df[col] = 0
            else:
                 if pd.api.types.is_numeric_dtype(merged_df[col]):
                    # Ensure NaNs introduced by outer merge are filled
                    merged_df[col] = merged_df[col].fillna(0).astype(int)
                 # else: # Column not numeric, do not fill with 0
                     # This case shouldn't happen for count columns if grouping was correct
        return merged_df

    # Merge RHS
    rhs_count_cols_p = ['prog_rhs_passes']
    rhs_count_cols_c = ['prog_rhs_carries']
    prog_result_df_rhs = merge_and_fill_counts(prog_rhs_passes_grouped, prog_rhs_carries_grouped, group_cols, rhs_count_cols_p, rhs_count_cols_c)
    if 'prog_rhs_passes' not in prog_result_df_rhs: prog_result_df_rhs['prog_rhs_passes'] = 0 # Ensure columns exist after potential failed merge
    if 'prog_rhs_carries' not in prog_result_df_rhs: prog_result_df_rhs['prog_rhs_carries'] = 0
    prog_result_df_rhs['prog_rhs_actions'] = prog_result_df_rhs['prog_rhs_passes'] + prog_result_df_rhs['prog_rhs_carries']

    # Merge LHS
    lhs_count_cols_p = ['prog_lhs_passes']
    lhs_count_cols_c = ['prog_lhs_carries']
    prog_result_df_lhs = merge_and_fill_counts(prog_lhs_passes_grouped, prog_lhs_carries_grouped, group_cols, lhs_count_cols_p, lhs_count_cols_c)
    if 'prog_lhs_passes' not in prog_result_df_lhs: prog_result_df_lhs['prog_lhs_passes'] = 0
    if 'prog_lhs_carries' not in prog_result_df_lhs: prog_result_df_lhs['prog_lhs_carries'] = 0
    prog_result_df_lhs['prog_lhs_actions'] = prog_result_df_lhs['prog_lhs_passes'] + prog_result_df_lhs['prog_lhs_carries']

    # Combine RHS and LHS
    rhs_action_cols = ['prog_rhs_passes', 'prog_rhs_carries', 'prog_rhs_actions']
    lhs_action_cols = ['prog_lhs_passes', 'prog_lhs_carries', 'prog_lhs_actions']
    combined_prog_df = merge_and_fill_counts(prog_result_df_rhs, prog_result_df_lhs, group_cols, rhs_action_cols, lhs_action_cols)
    if 'prog_rhs_actions' not in combined_prog_df: combined_prog_df['prog_rhs_actions'] = 0
    if 'prog_lhs_actions' not in combined_prog_df: combined_prog_df['prog_lhs_actions'] = 0
    combined_prog_df['prog_HS_actions'] = combined_prog_df['prog_rhs_actions'] + combined_prog_df['prog_lhs_actions']

    # --- Merge with Minutes Data ---
    # Initialize p90 cols to avoid errors if merge fails or mins_data is bad
    combined_prog_df['90s'] = np.nan
    combined_prog_df['position'] = 'Unknown'
    combined_prog_df['prog_act_HS_p90'] = 0.0
    combined_prog_df['prog_rhs_act_p90'] = 0.0
    combined_prog_df['prog_lhs_act_p90'] = 0.0

    if not mins_data.empty:
        mins_cols_to_merge = ['player', 'team', '90s', 'position']
        if all(col in mins_data.columns for col in mins_cols_to_merge):
             try:
                # Use a temporary df for merge to avoid modifying original combined_prog_df inplace yet
                merged_with_mins = pd.merge(
                    combined_prog_df[group_cols + rhs_action_cols + lhs_action_cols + ['prog_HS_actions']], # Select only needed cols
                    mins_data[mins_cols_to_merge],
                    on=['player', 'team'],
                    how='left'
                )
                # Assign back columns that were successfully merged
                combined_prog_df['90s'] = merged_with_mins['90s']
                combined_prog_df['position'] = merged_with_mins['position']

                # --- Calculate p90 Metrics --- (Only if merge succeeded and 90s exist)
                if '90s' in combined_prog_df.columns and combined_prog_df['90s'].notna().any():
                    combined_prog_df['prog_act_HS_p90'] = (combined_prog_df['prog_HS_actions'] / combined_prog_df['90s']).replace([np.inf, -np.inf], np.nan).fillna(0)
                    combined_prog_df['prog_rhs_act_p90'] = (combined_prog_df['prog_rhs_actions'] / combined_prog_df['90s']).replace([np.inf, -np.inf], np.nan).fillna(0)
                    combined_prog_df['prog_lhs_act_p90'] = (combined_prog_df['prog_lhs_actions'] / combined_prog_df['90s']).replace([np.inf, -np.inf], np.nan).fillna(0)

             except Exception as merge_error:
                 st.warning(f"Could not merge with minutes data: {merge_error}. p90 stats will be zero.")
        else:
            missing_mins_cols = [col for col in mins_cols_to_merge if col not in mins_data.columns]
            st.warning(f"Minutes data missing columns: {missing_mins_cols}. Cannot calculate p90 stats.")
    else:
        st.warning("Minutes data is empty. Cannot calculate p90 stats.")


    # --- Final Filtering & Cleanup ---
    if 'position' in combined_prog_df.columns:
        # Ensure position column is not all NaN before filtering
        if combined_prog_df['position'].notna().any():
             combined_prog_df = combined_prog_df[combined_prog_df['position'] != 'GK'].copy()

    # Drop duplicates based on most reliable key available
    if 'playerId' in combined_prog_df.columns and combined_prog_df['playerId'].notna().any():
         combined_prog_df = combined_prog_df.drop_duplicates(subset=['playerId'], keep='first')
    elif all(col in combined_prog_df.columns for col in ['player', 'team']):
         combined_prog_df = combined_prog_df.drop_duplicates(subset=['player', 'team'], keep='first')

    return combined_prog_df, prog_rhs_passes, prog_lhs_passes, prog_rhs_carries, prog_lhs_carries

@st.cache_data
def plot_player_halfspace_actions(player_data, player_id, prog_rhs_passes, prog_lhs_passes,
                                   prog_rhs_carries, prog_lhs_carries, action_type):
    """Generates the pitch plot for a selected player - REVERTED APPEARANCE."""
    fig, ax = plt.subplots(figsize=(15, 10), facecolor='#1e1e1e')

    pitch = Pitch(pitch_type='statsbomb', pitch_color='#1e1e1e', line_color='#FFFFFF', line_zorder=2)
    pitch.draw(ax=ax)
    ax.invert_yaxis() # StatsBomb pitch vertical orientation

    # Watermark - REVERTED
    ax.text(60, 78, '@pranav_m28', fontsize=17, color='white', alpha=0.75, ha='center', va='center', zorder=1)

    # Filter actions for the specific player ID
    player_prog_rhs_passes = prog_rhs_passes[prog_rhs_passes['playerId'] == player_id]
    player_prog_lhs_passes = prog_lhs_passes[prog_lhs_passes['playerId'] == player_id]
    player_prog_rhs_carries = prog_rhs_carries[prog_rhs_carries['playerId'] == player_id]
    player_prog_lhs_carries = prog_lhs_carries[prog_lhs_carries['playerId'] == player_id]

    pass_color = '#24a8ff'
    carry_color = '#FF5959'

    # Plot based on selected action type
    if action_type == "Right Half-Space Actions":
        pitch.lines(player_prog_rhs_passes.x, player_prog_rhs_passes.y, player_prog_rhs_passes.endX, player_prog_rhs_passes.endY,
                    lw=3, transparent=True, comet=True, color=pass_color, ax=ax, alpha=1)
        pitch.scatter(player_prog_rhs_passes.endX, player_prog_rhs_passes.endY, s=40, c=pass_color, edgecolors='none', ax=ax, zorder=2, alpha=1)
        pitch.lines(player_prog_rhs_carries.x, player_prog_rhs_carries.y, player_prog_rhs_carries.endX, player_prog_rhs_carries.endY,
                    ls='dashed', lw=3, color=carry_color, ax=ax)
        pitch.scatter(player_prog_rhs_carries.endX, player_prog_rhs_carries.endY, s=40, c=carry_color, edgecolors='none', ax=ax, alpha=1)
        # Title - REVERTED
        title_text = f'{player_data["player"]} - Right Half-Space Progressive Actions\nRight Half-Space Actions p90: {player_data["prog_rhs_act_p90"]:.2f}'

    elif action_type == "Left Half-Space Actions":
        pitch.lines(player_prog_lhs_passes.x, player_prog_lhs_passes.y, player_prog_lhs_passes.endX, player_prog_lhs_passes.endY,
                    lw=3, transparent=True, comet=True, color=pass_color, ax=ax, alpha=1)
        pitch.scatter(player_prog_lhs_passes.endX, player_prog_lhs_passes.endY, s=40, c=pass_color, edgecolors='none', ax=ax, zorder=2, alpha=1)
        pitch.lines(player_prog_lhs_carries.x, player_prog_lhs_carries.y, player_prog_lhs_carries.endX, player_prog_lhs_carries.endY,
                    ls='dashed', lw=3, color=carry_color, ax=ax)
        pitch.scatter(player_prog_lhs_carries.endX, player_prog_lhs_carries.endY, s=40, c=carry_color, edgecolors='none', ax=ax, alpha=1)
        # Title - REVERTED
        title_text = f'{player_data["player"]} - Left Half-Space Progressive Actions\nLeft Half-Space Actions p90: {player_data["prog_lhs_act_p90"]:.2f}'

    else: # "All Half-Space Actions"
        # Plot RHS
        pitch.lines(player_prog_rhs_passes.x, player_prog_rhs_passes.y, player_prog_rhs_passes.endX, player_prog_rhs_passes.endY,
                    lw=3, transparent=True, comet=True, color=pass_color, ax=ax, alpha=1)
        pitch.scatter(player_prog_rhs_passes.endX, player_prog_rhs_passes.endY, s=40, c=pass_color, edgecolors='none', ax=ax, zorder=2, alpha=1)
        pitch.lines(player_prog_rhs_carries.x, player_prog_rhs_carries.y, player_prog_rhs_carries.endX, player_prog_rhs_carries.endY,
                    ls='dashed', lw=3, color=carry_color, ax=ax)
        pitch.scatter(player_prog_rhs_carries.endX, player_prog_rhs_carries.endY, s=40, c=carry_color, edgecolors='none', ax=ax, alpha=1)
        # Plot LHS
        pitch.lines(player_prog_lhs_passes.x, player_prog_lhs_passes.y, player_prog_lhs_passes.endX, player_prog_lhs_passes.endY,
                    lw=3, transparent=True, comet=True, color=pass_color, ax=ax, alpha=1)
        pitch.scatter(player_prog_lhs_passes.endX, player_prog_lhs_passes.endY, s=40, c=pass_color, edgecolors='none', ax=ax, zorder=2, alpha=1)
        pitch.lines(player_prog_lhs_carries.x, player_prog_lhs_carries.y, player_prog_lhs_carries.endX, player_prog_lhs_carries.endY,
                    ls='dashed', lw=3, color=carry_color, ax=ax)
        pitch.scatter(player_prog_lhs_carries.endX, player_prog_lhs_carries.endY, s=40, c=carry_color, edgecolors='none', ax=ax, alpha=1)
        # Title - REVERTED
        title_text = f'{player_data["player"]} - Half-Space Progressive Actions\nTotal Half-Space Actions p90: {player_data["prog_act_HS_p90"]:.2f}'

    # Apply title using Axes method for better control - REVERTED POSITIONING similar to original
    ax.set_title(title_text, fontsize=24, color='white', fontweight='bold', pad=20) # Use pad for spacing
    ax.text(48, 84, 'Progressive Carries', fontsize=16, color=carry_color, ha='center', va='center', fontweight='bold')
    ax.text(74, 84, 'Progressive Passes', fontsize=16, color=pass_color, ha='center', va='center', fontweight='bold')
    # Save plot to buffer
    buffer = io.BytesIO()
    plt.tight_layout() # Adjust layout automatically
    plt.savefig(buffer, format='png', facecolor='#1e1e1e', edgecolor='none', dpi=300)
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close(fig)

    return plot_data


# --- Main Application Logic ---
def main():
    st.set_page_config(page_title="Half-Spaces Progressive Actions", layout="wide")

    # --- Configuration ---
    hf_url = "Top_5_Leagues_23_24.parquet"
    mins_csv_path = "T5 Leagues Mins 23-24.csv"
    required_event_columns = [
        "league", "season", "gameId", "period", "minute", "second", "expandedMinute",
        "type", "outcomeType", "teamId", "team", "playerId", "player",
        "x", "y", "endX", "endY"
    ]
    # League and Season Definitions
    leagues = ['ESP-La Liga', 'ENG-Premier League', 'ITA-Serie A', 'GER-Bundesliga', 'FRA-Ligue 1'] # Match Parquet 'league' column
    # Season Mapping: Display Value -> Internal Value (for loading)
    season_mapping = {
        "2023/2024": 2324,
        # Add more seasons here if needed, e.g., "2022/2023": "2223"
    }
    season_display_options = list(season_mapping.keys())


    st.title("Top 5 Leagues Half-Spaces Progressive Actions")
    st.sidebar.header("Filters")

    # --- Sidebar Widgets ---
    # Season Selection with Mapping
    selected_season_display = st.sidebar.selectbox("Select Season", season_display_options)
    selected_season_internal = season_mapping[selected_season_display] # Get internal value for loading

    selected_league = st.sidebar.selectbox("Select League", leagues)

    # --- Cache Clearing Logic ---
    if 'previous_league' not in st.session_state:
        st.session_state.previous_league = selected_league # Initialize
        st.session_state.previous_season = selected_season_internal # Also track season changes

    # Clear if EITHER league OR season changes
    if (selected_league != st.session_state.previous_league or
        selected_season_internal != st.session_state.previous_season):
        # st.warning(f"Filters changed. Clearing processing caches...") # Simplified message
        try:
            add_carries.clear()
            prepare_data.clear()
            calculate_progressive_actions.clear()
            process_halfspace_data.clear()
            plot_player_halfspace_actions.clear()
            # No success message needed, warning implies action
        except Exception as e:
            st.error(f"Error clearing caches: {e}") # Keep error for debugging
        # Update session state
        st.session_state.previous_league = selected_league
        st.session_state.previous_season = selected_season_internal
    # --- End Cache Clearing ---

    # --- Data Loading ---
    # Pass the INTERNAL season value to the loading function
    data = load_data_filtered(hf_url, selected_league, selected_season_internal, columns=required_event_columns)
    if data.empty:
        # Warning was shown in load_data_filtered, maybe add specific guidance
        st.info("No event data found for the selected filters. Try different selections.")
        st.stop() # Stop if essential data is missing

    # Load Minutes Data
    try:
        mins_data_full = pd.read_csv(mins_csv_path)
        # Essential preprocessing
        if 'Mins' not in mins_data_full.columns or 'player' not in mins_data_full.columns or 'team' not in mins_data_full.columns:
             st.error("Minutes data file is missing required columns ('Mins', 'player', 'team'). Cannot proceed.")
             st.stop()
        if '90s' not in mins_data_full.columns:
            mins_data_full['90s'] = mins_data_full['Mins'] / 90.0
        if 'position' not in mins_data_full.columns:
            mins_data_full['position'] = 'Unknown'
    except FileNotFoundError:
        st.error(f"CRITICAL ERROR: Minutes data file not found at '{mins_csv_path}'. Place the file correctly.")
        st.stop()
    except Exception as e:
        st.error(f"CRITICAL ERROR loading or processing minutes data: {e}")
        st.stop()

    # --- Team Selection ---
    league_teams = sorted(data['team'].unique()) if not data.empty else []
    selected_teams = st.sidebar.multiselect("Select Teams", league_teams, default=league_teams)
    if not selected_teams:
        st.warning("No teams selected. Please select at least one team.")
        st.stop() # Stop if no teams are selected

    # --- Filter by Team ---
    filtered_data = data[data['team'].isin(selected_teams)].copy()
    filtered_mins_data = mins_data_full[mins_data_full['team'].isin(selected_teams)].copy()

    # --- Memory Management ---
    del data, mins_data_full
    gc.collect()

    # --- Check Filtered Data ---
    if filtered_data.empty:
         st.warning("No event data found for the selected teams after filtering.")
         st.stop() # Stop if no data to process
    # No need to stop if mins data is empty, process_halfspace_data handles it

    # --- Minimum 90s Slider - CORRECTED MAX VALUE ---
    min_90s_value = 0.0
    max_90s_value = 38.0 # Set fixed maximum based on league games
    default_90s_value = 15.0
    # Adjust default if max available is lower than 15
    if not filtered_mins_data.empty:
         max_available = filtered_mins_data['90s'].max()
         if max_available < default_90s_value:
             default_90s_value = max(min_90s_value, max_available) # Ensure default is not negative

    selected_90s = st.sidebar.slider(
        "Minimum 90s Played",
        min_value=min_90s_value,
        max_value=max_90s_value, # Use the fixed max
        value=default_90s_value,
        step=0.5
    )

    # --- Core Data Processing Pipeline ---
    # No user info messages needed here, runs fairly quickly now
    filtered_data_with_carries = add_carries(filtered_data)
    del filtered_data; gc.collect()

    data_passes, data_carries = prepare_data(filtered_data_with_carries)
    del filtered_data_with_carries; gc.collect()

    combined_prog_df, prog_rhs_passes, prog_lhs_passes, prog_rhs_carries, prog_lhs_carries = process_halfspace_data(
        data_passes, data_carries, filtered_mins_data
    )
    del data_passes, data_carries, filtered_mins_data; gc.collect()

    # --- Final Filtering (Post-Processing) ---
    if not combined_prog_df.empty:
        final_df = combined_prog_df[
            (combined_prog_df['90s'] >= selected_90s) &
            (combined_prog_df['90s'].notna())
        ].copy()
    else:
        final_df = pd.DataFrame()

    # --- Display Results ---
    action_type = st.sidebar.radio("Action Type", ["All Half-Space Actions", "Right Half-Space Actions", "Left Half-Space Actions"])

    st.subheader("Half-Space Progressive Actions per 90")
    if not final_df.empty:
        sort_col_map = {
            "All Half-Space Actions": "prog_act_HS_p90",
            "Right Half-Space Actions": "prog_rhs_act_p90",
            "Left Half-Space Actions": "prog_lhs_act_p90"
        }
        sort_col = sort_col_map.get(action_type, "prog_act_HS_p90")

        display_columns = ['player', 'team', 'position', '90s', 'prog_act_HS_p90', 'prog_rhs_act_p90', 'prog_lhs_act_p90']
        display_columns = [col for col in display_columns if col in final_df.columns] # Ensure cols exist

        # Check if sort column exists before sorting
        if sort_col in final_df.columns:
            sorted_df = final_df.sort_values(sort_col, ascending=False)
        else:
            st.warning(f"Sorting column '{sort_col}' not found. Displaying unsorted.")
            sorted_df = final_df # Display unsorted

        st.dataframe(sorted_df[display_columns].round(2), use_container_width=True)

        # --- Player Visualization ---
        st.subheader("Player Actions Visualization")
        player_list = sorted_df['player'].unique()
        if len(player_list) > 0:
            selected_player = st.selectbox("Select a Player for Visualization", player_list)

            if selected_player:
                # Find the row for the selected player in the sorted DataFrame
                player_data_row = sorted_df[sorted_df['player'] == selected_player]
                if not player_data_row.empty:
                    player_data = player_data_row.iloc[0]
                    player_id = player_data.get('playerId', None) # Safely get playerId

                    if player_id is not None:
                        # Generate and display plot
                        try:
                            plot_data = plot_player_halfspace_actions(
                                player_data, player_id,
                                prog_rhs_passes, prog_lhs_passes,
                                prog_rhs_carries, prog_lhs_carries,
                                action_type
                            )
                            st.image(f"data:image/png;base64,{plot_data}")
                        except Exception as plot_error:
                             st.error(f"Could not generate plot for {selected_player}: {plot_error}")
                    else:
                        st.error(f"Player ID not found for {selected_player}. Cannot generate plot.")
                # else: No need for error here, selectbox handles selection
        else:
            st.info("No players meet the current criteria (incl. minimum 90s) for visualization.")

    else:
        st.warning("No player data available after applying all filters. Adjust filters (e.g., Minimum 90s Played).")

    # --- Sidebar Social Links ---
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Connect with me")
        st.markdown("- 🐦 [Twitter](https://twitter.com/pranav_m28)")
        st.markdown("- 🔗 [GitHub](https://github.com/pranavm28)")
        st.markdown("- ❤️ [BuyMeACoffee](https://buymeacoffee.com/pranav_m28)") # Corrected link text

# --- Run the App ---
if __name__ == "__main__":
    main()

# --- END OF FILE app.py ---