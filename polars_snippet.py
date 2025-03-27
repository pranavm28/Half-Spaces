import streamlit as st
import pandas as pd
import numpy as np
import polars as pl # Import Polars

# Keep other functions (load_data, add_carries, prepare_data, calculate_progressive_actions, plot) using pandas for now

@st.cache_data # Keep caching this function
def process_halfspace_data_pl(data_passes_pd, data_carries_pd, mins_data_pd):
    """
    Polars implementation for processing half-space data.
    Accepts pandas DFs, converts to Polars, processes, converts back to pandas.
    """
    st.write("--- Entering process_halfspace_data_pl (Polars Version) ---")

    # --- Convert Inputs to Polars DataFrames ---
    try:
        # Use lazy loading potentially for initial conversion if inputs are huge
        # pl.from_pandas might eager load, check Polars docs if needed
        pl_passes = pl.from_pandas(data_passes_pd)
        pl_carries = pl.from_pandas(data_carries_pd)
        pl_mins = pl.from_pandas(mins_data_pd)
        st.write("Inputs converted to Polars DFs")
    except Exception as e:
        st.error(f"Error converting pandas to Polars: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Return empty pandas DFs

    # Basic empty checks for Polars DFs
    if pl_passes.height == 0 and pl_carries.height == 0:
        st.warning("Both Polars passes and carries DFs are empty.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if pl_mins.height == 0:
        st.warning("Polars minutes DF is empty.")
        # Decide if you can proceed without minutes or return empty
        # For now, let's assume we need minutes data later.
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


    # --- Define required columns ---
    group_cols = ['playerId', 'player', 'team']
    # Ensure mins data has necessary columns (do this check earlier maybe?)
    if '90s' not in pl_mins.columns and 'Mins' in pl_mins.columns:
         pl_mins = pl_mins.with_columns((pl.col('Mins') / 90.0).alias('90s'))
    elif '90s' not in pl_mins.columns:
         st.error("Polars mins data missing '90s'/'Mins'.")
         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if 'position' not in pl_mins.columns:
         pl_mins = pl_mins.with_columns(pl.lit('Unknown').alias('position'))


    # --- Polars Equivalent of Calculations ---
    st.write("Calculating progressive actions and grouping (Polars)...")

    # Note: calculate_progressive_actions needs to be compatible or rewritten.
    # Assuming it returns a pandas DF for now, convert its results.
    # Or integrate its logic directly using Polars expressions.
    # Let's assume for now we call the pandas version and convert results back.
    # This adds overhead, rewriting calculate_progressive_actions in Polars would be better.

    # --- Placeholder: Integrate Progressive Calculation using Polars ---
    # This logic needs to replace the call to the external function.
    # Example: Adding a 'progressive' column directly
    goal_center_x = 120
    goal_center_y = 40
    min_prog_ratio = 0.75

    def calculate_prog_polars(df):
        if df.height == 0: return df.with_columns(pl.lit(False).alias('progressive'))
        return df.with_columns([
            (((goal_center_x - pl.col('x'))**2 + (goal_center_y - pl.col('y'))**2)**0.5).alias('beginning'),
            (((goal_center_x - pl.col('endX'))**2 + (goal_center_y - pl.col('endY'))**2)**0.5).alias('end')
        ]).filter(pl.col('beginning') > 0 # Avoid division by zero
        ).with_columns(
           ((pl.col('end') / pl.col('beginning')) < min_prog_ratio).alias('progressive')
        ).filter(pl.col('progressive')) # Keep only progressive

    # Apply calculation and filtering
    pl_prog_rhs_passes = calculate_prog_polars(pl_passes.filter(pl.col('in_rhs')))
    pl_prog_lhs_passes = calculate_prog_polars(pl_passes.filter(pl.col('in_lhs')))
    pl_prog_rhs_carries = calculate_prog_polars(pl_carries.filter(pl.col('in_rhs')))
    pl_prog_lhs_carries = calculate_prog_polars(pl_carries.filter(pl.col('in_lhs')))


    # --- Polars GroupBy ---
    prog_rhs_passes_grouped = pl_prog_rhs_passes.group_by(group_cols).agg(pl.count().alias('prog_rhs_passes')) if pl_prog_rhs_passes.height > 0 else pl.DataFrame({**{c:[] for c in group_cols}, 'prog_rhs_passes':[]})
    prog_lhs_passes_grouped = pl_prog_lhs_passes.group_by(group_cols).agg(pl.count().alias('prog_lhs_passes')) if pl_prog_lhs_passes.height > 0 else pl.DataFrame({**{c:[] for c in group_cols}, 'prog_lhs_passes':[]})
    prog_rhs_carries_grouped = pl_prog_rhs_carries.group_by(group_cols).agg(pl.count().alias('prog_rhs_carries')) if pl_prog_rhs_carries.height > 0 else pl.DataFrame({**{c:[] for c in group_cols}, 'prog_rhs_carries':[]})
    prog_lhs_carries_grouped = pl_prog_lhs_carries.group_by(group_cols).agg(pl.count().alias('prog_lhs_carries')) if pl_prog_lhs_carries.height > 0 else pl.DataFrame({**{c:[] for c in group_cols}, 'prog_lhs_carries':[]})
    st.write("Grouping complete (Polars)")


    # --- Polars Joins (Merge Equivalents) ---
    st.write("Joining grouped counts (Polars)...")

    # Create base of unique players using concat and unique
    all_players_base = pl.concat([
        prog_rhs_passes_grouped.select(group_cols),
        prog_lhs_passes_grouped.select(group_cols),
        prog_rhs_carries_grouped.select(group_cols),
        prog_lhs_carries_grouped.select(group_cols)
    ]).unique(subset=group_cols)
    st.write(f"Shape all_players_base (Polars): {all_players_base.shape}")

    if all_players_base.height == 0:
         st.warning("No unique players found after grouping (Polars).")
         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Perform joins - Polars joins are fast. Use 'outer' join strategy.
    # Start with base and join each grouped df. Use coalesce to fill nulls with 0.
    final_pl_df = all_players_base
    final_pl_df = final_pl_df.join(prog_rhs_passes_grouped, on=group_cols, how='left').with_columns(pl.col('prog_rhs_passes').fill_null(0))
    final_pl_df = final_pl_df.join(prog_lhs_passes_grouped, on=group_cols, how='left').with_columns(pl.col('prog_lhs_passes').fill_null(0))
    final_pl_df = final_pl_df.join(prog_rhs_carries_grouped, on=group_cols, how='left').with_columns(pl.col('prog_rhs_carries').fill_null(0))
    final_pl_df = final_pl_df.join(prog_lhs_carries_grouped, on=group_cols, how='left').with_columns(pl.col('prog_lhs_carries').fill_null(0))
    st.write(f"Shape after joining counts (Polars): {final_pl_df.shape}")


    # Calculate action totals
    final_pl_df = final_pl_df.with_columns([
        (pl.col('prog_rhs_passes') + pl.col('prog_rhs_carries')).alias('prog_rhs_actions'),
        (pl.col('prog_lhs_passes') + pl.col('prog_lhs_carries')).alias('prog_lhs_actions'),
    ]).with_columns(
        (pl.col('prog_rhs_actions') + pl.col('prog_lhs_actions')).alias('prog_HS_actions')
    )

    # --- Join with Minutes Data (Polars) ---
    st.write("Joining with minutes data (Polars)...")
    try:
        # Ensure join keys are compatible type (Polars is stricter than pandas sometimes)
        # Assuming they are strings/categories already based on earlier steps
        mins_join_cols = ['player', 'team', '90s', 'position']
        mins_join_cols = [c for c in mins_join_cols if c in pl_mins.columns] # Re-check columns exist

        final_pl_df = final_pl_df.join(
            pl_mins.select(mins_join_cols), # Select only needed cols
            on=['player', 'team'],
            how='left' # Keep all players from event data
        )
        st.write(f"Shape after minutes join (Polars): {final_pl_df.shape}")
        # Check for nulls introduced in '90s'
        null_90s = final_pl_df.filter(pl.col('90s').is_null()).height
        st.write(f"Null count in '90s' after Polars join: {null_90s}")

    except Exception as e:
         st.error(f"Error during Polars join with minutes data: {e}")
         # Decide how to return - maybe convert current state back to pandas?
         return final_pl_df.to_pandas(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


    # --- Calculate p90 Metrics (Polars) ---
    st.write("Calculating p90 metrics (Polars)...")
    if '90s' in final_pl_df.columns:
        final_pl_df = final_pl_df.with_columns([
            pl.when(pl.col("90s").is_not_null() & (pl.col("90s") > 0))
              .then(pl.col("prog_HS_actions") / pl.col("90s"))
              .otherwise(0.0) # Handle null or zero 90s
              .alias("prog_act_HS_p90"),
            pl.when(pl.col("90s").is_not_null() & (pl.col("90s") > 0))
              .then(pl.col("prog_rhs_actions") / pl.col("90s"))
              .otherwise(0.0)
              .alias("prog_rhs_act_p90"),
            pl.when(pl.col("90s").is_not_null() & (pl.col("90s") > 0))
              .then(pl.col("prog_lhs_actions") / pl.col("90s"))
              .otherwise(0.0)
              .alias("prog_lhs_act_p90")
        ])
    else:
        st.warning("Cannot calculate p90 (Polars) - '90s' column missing.")
        # Add placeholder columns if needed
        final_pl_df = final_pl_df.with_columns([
            pl.lit(0.0).alias("prog_act_HS_p90"),
            pl.lit(0.0).alias("prog_rhs_act_p90"),
            pl.lit(0.0).alias("prog_lhs_act_p90")
        ])

    # --- Drop Duplicates (Polars) ---
    # Polars 'unique' keeps first by default
    final_pl_df = final_pl_df.unique(subset=['player', 'team'], keep='first', maintain_order=True)
    st.write(f"Shape after unique filter (Polars): {final_pl_df.shape}")


    # --- Convert Final Result Back to Pandas ---
    st.write("Converting final result back to pandas...")
    try:
        final_pd_df = final_pl_df.to_pandas()
        st.write(f"--- Exiting process_halfspace_data_pl (Final Pandas Shape: {final_pd_df.shape}) ---")
        # Return pandas DataFrame for compatibility with rest of the app
        # Also need to return the intermediate progressive actions (still pandas DFs if not converted)
        # For simplicity now, return empty pandas DFs for those, or convert them back too if needed.
        return final_pd_df, data_passes_pd, data_carries_pd, pd.DataFrame(), pd.DataFrame() # Adjust return as needed

    except Exception as e:
        st.error(f"Error converting final Polars DF to pandas: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame() # Return empty pandas DFs

# --- In main() ---
# Replace the call to the old function with the new one:
# combined_prog_df, _, _, _, _ = process_halfspace_data_pl( # Call the Polars version
#     data_passes, data_carries, filtered_mins_data
# )
# The rest of main() should work as final_pd_df is a pandas DF
