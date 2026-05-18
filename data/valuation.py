import pandas as pd
import numpy as np

BATTER_POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'MI', 'CI', 'OF', 'UTIL']
PITCHER_POSITIONS = ['SP', 'RP', 'P']
TOTAL_ROSTER_LIMIT = 40
VALID_PLAYER_POSITIONS = {
    'C', '1B', '2B', '3B', 'SS', 'OF', 'DH', 'UTIL', 'MI', 'CI', 'SP', 'RP', 'P'
}

ROSTER_SLOTS = {
    'C': 2, '1B': 1, '2B': 1, '3B': 1, 'SS': 1, 'OF': 5, 
    'MI': 1, 'CI': 1, 'UTIL': 1, 'SP': 5, 'RP': 5, 'P': 0,
    'Bench': 16 
}

def eligible_roster_slots(player_positions):
    normalized_positions = [str(pos).strip().upper() for pos in player_positions]
    slots = []

    exact_slots = ['C', '1B', '2B', '3B', 'SS', 'OF', 'SP', 'RP']
    for pos in normalized_positions:
        if pos in exact_slots and pos not in slots:
            slots.append(pos)

    if 'P' in normalized_positions:
        for slot in ['SP', 'RP']:
            if slot not in slots:
                slots.append(slot)

    if any(pos in {'2B', 'SS', 'MI'} for pos in normalized_positions) and 'MI' not in slots:
        slots.append('MI')

    if any(pos in {'1B', '3B', 'CI'} for pos in normalized_positions) and 'CI' not in slots:
        slots.append('CI')

    hitter_positions = {'C', '1B', '2B', '3B', 'SS', 'OF', 'DH', 'UTIL', 'MI', 'CI'}
    if any(pos in hitter_positions for pos in normalized_positions) and 'UTIL' not in slots:
        slots.append('UTIL')

    return slots


def find_open_roster_slot(roster_counts, player_positions):
    for slot in eligible_roster_slots(player_positions):
        if roster_counts.get(slot, 0) < ROSTER_SLOTS.get(slot, 0):
            return slot
    if roster_counts.get('Bench', 0) < ROSTER_SLOTS.get('Bench', 0):
        return 'Bench'
    return None


def assign_player_to_roster_slot(roster_counts, player_positions):
    slot = find_open_roster_slot(roster_counts, player_positions)
    if slot is not None:
        roster_counts[slot] += 1
    return slot


def has_open_starting_slot(roster_counts, player_positions):
    slot = find_open_roster_slot(roster_counts, player_positions)
    return slot is not None and slot != 'Bench'

def clean_prospect_name(name_str):
    """Converts 'Griffin, Konnor' to 'Konnor Griffin'"""
    if ',' in str(name_str):
        parts = str(name_str).split(',')
        return f"{parts[1].strip()} {parts[0].strip()}"
    return name_str


def normalize_player_positions(position_value):
    positions = []
    for pos in str(position_value).split('/'):
        clean_pos = pos.strip().upper()
        if clean_pos in VALID_PLAYER_POSITIONS and clean_pos not in positions:
            positions.append(clean_pos)
    return positions or ['UTIL']


def merge_position_lists(position_lists):
    merged = []
    for positions in position_lists:
        if not isinstance(positions, list):
            positions = normalize_player_positions(positions)
        for pos in positions:
            clean_pos = str(pos).strip().upper()
            if clean_pos in VALID_PLAYER_POSITIONS and clean_pos not in merged:
                merged.append(clean_pos)
    return merged or ['UTIL']


def merge_duplicate_players(df):
    if df.empty:
        return df

    sorted_df = df.sort_values('dollars', ascending=False).copy()
    rows = []
    for _, group in sorted_df.groupby('Name', sort=False):
        row = group.iloc[0].copy()
        row['positions'] = merge_position_lists(group['positions'])
        row['dollars'] = group['dollars'].max()
        if 'is_prospect' in row.index:
            row['is_prospect'] = bool(group['is_prospect'].all())
        if 'active_eligible' in row.index:
            row['active_eligible'] = bool(group['active_eligible'].any())
        if 'prospect_level' in row.index:
            levels = [str(level).strip().upper() for level in group['prospect_level'].dropna() if str(level).strip()]
            row['prospect_level'] = 'MLB' if 'MLB' in levels else (levels[0] if levels else '')
        if 'Dollars' in row.index:
            row['Dollars'] = group.get('Dollars', group['dollars']).max()
        if 'POS' in row.index:
            row['POS'] = '/'.join(row['positions'])
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def load_player_pool(batters_csv, pitchers_csv, prospects_csv=None):
    # 1. Load MLB FanGraphs Projections
    b_df = pd.read_csv(batters_csv)
    p_df = pd.read_csv(pitchers_csv)
    
    pool_df = pd.concat([b_df, p_df], ignore_index=True)
    pool_df = pool_df[pool_df['Name'].notna()].copy()
    pool_df['positions'] = pool_df['POS'].apply(normalize_player_positions)
    pool_df['dollars'] = pool_df['Dollars'].clip(lower=0.2)
    pool_df['Name'] = pool_df['Name'].str.strip()
    pool_df['is_prospect'] = False # Flag for Thompson Sampling
    pool_df['prospect_level'] = 'MLB'
    pool_df['active_eligible'] = True
    
    # 2. Inject the Prospects
    if prospects_csv:
        pros_df = pd.read_csv(prospects_csv)
        
        # Clean Names
        pros_df['Name'] = pros_df['Name'].apply(clean_prospect_name)
        
        # Create positions list
        pros_df['positions'] = pros_df['Pos'].apply(normalize_player_positions)
        pros_df['prospect_level'] = pros_df.get('Highest Level', '').fillna('').astype(str).str.upper()
        
        # Create a "Dynasty Value" Curve based on Composite Rank
        # e.g., Rank 1 = $25, Rank 10 = $15, Rank 50 = $3, Rank 100+ = $1
        def assign_prospect_value(rank):
            if pd.isna(rank): return 1.0
            if rank <= 5: return 25.0
            if rank <= 20: return 15.0
            if rank <= 50: return 8.0
            if rank <= 100: return 3.0
            return 1.0
            
        pros_df['dollars'] = pros_df['Rank'].apply(assign_prospect_value)
        pros_df['is_prospect'] = True
        pros_df['active_eligible'] = False
        
        # Only keep the columns we need to merge
        pros_slim = pros_df[['Name', 'dollars', 'positions', 'is_prospect', 'prospect_level', 'active_eligible']]
        
        # 3. Merge and Deduplicate
        # If a prospect is already in the FanGraphs CSV (e.g. Paul Skenes), 
        # we keep the FanGraphs row but take the HIGHER of the two values.
        combined = pd.concat([pool_df, pros_slim], ignore_index=True)
        
        pool_df = merge_duplicate_players(combined)
    else:
        pool_df = merge_duplicate_players(pool_df)
    
    return pool_df.to_dict('records')

    
