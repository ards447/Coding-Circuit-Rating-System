import pandas as pd
import numpy as np
import ast
import os

## Setup and file loading

# Create output directory if it doesn't exist
output_dir = './output'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
# Read event weights and difficulty levels
events_df = pd.read_csv('./input/event_weights.csv')
event_info = events_df.set_index('Event Number').to_dict(orient='index')

# Read participant details and parse the list of events participated in
participants_df = pd.read_csv('./input/Participant_details.csv')
participants_df['Events Participated'] = participants_df['Events Participated'].apply(ast.literal_eval)

## Initialising participant data

participant_data = {}
event_participants = {}

# Initialize participant data with default rating and participation details
for _, row in participants_df.iterrows():
    participant = row['Participant Number']
    events = row['Events Participated']
    participant_data[participant] = {
        'rating': 1000,
        'events_participated': 0,
        'consicutive_misses': 0,
        'events': events
    }

# Build a mapping of events to list of participants in each event
    for event in events:
        event_participants.setdefault(event, []).append(participant)

## Helper Functions

# Compute expected number of problems solved based on rating and difficulty
def compute_mu_R(R, R_mid):
    mu_R = 6 / (1 + 10 ** ((R_mid - R) / 400))
    return np.round(mu_R)

# Convert difficulty level to corresponding mean rating value
def get_R_mid(difficulty):
    if difficulty == 1:     # Easy
        return 1400
    elif difficulty == 2:   # Medium
        return 1500
    elif difficulty == 3:   # Hard
        return 1600

# Generate a simulated number of problems solved based on participant rating
def generate_solved_problems(rating):
    values = [1, 2, 3, 4, 5, 6]

    if rating <= 1300:
        weights = [0.4, 0.3, 0.15, 0.1, 0.04, 0.01]  # More likely to solve fewer problems
    elif rating <= 1500:
        weights = [0.05, 0.1, 0.5, 0.2, 0.1, 0.05]  # More likely to solve 3
    elif rating <= 1800:
        weights = [0.01, 0.04, 0.1, 0.35, 0.35, 0.15]  # More likely to solve 4 or 5
    else:
        weights = [0.01, 0.01, 0.02, 0.05, 0.2, 0.71]  # More likely to solve all problems

    return np.random.choice(values, p=weights)

# Apply decay to rating for consecutive missed events
def apply_decay(rating, missed_events):
    if missed_events <= 1:
        return rating  # No decay for 1 or fewer missed events
    decay = int(rating * (1 - np.exp(-0.1 * (missed_events - 1))))
    return max(1000, rating - decay)  # Rating should not drop below 1000

# Save leaderboard to csv
def save_leaderboard(participant_data, filename):
    leaderboard = pd.DataFrame([
        {'Participant Number': pid, 'Rating': data['rating']}
        for pid, data in participant_data.items()
    ])
    leaderboard = leaderboard.sort_values(by='Rating', ascending=False)
    leaderboard['Rank'] = range(1, len(leaderboard) + 1)
    cols = ['Rank'] + [col for col in leaderboard.columns if col != 'Rank']
    leaderboard = leaderboard[cols]
    leaderboard.to_csv(filename, index=False)

## Process events in chronological order

events = sorted(event_participants.keys())

for event in events:
    participants_in_event = event_participants.get(event, [])
    if event in event_info:
        W = event_info[event]['Event Weightage']
        difficulty = event_info[event]['Event Difficulty']
        R_mid = get_R_mid(difficulty)
        
        # Update rating for each participant in the event
        for participant in participants_in_event:
            R = participant_data[participant]['rating']
            mu_R = compute_mu_R(R, R_mid)
            S = generate_solved_problems(R)
            K = 100 if participant_data[participant]['events_participated'] < 3 else 50  # Higher K for early games
            delta = np.sign(S - mu_R) * np.log(1 + abs(S - mu_R))  # Scaled difference
            del_R = W * K * delta
            participant_data[participant]['rating'] = max(1000, np.round(R + del_R))
            participant_data[participant]['events_participated'] += 1
            participant_data[participant]['consicutive_misses'] = 0  # Reset missed event count

    # Apply rating decay for participants who missed the event
    for participant in participant_data:
        if participant not in participants_in_event:
            R = participant_data[participant]['rating']
            participant_data[participant]['consicutive_misses'] += 1
            participant_data[participant]['rating'] = apply_decay(R, participant_data[participant]['consicutive_misses'])

    # Save leaderboard after each event
    save_leaderboard(participant_data, f'./output/leaderboard_after_{event}.csv')
   
# Generate final leaderboard
save_leaderboard(participant_data, './output/final_leaderboard.csv')