import pandas as pd
import numpy as np
import random

# Load the schedule data
schedule_df = pd.read_csv("entire_schedule.csv")

# Define workers and skills (same as in the model)
Workers = ["Amy", "Bob", "Cathy", "Dan", "Ed", "Fred", "Gu", "Tobi"]
Skills = {
    "Amy": [1, 0, 0],
    "Bob": [0, 1, 0],
    "Cathy": [0, 0, 1],
    "Dan": [1, 0, 0],
    "Ed": [0, 1, 0],
    "Fred": [0, 0, 1],
    "Gu": [1, 0, 0],
    "Tobi": [0, 1, 0]
}

# Define position requirements for the first 6 positions
PositionRequirements = [
    [1, 0, 0],  # Position 1: requires skill 1
    [0, 1, 0],  # Position 2: requires skill 2
    [0, 0, 1],  # Position 3: requires skill 3
    [1, 1, 0],  # Position 4: requires skill 1 and 2
    [0, 1, 1],  # Position 5: requires skill 2 and 3
    [1, 0, 1],  # Position 6: requires skill 1 and 3
]

# Simulation parameters
no_show_probability = 0.1
understaffing_count = 0


# Function to check if a worker meets the position requirements
def meets_requirements(worker, requirements):
    worker_skills = Skills[worker]
    return all(ws >= req for ws, req in zip(worker_skills, requirements))


# Simulate attendance for each shift
for _, row in schedule_df.iterrows():
    shift = row['Shift']
    position = row['Position']
    assigned_workers = str(row['Assigned Workers']).split(', ')

    if "Training" not in position:  # Only consider main positions for understaffing
        position_index = int(position.split()[-1]) - 1  # Get position index
        requirements = PositionRequirements[position_index]

        # Simulate no-show for each assigned worker
        for worker in assigned_workers:
            if worker == 'None':  # Skip if no one was assigned
                understaffing_count += 1
                continue

            # Determine if worker shows up
            if random.random() < no_show_probability:
                # Attempt to find a replacement
                replacement_found = False
                for candidate in Workers:
                    if candidate != worker and meets_requirements(candidate, requirements):
                        replacement_found = True
                        break  # Replacement found, no understaffing here

                if not replacement_found:
                    understaffing_count += 1  # No replacement found, count as understaffing

print(f"Total understaffing events: {understaffing_count}")
