import sys
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Initialize shifts (14 initial shifts + 18 additional weeks of 7 days each)
Shifts = [f"Mon{d}" if d % 7 == 1 else f"Tue{d}" if d % 7 == 2 else f"Wed{d}" if d % 7 == 3 else
          f"Thu{d}" if d % 7 == 4 else f"Fri{d}" if d % 7 == 5 else f"Sat{d}" if d % 7 == 6 else f"Sun{d}"
          for d in range(1, 127)]

# Workers and initial skill sets (binary: 1 = has skill, 0 = no skill)
Workers = ["Amy", "Bob", "Cathy", "Dan", "Ed", "Fred", "Gu", "Tobi"]
Skills = [
    [1, 0, 0],  # Amy only has skill 1
    [0, 1, 0],  # Bob only has skill 2
    [0, 0, 1],  # Cathy only has skill 3
    [1, 0, 0],  # Dan only has skill 1
    [0, 1, 0],  # Ed only has skill 2
    [0, 0, 1],  # Fred only has skill 3
    [1, 0, 0],  # Gu only has skill 1
    [0, 1, 0],  # Tobi only has skill 2
]

# Position skill requirements (6 main positions, indexed 0-5)
PositionRequirements = [
    [1, 0, 0],  # Position 1: requires skill 1
    [0, 1, 0],  # Position 2: requires skill 2
    [0, 0, 1],  # Position 3: requires skill 3
    [1, 1, 0],  # Position 4: requires skill 1 and 2
    [0, 1, 1],  # Position 5: requires skill 2 and 3
    [1, 0, 1],  # Position 6: requires skill 1 and 3
]

try:
    # Create Gurobi model
    with gp.Model("workforce_with_training_positions") as model:

        # Decision variables: x[w, s, p] for worker w, shift s, position p (including 6 training positions)
        x = model.addVars([(w, s, p) for w in Workers for s in Shifts for p in range(12)], vtype=GRB.BINARY, name="x")

        # Total number of shifts per worker
        totShifts = model.addVars(Workers, name="TotShifts")

        # Binary variables to track if worker has a skill: hasSkill[w, k, s] (worker w, skill k, shift s)
        hasSkill = model.addVars([(w, k, s) for w in Workers for k in range(3) for s in range(len(Shifts))],
                                 vtype=GRB.BINARY, name="hasSkill")

        # Initialize worker skills at the first shift based on input Skills
        for w in Workers:
            for k in range(3):
                model.addConstr(Skills[Workers.index(w)][k] == hasSkill[w, k, 0])

        # Skill acquisition: workers gain skill k after training on corresponding training position (p+6)
        for w in Workers:
            for s in range(1, len(Shifts)):
                for k in range(3):
                    # Training positions are 6-11 (index p+6 represents training positions)
                    model.addConstr(
                        hasSkill[w, k, s] >= hasSkill[w, k, s - 1] + x[w, Shifts[s - 1], k + 6],
                        name=f"skill_acquisition_{w}_{s}_{k}"
                    )

        # Workers can only train on a position if they don't already have the skill
        for w in Workers:
            for s in Shifts:
                for k in range(3):
                    model.addConstr(
                        x[w, s, k + 6] <= 1 - hasSkill[w, k, Shifts.index(s)],
                        name=f"train_only_if_no_skill_{w}_{s}_{k}"
                    )

        # Each of the first 6 positions must always be filled for each shift
        model.addConstrs(
            (x.sum('*', s, p) == 1 for s in Shifts for p in range(6)),  # First 6 positions must be filled
            name="must_fill_first_6_positions"
        )

        # Only one position per shift for each worker (including training)
        model.addConstrs(
            (x.sum(w, s, "*") <= 1 for w in Workers for s in Shifts),
            name="one_position_per_shift"
        )

        # Training positions (6-11) can be unfilled, so no strict assignment is required
        model.addConstrs(
            (x.sum('*', s, p) <= 1 for s in Shifts for p in range(6, 12)),  # Training positions (6-11)
            name="training_optional"
        )

        # Total shifts worked by each worker
        model.addConstrs(
            (totShifts[w] == x.sum(w, "*", "*") for w in Workers),
            name="totShifts"
        )

        # Track underskilled assignments for normal positions (0-5)
        u = model.addVars([(w, s, p) for w in Workers for s in Shifts for p in range(6)], vtype=GRB.BINARY, name="underskilled")

        # Workers are underskilled if they don't meet the skill requirements for the position
        for w in Workers:
            for s in Shifts:
                for p in range(6):  # Standard positions (0-5)
                    position_skill_requirements = PositionRequirements[p]
                    worker_skills = [hasSkill[w, k, Shifts.index(s)] for k in range(3)]

                    # Only check skills of workers in main positions (not training)
                    model.addConstr(
                        u[w, s, p] >= gp.quicksum(
                            position_skill_requirements[k] * (1 - worker_skills[k]) for k in range(3)) * x[w, s, p],
                        name=f"underskilled_{w}_{s}_{p}"
                    )

        # Objective: minimize underskilled workers and slightly encourage training assignments
        model.setObjective(u.sum() - 0*(0.1 * x.sum('*', '*', range(6, 12))), GRB.MINIMIZE)

        # Optimize the model
        model.optimize()

        # Check for infeasibility or unboundedness
        if model.Status in (GRB.INFEASIBLE, GRB.UNBOUNDED):
            print("Model cannot be solved because it is infeasible or unbounded")
            sys.exit(0)

        if model.Status != GRB.OPTIMAL:
            print(f"Optimization stopped with status {model.Status}")
            sys.exit(0)

        # Gather the worker assignments for the first 5 days (first 5 shifts)
        position_assignments_first_5_days = {f"Position {p+1}": [] for p in range(6)}
        training_assignments_first_5_days = {f"Training {p-5}": [] for p in range(6, 12)}

        # Collect assignments for the first 5 days (shifts 0 to 4)
        for s in range(5):
            for p in range(6):  # Standard positions (0-5)
                assigned_workers = []
                for w in Workers:
                    if x[w, Shifts[s], p].X > 0.5:  # Collect workers assigned to the position
                        assigned_workers.append(w)
                position_assignments_first_5_days[f"Position {p+1}"].append(', '.join(assigned_workers) if assigned_workers else 'None')

            for p in range(6, 12):  # Training positions (6-11)
                assigned_workers = []
                for w in Workers:
                    if x[w, Shifts[s], p].X > 0.5:  # Collect workers assigned to the training position
                        assigned_workers.append(w)
                training_assignments_first_5_days[f"Training {p-5}"].append(', '.join(assigned_workers) if assigned_workers else 'None')

        # Display assignments for the first 5 days (positions and training)
        print("\nWorker Assignments for the First 5 Days (Positions 1-6 and Training Positions):")
        for s in range(5):
            print(f"\nDay {Shifts[s]} Assignments:")
            for p in range(6):
                print(f"Position {p+1}: {position_assignments_first_5_days[f'Position {p+1}'][s]}")
            for p in range(6, 12):
                print(f"Training {p-5}: {training_assignments_first_5_days[f'Training {p-5}'][s]}")

        # Gather the skillset of each employee at the end of the planning horizon
        last_shift = len(Shifts) - 1
        final_skills = {w: [int(hasSkill[w, k, last_shift].X) for k in range(3)] for w in Workers}

        # Count how many trainings each worker participated in
        trainings_completed = {w: sum(x[w, s, p].X > 0.5 for s in Shifts for p in range(6, 12)) for w in Workers}

        print("\nFinal Skillset of Each Worker at the End of the Planning Horizon:")
        for w in Workers:
            print(f"{w}: Skill 1: {final_skills[w][0]}, Skill 2: {final_skills[w][1]}, Skill 3: {final_skills[w][2]}")

        print("\nNumber of Trainings Completed by Each Worker:")
        for w in Workers:
            print(f"{w}: {trainings_completed[w]} trainings")

        # Gather the worker assignments for the last 5 days (last 5 shifts)
        position_assignments_last_5_days = {f"Position {p+1}": [] for p in range(6)}
        training_assignments_last_5_days = {f"Training {p-5}": [] for p in range(6, 12)}

        # Collect assignments for the last 5 days
        for s in range(last_shift - 4, last_shift + 1):
            for p in range(6):  # Standard positions (0-5)
                assigned_workers = []
                for w in Workers:
                    if x[w, Shifts[s], p].X > 0.5:  # Collect workers assigned to the position
                        assigned_workers.append(w)
                position_assignments_last_5_days[f"Position {p+1}"].append(', '.join(assigned_workers) if assigned_workers else 'None')

            for p in range(6, 12):  # Training positions (6-11)
                assigned_workers = []
                for w in Workers:
                    if x[w, Shifts[s], p].X > 0.5:  # Collect workers assigned to the training position
                        assigned_workers.append(w)
                training_assignments_last_5_days[f"Training {p-5}"].append(', '.join(assigned_workers) if assigned_workers else 'None')

        # Display assignments for the last 5 days (positions and training)
        print("\nWorker Assignments for the Last 5 Days (Positions 1-6 and Training Positions):")
        for s in range(last_shift - 4, last_shift + 1):
            print(f"\nDay {Shifts[s]} Assignments:")
            for p in range(6):
                print(f"Position {p+1}: {position_assignments_last_5_days[f'Position {p+1}'][s - (last_shift - 4)]}")
            for p in range(6, 12):
                print(f"Training {p-5}: {training_assignments_last_5_days[f'Training {p-5}'][s - (last_shift - 4)]}")

except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")
except AttributeError as e:
    print(f"Encountered an attribute error: {e}")
