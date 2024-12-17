import sys
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Initialize shifts (14 initial shifts + 18 additional weeks of 7 days each)
Shifts = [f"Mon{d}" if d % 7 == 1 else f"Tue{d}" if d % 7 == 2 else f"Wed{d}" if d % 7 == 3 else
          f"Thu{d}" if d % 7 == 4 else f"Fri{d}" if d % 7 == 5 else f"Sat{d}" if d % 7 == 6 else f"Sun{d}"
          for d in range(1, 127)]

# Workers
Workers = ["Amy", "Bob", "Cathy", "Dan", "Ed", "Fred", "Gu", "Tobi"]

# Predefined skill levels (2 or higher for max 1 skill per person)
InitialSkillLevels = {
    "Amy": [3, 3, 0],    # Skill 1, Level 3
    "Bob": [0, 3, 4],    # Skill 2, Level 3
    "Cathy": [0, 4, 4],  # Skill 3, Level 4
    "Dan": [2, 3, 0],    # Skill 1, Level 2
    "Ed": [0, 2, 4],     # Skill 2, Level 2
    "Fred": [0, 0, 4],   # Skill 3, Level 4
    "Gu": [2, 0, 0],     # Skill 1, Level 2
    "Tobi": [0, 3, 0],   # Skill 2, Level 3
}

# Position skill requirements (6 main positions, indexed 0-5)
PositionRequirements = [
    [1, 0, 0],  # Position 1: requires skill 1
    [0, 1, 0],  # Position 2: requires skill 2
    [0, 0, 1],  # Position 3: requires skill 3
    [1, 1, 0],  # Position 4: requires skill 1 and 2
    [0, 1, 1],  # Position 5: requires skill 2 and 3
    [1, 0, 1],  # Position 6: requires skill 1 and 3
]

# Skill acquisition time for each skill
SkillAcquisitionTime = [8, 12, 24]

try:
    # Create Gurobi model
    with gp.Model("workforce_with_training_time") as model:

        model.setParam("OutputFlag", 1)  # Display solver output
        model.setParam("MIPGap", 0.01)  # Set gap tolerance to 1%

        # Decision variables
        x = model.addVars([(w, s, p) for w in Workers for s in Shifts for p in range(12)],
                          vtype=GRB.BINARY, name="x")
        y = model.addVars([(w, s, p) for w in Workers for s in Shifts for p in range(6)],
                          vtype=GRB.BINARY, name="y")
        hasSkill = model.addVars([(w, k, s) for w in Workers for k in range(3) for s in range(len(Shifts))],
                                 vtype=GRB.BINARY, name="hasSkill")
        acquiredSkill = model.addVars([(w, k, s) for w in Workers for k in range(3) for s in range(len(Shifts))],
                                      vtype=GRB.BINARY, name="acquiredSkill")
        trainingProgress = model.addVars([(w, k, s) for w in Workers for k in range(3) for s in range(len(Shifts))],
                                         lb=0, vtype=GRB.CONTINUOUS, name="trainingProgress")
        trainingActive = model.addVars([(s, p) for s in Shifts for p in range(6)],
                                       vtype=GRB.BINARY, name="trainingActive")
        expertPresent = model.addVars([(s, p) for s in Shifts for p in range(6)],
                                       vtype=GRB.BINARY, name="expertPresent")

        # Initialize skill levels at the first shift
        for w in Workers:
            for k in range(3):
                model.addConstr(hasSkill[w, k, 0] == (1 if InitialSkillLevels[w][k] >= 2 else 0))
                model.addConstr(trainingProgress[w, k, 0] == (SkillAcquisitionTime[k] if InitialSkillLevels[w][k] >= 2 else 0))

        # Skill acquisition constraints
        for w in Workers:
            for s in range(1, len(Shifts)):
                for k in range(3):
                    model.addConstr(
                        trainingProgress[w, k, s] == trainingProgress[w, k, s - 1] + x[w, Shifts[s - 1], 6 + k],
                        name=f"training_progress_{w}_{k}_{s}"
                    )
                    model.addConstr(
                        acquiredSkill[w, k, s] <= trainingProgress[w, k, s] / SkillAcquisitionTime[k],
                        name=f"training_completion_{w}_{k}_{s}"
                    )
                    model.addConstr(
                        trainingProgress[w, k, s] - SkillAcquisitionTime[k] * acquiredSkill[w, k, s] >= 0,
                        name=f"acquiredSkill_binary_{w}_{k}_{s}"
                    )
                    model.addConstr(
                        hasSkill[w, k, s] <= acquiredSkill[w, k, s],
                        name=f"link_acquiredSkill_to_hasSkill_{w}_{k}_{s}"
                    )
                    model.addConstr(
                        hasSkill[w, k, s] >= hasSkill[w, k, s - 1],
                        name=f"maintain_hasSkill_{w}_{k}_{s}"
                    )

        for s in Shifts:
            for p in range(6):
                required_skills = PositionRequirements[p]

                # Expert presence condition
                model.addConstr(
                    expertPresent[s, p] <= gp.quicksum(
                        x[w, s, p] * (InitialSkillLevels[w][k] >= 3) * required_skills[k]
                        for w in Workers for k in range(3)
                    ),
                    name=f"expert_present_{s}_{p}_upper"
                )
                model.addConstr(
                    expertPresent[s, p] >= gp.quicksum(
                        x[w, s, p] * (InitialSkillLevels[w][k] >= 3) * required_skills[k]
                        for w in Workers for k in range(3)
                    ),
                    name=f"expert_present_{s}_{p}_lower"
                )

        # Ensure expert presence for training
        for s in Shifts:
            for p in range(6):  # Regular positions
                required_skills = PositionRequirements[p]  # Get required skills for position p
                model.addConstr(
                    expertPresent[s, p] >= gp.quicksum(
                        x[w, s, p] * any(InitialSkillLevels[w][k] >= 3 and required_skills[k] == 1 for k in range(3))
                        for w in Workers
                    ),
                    name=f"expert_present_{s}_{p}"
                )

                model.addConstr(
                    gp.quicksum(trainingActive[s, p] for w in Workers) <= expertPresent[s, p],
                    name=f"training_requires_expert_{s}_{p}"
                )

        # Worker qualification for positions
        for w in Workers:
            for s in Shifts:
                for p in range(6):
                    required_skills = PositionRequirements[p]
                    model.addConstr(
                        y[w, s, p] <= gp.quicksum(hasSkill[w, k, Shifts.index(s)] * required_skills[k] for k in range(3)),
                        name=f"qualification_{w}_{s}_{p}"
                    )
                    model.addConstr(
                        x[w, s, p] <= y[w, s, p],
                        name=f"assign_only_if_qualified_{w}_{s}_{p}"
                    )

        # Constraints for worker assignment
        model.addConstrs(
            (x.sum('*', s, p) == 1 for s in Shifts for p in range(6)),
            name="must_fill_positions"
        )
        model.addConstrs(
            (x.sum(w, s, '*') <= 1 for w in Workers for s in Shifts),
            name="one_position_per_worker_per_shift"
        )
        model.addConstrs(
            (x.sum('*', s, p) <= 1 for s in Shifts for p in range(6, 12)),
            name="training_optional"
        )

        # Objective: maximize the number of workers qualified for positions 0-5
        model.setObjective(
            y.sum('*', '*', '*') + 0.1 * trainingActive.sum('*', '*'),
            GRB.MAXIMIZE
        )

        # Optimize the model
        model.optimize()

        if model.Status in (GRB.INFEASIBLE, GRB.UNBOUNDED):
            print("Model cannot be solved because it is infeasible or unbounded")
            sys.exit(0)

        if model.Status != GRB.OPTIMAL:
            print(f"Optimization stopped with status {model.Status}")
            sys.exit(0)

        # Save the initial skill matrix
        initial_skill_matrix = pd.DataFrame(InitialSkillLevels, index=["Skill 1", "Skill 2", "Skill 3"]).T

        # Derive the final skill matrix
        final_skills = [[int(hasSkill[w, k, len(Shifts) - 1].X) for k in range(3)] for w in Workers]
        final_skill_matrix = pd.DataFrame(final_skills, index=Workers, columns=[f"Skill {k + 1}" for k in range(3)])

        # Count the number of training sessions per worker
        training_summary = {
            w: [sum(int(x[w, s, k + 6].X) for s in Shifts) for k in range(3)]
            for w in Workers
        }
        training_summary_df = pd.DataFrame(training_summary, index=[f"Skill {k + 1}" for k in range(3)]).T
        training_summary_df.columns = [f"Training for {col}" for col in training_summary_df.columns]



        # Generate the schedule
        schedule = pd.DataFrame(index=Shifts, columns=[f"Position {p + 1}" for p in range(12)])
        for s in Shifts:
            for p in range(12):
                for w in Workers:
                    if x[w, s, p].X > 0.5:
                        schedule.loc[s, f"Position {p + 1}"] = w

        # Combine all data into a single file
        with pd.ExcelWriter("skill_matrices_with_training_time.xlsx") as writer:
            initial_skill_matrix.to_excel(writer, sheet_name="Initial Skills")
            final_skill_matrix.to_excel(writer, sheet_name="Final Skills")
            training_summary_df.to_excel(writer, sheet_name="Training Summary")
            schedule.to_excel(writer, sheet_name="Schedule")

        print("Skill matrices, training summary, and schedule have been exported to 'skill_matrices_with_training_time.xlsx'.")


except gp.GurobiError as e:
    print(f"Error code {e.errno}: {e}")
except AttributeError as e:
    print(f"Encountered an attribute error: {e}")