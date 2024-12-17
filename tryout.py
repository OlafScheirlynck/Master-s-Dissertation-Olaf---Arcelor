import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

# Basisparameters
num_shifts = 20  # Aantal shifts in de planning
team_size = 15  # Totale teamgrootte
positions = list(range(1, 18))  # Posities 1 tot 17 (1-7 standaard, 8-10 reserve, 11-17 training)
required_team_size = 8  # Minimale bezetting per shift
M = 1000  # Grote constante voor de implementatie van bepaalde constraints

# Initialisatie van functietoewijzingen en trainingsvereisten
F = np.random.choice([1, 2, 3, 4], team_size)  # Toewijzing van teamleden aan functies
T = np.random.randint(0, 5, size=(team_size, 7))  # Initiële taakniveaus, aangepast om alleen posities 1-7 te bevatten

# Gurobi-model
model = gp.Model("TeamScheduleOptimization")

# Variabelen
x = model.addVars(team_size, len(positions), num_shifts, vtype=GRB.BINARY, name="x")
y = model.addVars(team_size, 7, vtype=GRB.BINARY, name="y")  # Alleen voor posities 1-7
m = model.addVars(team_size, vtype=GRB.INTEGER, name="m")
b = model.addVars(team_size, vtype=GRB.BINARY, name="b")

# Doelfunctie: minimaliseer maximum herhalingen per positie en onderbezetting
P = 10  # Strafparameter voor onderbezetting
model.setObjective(gp.quicksum(m[i] for i in range(team_size))
                   - P * gp.quicksum(x[i, j, k] for i in range(team_size)
                                     for j in range(7) for k in range(num_shifts)), GRB.MINIMIZE)

# Constraints
# Constraint 1: Dagelijkse bezetting van standaardposities
for k in range(num_shifts):
    for j in range(7):
        model.addConstr(gp.quicksum(x[i, j, k] for i in range(team_size)) == 1,
                        f"PositionAssignment_{j}_{k}")

# Constraint 2: Optionele bezetting van reserveposities
for i in range(team_size):
    for j in range(8, 11):
        for k in range(num_shifts):
            model.addConstr(x[i, j, k] >= 0, f"ReservePosition_{i}_{j}_{k}")

# Constraint 3: Optionele bezetting van trainingsposities
for i in range(team_size):
    for j in range(7):  # Alleen posities 1-7 hebben gekoppelde trainingsposities (11-17)
        for k in range(num_shifts):
            model.addConstr(x[i, j + 10, k] <= (2 - T[i, j]) / 2, f"TrainingPosition_{i}_{j}_{k}")

# Constraint 4: Relatie tussen x en y variabelen
for i in range(team_size):
    for j in range(7):
        model.addConstr(y[i, j] <= gp.quicksum(x[i, j, k] for k in range(num_shifts)),
                        f"RelateXY_{i}_{j}")

# Constraint 5: Shiftlimiet per persoon
M_i = 10  # Voorbeeld maximum shifts per persoon
for i in range(team_size):
    model.addConstr(gp.quicksum(x[i, j, k] for j in range(17) for k in range(num_shifts)) <= M_i,
                    f"MaxShifts_{i}")

# Constraint 6: Kwalificatie per functie
for i in range(team_size):
    for j in range(7):
        for k in range(num_shifts):
            model.addConstr(x[i, j, k] <= T[i, j], f"Qualification_{i}_{j}_{k}")

# Constraint 7: Eerste opleiding moet positie 1 zijn
for i in range(team_size):
    model.addConstr(gp.quicksum(x[i, 11, k] for k in range(num_shifts)) >= 1, f"FirstTraining_{i}")

# Constraint 8: Vooruitgang naar een nieuw functieniveau (vervangende logica voor T[i, j] >= 2)
for i in range(team_size):
    if F[i] < 4:  # Alleen als ze nog kunnen promoveren
        for j in range(7):
            # Definieer binaire variabele die aangeeft of T[i, j] >= 2
            t_geq_2 = model.addVar(vtype=GRB.BINARY, name=f"T_geq_2_{i}_{j}")
            model.addConstr(t_geq_2 == (1 if T[i, j] >= 2 else 0), f"CheckT_{i}_{j}")

            model.addConstr(gp.quicksum(x[i, j + 10, k] for k in range(num_shifts)) >= t_geq_2,
                            f"TrainingProgress_{i}_{j}")

# Constraint 9: Maximalisatie van diversiteit binnen functies
for i in range(team_size):
    for j in range(7):
        model.addConstr(gp.quicksum(x[i, j, k] for k in range(num_shifts)) <= m[i],
                        f"MaxDiversity_{i}_{j}")

# Constraint 10: Bijhouden van trainingsuren
R_j, H_j = 3, 2  # Voorbeelden voor aantal trainingen en uren op reservepositie
for i in range(team_size):
    for j in range(7):
        z = gp.quicksum(x[i, j + 10, k] for k in range(num_shifts)) + gp.quicksum(
            x[i, j + 7, k] for k in range(num_shifts))
        model.addConstr(z >= R_j + H_j, f"TrainingHours_{i}_{j}")

# Constraint 11: Behouden van vaardigheden
for i in range(team_size):
    for j in range(7):
        model.addConstr(gp.quicksum(x[i, j, k] for k in range(max(0, num_shifts - 40), num_shifts)) >= 1,
                        f"SkillRetention_{i}_{j}")

# Constraint 12: Eerste positie op Niveau 2 moet positie 1 zijn
for i in range(team_size):
    for j in range(1, 7):
        model.addConstr(y[i, j] <= b[i] * M, f"FirstPosition2_{i}_{j}")

# Constraint 13: Aanwezigheid van Functie 3 en Functie 4
for k in range(num_shifts):
    model.addConstr(gp.quicksum(x[i, 6, k] for i in range(team_size) if F[i] == 3) >= 1, f"Func3Presence_{k}")
    model.addConstr(gp.quicksum(x[i, 7, k] for i in range(team_size) if F[i] == 4) >= 1, f"Func4Presence_{k}")

# Optimalisatie uitvoeren
model.optimize()

# Resultaten opslaan als basisplanning
base_schedule = []
if model.status == GRB.OPTIMAL:
    for i in range(team_size):
        for j in range(len(positions)):
            for k in range(num_shifts):
                if x[i, j, k].x > 0.5:
                    base_schedule.append({"Shift": k, "Position": j + 1, "TeamMember": f"TeamMember_{i}"})

base_schedule_df = pd.DataFrame(base_schedule)
print(base_schedule_df.head())
