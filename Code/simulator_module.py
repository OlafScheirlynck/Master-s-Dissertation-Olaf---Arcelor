# simulator_module.py

from collections import defaultdict
import numpy as np
from math import log, sqrt
import pandas as pd

# Statuscodes voor verschillende planningssituaties
STATUS = {
    "WORK": "Y",             # Gewone werktoewijzing (x)
    "TRAIN": "G",            # Trainingstoewijzing (y)
    "RESERVE": "R",          # Reservepositie (z)
    "SICK": "X",             # Ziek (ziektedag)
    "REMOVED": "O",          # Verwijderd wegens ziekte (positie onbemand)
    "REPL_RESERVE": "r_reserve",     # Vervanging door reserve
    "REPL_TRAINING": "r_training",   # Vervanging door iemand in training
    "REPL_OTHER": "r"                 # Vervanging zonder specifieke bron
}

def build_assignment_calendar(parsed_solution: dict) -> pd.DataFrame:
    """
    Bouwt de initiële kalender van shift-assignments op basis van het geoptimaliseerde Gurobi-model.
    Elke werknemer kan worden toegewezen aan een werkpositie ('Y'), training ('G'), of reserve ('R').

    Parameters:
    - parsed_solution: dictionary met Gurobi-oplossing in de vorm van {"x": ..., "y": ..., "z": ...}

    Returns:
    - pd.DataFrame: Kalendermatrix met tuples van (statuscode, positie) per shift/werknemer
    """
    required_keys = {"x", "y", "z"}
    if not required_keys.issubset(parsed_solution):
        raise ValueError(f"parsed_solution mist één of meerdere vereiste keys: {required_keys}")

    shifts = sorted({s for (_, _, s) in parsed_solution["x"]} |
                    {s for (_, _, s) in parsed_solution["y"]} |
                    {s for (_, s) in parsed_solution["z"]})

    workers = sorted({w for (w, _, _) in parsed_solution["x"]} |
                     {w for (w, _, _) in parsed_solution["y"]} |
                     {w for (w, s) in parsed_solution["z"]})

    calendar = pd.DataFrame(index=shifts, columns=workers)

    for (w, p, s), val in parsed_solution["x"].items():
        if val > 0.5:
            calendar.at[s, w] = (STATUS["WORK"], p)

    for (w, p, s), val in parsed_solution["y"].items():
        if val > 0.5:
            calendar.at[s, w] = (STATUS["TRAIN"], p)

    for (w, s), val in parsed_solution["z"].items():
        if val > 0.5:
            calendar.at[s, w] = (STATUS["RESERVE"], None)

    return calendar


def apply_sickdays_and_replacements(
    calendar: pd.DataFrame,
    sickdays: dict[int, set[int]],
    replacements: pd.DataFrame,
    parsed_solution: dict
) -> pd.DataFrame:
    """
    Past ziektedagen en vervangingen toe op een bestaande kalender.

    Parameters:
    - calendar: DataFrame met (status, positie)-tuples voor elke shift en werknemer
    - sickdays: dict met {werknemer_id: set van shifts waarin hij/zij ziek is}
    - replacements: DataFrame met kolommen ["Shift", "Replaced", "Replacement", "Position", "Type"]
    - parsed_solution: dictionary met oorspronkelijke x/y/z-structuur, gebruikt voor training lookup

    Returns:
    - pd.DataFrame: geüpdatete kalender met ziektes en vervangingen gemarkeerd
    """

    required_keys = {"x", "y", "z"}
    if not required_keys.issubset(parsed_solution):
        raise ValueError(f"parsed_solution mist vereiste keys: {required_keys}")

    # 1. Markeer ziektedagen
    for w, sick_shifts in sickdays.items():
        for s in sick_shifts:
            current = calendar.at[s, w] if (s in calendar.index and w in calendar.columns) else None
            if isinstance(current, tuple) and current[1] is not None:
                calendar.at[s, w] = (STATUS["SICK"], current[1])
            else:
                calendar.at[s, w] = (STATUS["SICK"], None)

    # 2. Pas vervangingen toe
    for _, row in replacements.iterrows():
        s = row["Shift"]
        replaced = row["Replaced"]
        replacement = row["Replacement"]
        pos = row["Position"]
        replacement_type = row["Type"]

        # Markeer verwijderde werknemer
        calendar.at[s, replaced] = (STATUS["REMOVED"], pos)

        # Vervanger bepalen
        if "Reserve" in replacement_type:
            calendar.at[s, replacement] = (STATUS["REPL_RESERVE"], pos)

        elif "Training" in replacement_type:
            # Zoek originele training waarvoor de vervanger ingeschreven stond
            training_pos = next((p for (w2, p, s2), v in parsed_solution["y"].items()
                                 if w2 == replacement and s2 == s and v > 0.5), None)
            calendar.at[s, replacement] = (STATUS["REPL_TRAINING"], (pos, training_pos))

        else:
            calendar.at[s, replacement] = (STATUS["REPL_OTHER"], pos)

    return calendar


def simulate_sickdays_from_assignments(parsed_solution, ziekte_mapping):
    sickdays = defaultdict(set)
    worked_shifts = sorted(parsed_solution["x"].keys(), key=lambda x: (x[0], x[2]))
    ongoing_illness_until = defaultdict(lambda: -1)

    for (w, p, s) in worked_shifts:
        if parsed_solution["x"].get((w, p, s), 0) < 0.5:
            continue
        if s <= ongoing_illness_until[w]:
            continue
        if w not in ziekte_mapping:
            continue

        z = ziekte_mapping[w]
        prob_sick = z["prob_sick"]


        if np.random.rand() < prob_sick:
            duration = int()
            duration = max(1, duration)
            sick_shifts = range(s, s + duration)
            sickdays[w].update(sick_shifts)
            ongoing_illness_until[w] = s + duration - 1

    return sickdays

def detect_missed_trainings(parsed_solution, sickdays):
    missed_trainings = defaultdict(set)
    for (w, p, s), val in parsed_solution["y"].items():
        if val > 0.5 and s in sickdays.get(w, set()):
            missed_trainings[w].add((p, s))
    return missed_trainings

def catch_up_missed_trainings(parsed_solution, missed_trainings):
    updated_training = dict(parsed_solution["y"])
    for w, missed in missed_trainings.items():
        all_used_shifts = {
            s for (ww, pp, s), v in parsed_solution["x"].items() if ww == w and v > 0.5
        } | {
            s for (ww, pp, s), v in parsed_solution["y"].items() if ww == w and v > 0.5
        } | {
            s for (ww, s), v in parsed_solution["z"].items() if ww == w and v > 0.5
        }

        for p, s_missed in sorted(missed):
            s_new = s_missed + 1
            while (w, p, s_new) in updated_training or s_new in all_used_shifts:
                s_new += 1
                if s_new > max(s for (_, _, s) in parsed_solution["y"]):
                    break
            updated_training[(w, p, s_new)] = 1.0
            all_used_shifts.add(s_new)
    return updated_training

def pick_worker_to_leave(W: list[int], mutation_probs: dict[int, float]) -> int:
    """
    Kies op basis van individuele mutatiekans een werknemer die vertrekt.
    """
    weights = [mutation_probs.get(w, 0.01) for w in W]
    chosen = np.random.choice(W, p=np.array(weights) / sum(weights))
    return chosen

def reoptimize_without_worker(parsed_inputs: dict, removed_worker: int, start_date, end_date):
    """
    Heroptimaliseer model zonder een specifieke werknemer.
    """
    inputs_new = {**parsed_inputs}
    inputs_new["W"] = [w for w in parsed_inputs["W"] if w != removed_worker]
    inputs_new["I_wks"] = {w: v for w, v in parsed_inputs["I_wks"].items() if w != removed_worker}
    inputs_new["experts"] = {w: v for w, v in parsed_inputs["experts"].items() if w != removed_worker}
    inputs_new["f_w_s"] = {w: v for w, v in parsed_inputs["f_w_s"].items() if w != removed_worker}
    # alle andere structuren zijn compatibel

    from solver_module import build_gurobi_model
    model, vars = build_gurobi_model(inputs_new)
    model.optimize()

    from solution_utils import parse_solution
    parsed = parse_solution(model, vars)
    return parsed

def add_new_unskilled_worker(parsed_inputs: dict, new_id: int):
    """
    Voeg een nieuwe ongetrainde werknemer toe.
    """
    inputs = parsed_inputs.copy()
    inputs["W"].append(new_id)
    for k in inputs["K"]:
        inputs["I_wks"][new_id][k] = {0: 0}
        inputs["experts"][new_id][k] = 0
    inputs["f_w_s"][new_id] = {s: 0 for s in inputs["S"]}
    return inputs
