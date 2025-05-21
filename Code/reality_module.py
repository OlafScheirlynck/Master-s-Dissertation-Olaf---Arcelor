import os
import random as rd
from math import log, sqrt


from Code.matching_utils import hopcroft_karp


#===IMPORT FILES===

#simulate_vacation_days
verlofsaldo_path = r"C:\Users\olafs\OneDrive\Documenten\GitHub\Master-s-Dissertation-Olaf---Arcelor\Code\VerlofSaldo.xlsx"
simulation_parameters_path = r"C:\Users\olafs\OneDrive\Documenten\GitHub\Master-s-Dissertation-Olaf---Arcelor\Code\SimulatieResultaten.xlsx"



#===FUNCTIONS===
# new_vacation_simulation.py
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import random


def simulate_vacation_days(
    W: list[int],
    start_date: datetime,
    end_date: datetime,
    leave_targets: dict[int, float],
    month_probs: dict[int, float],
    vacation_sampler: 'EmpiricalSampler',
    mean_leave_duration: float,
    min_available: int = 8
) -> tuple:
    """
    Simuleert vakantiedagen per werknemer op basis van maandverdeling en individuele targets,
    met minimumbezetting en empirische sampling.
    """
    import pandas as pd
    from collections import defaultdict
    import random

    total_days = (end_date - start_date).days
    num_shifts = int(total_days * 0.75)
    S = list(range(num_shifts))
    max_absent = len(W) - min_available

    shift_to_date = {s: start_date + timedelta(days=int(s * 4 / 3)) for s in S}
    shift_to_month = {s: d.month for s, d in shift_to_date.items()}
    shift_to_day = {s: d.day for s, d in shift_to_date.items()}

    vacation_plan = {w: {m: [] for m in range(1, 13)} for w in W}
    vacation_matrix = pd.DataFrame(index=W, columns=S, data="")
    shift_leave_map = defaultdict(list)
    # Voeg standaard van 50 dagen toe voor werknemers zonder target
    leave_targets = {w: leave_targets.get(w, 50) for w in W}

    # Genormaliseerde maandverdeling
    normed_probs = {
        w: {
            m: (month_probs.get(m, 0.01) * leave_targets.get(w, 0)) / (mean_leave_duration if mean_leave_duration > 0 else 1)
            for m in range(1, 13)
        }
        for w in W
    }
    for w in normed_probs:
        total = sum(normed_probs[w].values())
        for m in normed_probs[w]:
            normed_probs[w][m] /= total if total > 0 else 1

    for w in W:
        used_days = 0
        tries = 0
        max_tries = 300

        while used_days < leave_targets.get(w, 0) and tries < max_tries:
            tries += 1

            month = random.choices(range(1, 13), weights=normed_probs[w].values())[0]
            candidate_shifts = [s for s in S if shift_to_month[s] == month]
            random.shuffle(candidate_shifts)

            for s in candidate_shifts:
                if len(shift_leave_map[s]) >= max_absent:
                    continue

                # Nieuw: duur komt uit vacation_sampler
                duration = vacation_sampler.sample()

                shift_indices = list(range(s, min(s + duration, max(S) + 1)))

                if any(len(shift_leave_map[ss]) >= max_absent for ss in shift_indices):
                    continue

                for ss in shift_indices:
                    vacation_matrix.at[w, ss] = "x"
                    shift_leave_map[ss].append(w)
                    m, d = shift_to_month[ss], shift_to_day[ss]
                    if (m, d) not in vacation_plan[w][m]:
                        vacation_plan[w][m].append((m, d))
                used_days += duration
                break
    os.makedirs("output", exist_ok=True)
    vacation_matrix.index.name = "Worker"
    export_path = os.path.join("output", "vacation_matrix.xlsx")
    vacation_matrix.to_excel(export_path)
    return vacation_plan, vacation_matrix, shift_leave_map



    # === Corrigeer overschrijdingen ===
    def count_unavailable(s, vacation_plan):
        m = shift_to_month[s]
        d = shift_to_day[s]
        return sum(1 for w in W if (m, d) in vacation_plan[w][m])

    for s in S:
        while len(W) - count_unavailable(s, vacation_plan) < min_available:
            shift_leave_map[s].sort(key=lambda w: sum(len(vacation_plan[w][m]) for m in vacation_plan[w]), reverse=True)
            if shift_leave_map[s]:
                w = shift_leave_map[s].pop(0)
            else:
                print(f"⚠️ Geen werknemers beschikbaar op shift {s} om verlof te herplannen.")
                continue  # skip naar volgende shift of probeer alternatief

            m = shift_to_month[s]
            d = shift_to_day[s]
            vacation_plan[w][m] = [(mm, dd) for (mm, dd) in vacation_plan[w][m] if dd != d]

            # Herplannen
            # Sorteer alle shifts op minst aantal afwezigen
            shifts_by_availability = sorted(S, key=lambda s: len(W) - count_unavailable(s, vacation_plan), reverse=True)
            max_attempts = 100
            attempts = 0
            herpland = False

            for s_new in shifts_by_availability:
                if attempts > max_attempts:
                    break
                attempts += 1

                # Niet plannen op dezelfde dag opnieuw
                if s_new == s:
                    continue

                m_new = shift_to_month[s_new]
                d_new = shift_to_day[s_new]
                # Controleer of werknemer daar al verlof heeft
                if (m_new, d_new) in vacation_plan[w][m_new]:
                    continue

                if len(W) - count_unavailable(s_new, vacation_plan) >= min_available:
                    vacation_plan[w][m_new].append((m_new, d_new))
                    shift_leave_map = build_shift_leave_map(vacation_plan)
                    herpland = True
                    break

            if not herpland:
                print(f"❌ KON werknemer {w} NIET herplannen vanaf shift {s} → verlofdag verwijderd.")

            for s_new in shifts_by_availability:
                if len(W) - count_unavailable(s_new, vacation_plan) >= min_available:
                    m_new = shift_to_month[s_new]
                    d_new = shift_to_day[s_new]
                    if (m_new, d_new) not in vacation_plan[w][m_new]:
                        vacation_plan[w][m_new].append((m_new, d_new))
                        shift_leave_map = build_shift_leave_map(vacation_plan)
                        herpland = True
                        break

            if not herpland:
                print(f"⚠️ Kon werknemer {w} niet herplannen vanaf shift {s}")

    # === Outputmatrix maken ===
    vacation_matrix = pd.DataFrame(index=W, columns=S, data="")
    for w in W:
        for m, days in vacation_plan[w].items():
            for (_, d) in days:
                try:
                    shift = int((datetime(start_date.year, m, d) - start_date).days * 0.75)
                    if shift in S:
                        vacation_matrix.at[w, shift] = "x"
                except ValueError:
                    continue

    vacation_matrix.columns = [f"S{s}" for s in S]
    vacation_matrix.index.name = "Worker"
    vacation_matrix.to_excel(output_path_vacation_days)

    print("✅ Verlof gesimuleerd en gecorrigeerd. Resultaat naar:", output_path_vacation_days)


    return vacation_plan, vacation_matrix, shift_leave_map


def get_shift_index_from_date(target_date: datetime, start_date: datetime) -> int:
    """
    Converteert een datum naar een shiftindex, uitgaande van 0.75 shifts per dag.
    """
    total_days = (target_date - start_date).days
    return int(total_days * 0.75)

def lognormal_params(mean: float, std: float) -> tuple[float, float]:
    sigma = sqrt(log(1 + (std / mean) ** 2))
    mu = log((mean ** 2) / sqrt(std ** 2 + mean ** 2))
    return mu, sigma




def simulate_full_sickness_effect(parsed_solution, inputs, ziekte_mapping, r_p_k, Pm, S):

    sickdays = defaultdict(set)
    missed_trainings = defaultdict(set)
    replacements_log = []
    updated_training = dict(parsed_solution['y'])
    updated_q = dict(parsed_solution['q'])
    updated_z = dict(parsed_solution['z'])
    updated_fws = defaultdict(lambda: defaultdict(int))
    uncaught_trainings = set()  # if using .add()

    ongoing_illness_until = defaultdict(lambda: -1)

    for (w, p, s) in sorted(parsed_solution["x"].keys(), key=lambda x: (x[0], x[2])):
        if parsed_solution["x"].get((w, p, s), 0) < 0.5:
            continue
        if s <= ongoing_illness_until[w]:
            continue
        z = ziekte_mapping.get(w, {"prob_sick": 0.005})  # standaard 5% ziektekans
        sick_sampler = inputs["sick_sampler"]
        if np.random.rand() < z["prob_sick"]:

            duration = sick_sampler.sample()

            sick_shifts = range(s, s + duration)
            sickdays[w].update(sick_shifts)
            ongoing_illness_until[w] = s + duration - 1

    for (w, p, s), val in parsed_solution["y"].items():
        if val > 0.5 and s in sickdays.get(w, set()):
            missed_trainings[w].add((p, s))

    for (w, p, s), val in parsed_solution['x'].items():
        if val < 0.5 or s not in sickdays[w]:
            continue

        qualified = [ww for (ww, pp, ss), qval in parsed_solution['q'].items()
                     if pp == p and ss == s and qval > 0.5]
        reserve_pool = [ww for (ww, ss), val in parsed_solution['z'].items() if ss == s and val > 0.5]
        qualified_reserve = [ww for ww in reserve_pool if ww in qualified]
        unqualified_reserve = [ww for ww in reserve_pool if ww not in qualified]
        training_pool = [ww for (ww, pp, ss), val in parsed_solution['y'].items() if ss == s and val > 0.5]
        qualified_training = [ww for ww in training_pool if ww in qualified]
        unqualified_training = [ww for ww in training_pool if ww not in qualified]

        replacement = None
        replacement_type = None

        for pool, label in [
            (qualified_reserve, "Reserve (Qualified)"),
            (qualified_training, "Training (Qualified)")
        ]:
            if pool:
                replacement = pool[0]
                replacement_type = label
                break

        # 3. Herstructurering via matching
        if replacement is None:
            staffed_workers = {w2 for (w2, p2, s2) in parsed_solution["x"] if s2 == s and parsed_solution["x"][(w2, p2, s2)] > 0.5 and w2 not in sickdays}
            edges = {}
            for ww in staffed_workers:
                edges[ww] = [pp for (_, pp, ss) in parsed_solution["q"] if
                             ss == s and parsed_solution["q"].get((ww, pp, ss), 0) > 0.5]

            open_positions = sorted(set(pp for lst in edges.values() for pp in lst))

            matching = hopcroft_karp(list(staffed_workers), open_positions, edges)

            if matching:
                for repl, pos in matching.items():
                    replacement = repl
                    replacement_type = "Reassigned via Matching"
                    break

        # 4. Unqualified options
        if replacement is None:
            for pool, label in [
                (unqualified_reserve, "Reserve (Unqualified)"),
                (unqualified_training, "Training (Unqualified)")
            ]:
                if pool:
                    replacement = pool[0]
                    replacement_type = label
                    break

        # 5. Fallback: van verlof halen
        if replacement is None:
            for (ww, ss), _ in parsed_solution['z'].items():
                if ss == s and ww in qualified:
                    replacement = ww
                    replacement_type = "From Leave (Qualified)"
                    updated_fws[ww][s] = 1
                    updated_q[(ww, p, s)] = 0.0
                    break

        if replacement:
            replacements_log.append({
                "Shift": s,
                "Replaced": w,
                "Replacement": replacement,
                "Position": p,
                "Type": replacement_type
            })
            updated_q[(replacement, p, s)] = 0.0
            updated_training.pop((replacement, p, s), None)

    # 4. Plan catch-up trainingen and track uncaught trainings
    for w, missed in missed_trainings.items():
        used_shifts = set(s for (ww, pp, s), val in parsed_solution["x"].items() if ww == w and val > 0.5)
        used_shifts |= set(s for (ww, pp, s), val in parsed_solution["y"].items() if ww == w and val > 0.5)
        used_shifts |= set(s for (ww, s), val in parsed_solution["z"].items() if ww == w and val > 0.5)

        for p, s_missed in sorted(missed):
            # Find next available shift
            s_new = s_missed + 1
            while (s_new in used_shifts or
                   s_new in sickdays.get(w, set()) or
                   (w, p, s_new) in updated_training):
                s_new += 1
                if s_new > max(S):  # If no more shifts available
                    uncaught_trainings.add((w, p, s_missed))  # Store as tuple
                    break
            else:
                updated_training[(w, p, s_new)] = 1.0
                # Update qualifications for future shifts
                for future_s in S:
                    if future_s >= s_new:
                        updated_q[(w, p, future_s)] = 1.0
                used_shifts.add(s_new)
    # 5. Pas het behalen van een kwalificatie aan aan het nieuwe opleidingsplan
        for p, s_missed in sorted(missed):
            candidate_shifts = [s_new for s_new in S if s_new > s_missed and s_new not in used_shifts]
            for s_new in candidate_shifts:
                updated_training[(w, p, s_new)] = 1.0
                # 🆕 Wanneer catch-up gedaan is, herstel q=1 vanaf die dag
                for future_s in S:
                    if future_s >= s_new:
                        updated_q[(w, p, future_s)] = 1.0
                used_shifts.add(s_new)
                break


    return {
        "x": parsed_solution["x"],
        "y": updated_training,
        "z": updated_z,
        "q": updated_q
    }, pd.DataFrame(replacements_log), sickdays, missed_trainings, updated_fws, uncaught_trainings
from collections import defaultdict
import numpy as np
import pandas as pd

def evaluate_solution_robustness(parsed_solution, inputs, ziekte_mapping, r_p_k, Pm, S, num_simulations=1000):
    total_underqualified_counts = []
    replacement_types_counter = defaultdict(int)
    avg_workers_per_shift_list = []

    for sim in range(num_simulations):
        # Simuleer ziekte-effecten
        from Code.reality_module import simulate_full_sickness_effect
        parsed_after_sickness, replacements, sickdays, missed_trainings, updated_fws = simulate_full_sickness_effect(
            parsed_solution, inputs, ziekte_mapping, r_p_k, Pm, S
        )

        # Tel underqualified toewijzingen
        from Code.solution_utils import extract_underqualified_counts
        underq_counts = extract_underqualified_counts(parsed_after_sickness, inputs["P_map"], sickdays)
        total = underq_counts["UnqualifiedCount"].sum() if "UnqualifiedCount" in underq_counts else 0
        total_underqualified_counts.append(total)

        # Tel type vervangingen
        for t in replacements["Type"].value_counts().to_dict().items():
            replacement_types_counter[t[0]] += t[1]

        # Gemiddeld aantal mensen per shift (excl. ziektedagen)
        shifts = list(set(s for (_, _, s) in parsed_after_sickness["x"]))
        worker_count = defaultdict(int)
        for (w, p, s), val in parsed_after_sickness["x"].items():
            if val > 0.5 and w not in sickdays or s not in sickdays[w]:
                worker_count[s] += 1
        avg_workers_per_shift = np.mean(list(worker_count.values()))
        avg_workers_per_shift_list.append(avg_workers_per_shift)

    results = {
        "avg_underqualified": np.mean(total_underqualified_counts),
        "std_underqualified": np.std(total_underqualified_counts),
        "avg_workers_per_shift": np.mean(avg_workers_per_shift_list),
        "replacement_types": dict(replacement_types_counter),
        "robustness_metric": np.percentile(total_underqualified_counts, 90)  # bijv. 90e percentiel
    }

    return results



