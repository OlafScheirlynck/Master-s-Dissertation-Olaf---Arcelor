# Integratie in adaptive_optimizer.py — toevoegen van automatisch opslaan
from collections import defaultdict
import os
import pandas as pd
from copy import deepcopy

from Code.run_simulation import simulate_mutation
from input_module import load_all_inputs, save_alfa_p_weights_to_excel
from solver_module import solve_workforce_model
from solution_utils import export_solution_to_excel, extract_underqualified_counts, parse_multiple_solutions
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import defaultdict

from Code.simulator_module import build_assignment_calendar
from mutation_module import simulate_mutation
from Code.reality_module import simulate_vacation_days
from Code.input_module import load_all_inputs, build_sickness_mapping
from Code.solver_module import solve_workforce_model, prepare_model_inputs
from Code.solution_utils import (
    parse_multiple_solutions,
    export_solution_to_excel,
    extract_underqualified_counts,
    generate_sickday_calendar_detailed,
    export_styled_calendar_detailed,
    export_sickness_logs, debug_initial_qualifications, debug_competency_matrix, debug_inputs,
    export_inputs_to_excel, debug_inzetbaarheid
)
from input_module import load_mutation_probability
'''

# === MODEL PARAMETERS ===
MAX_ITER = 10
MIPGAP = 0.2
TOLERANCE = 1
NUM_SOLUTIONS = 1
SICKNESS_ITERATIONS = 10000
TIMEHORIZON_IN_YEARS = 0#in years
TIMEHORIZON_IN_MONTHS = 6
TIMEHORIZON_IN_DAYS = 0

# === BUFFERFACTOREN ===
GAMMA_SICK_DURATION = 1.0
GAMMA_VACATION_DURATION = 1.0
GAMMA_TRAINING_DURATION = 1.0
GAMMA_SICK_PROB = 1.0
GAMMA_VACATION_PROB = 1.0
GAMMA_MUTATION_PROB = 1.0

# === OTHER RULES ===
MIN_AVAILABLE = 8
# === CONFIGURATIE ===
TEAM_NAME = "Ploeg C"
RUN_NAME = f"simulation_{TEAM_NAME}_{TIMEHORIZON_IN_YEARS}Y_{TIMEHORIZON_IN_MONTHS}M_G{MIPGAP}_MA{MIN_AVAILABLE}_{datetime.today().strftime('d_%H%M')}"
START_DATE = datetime.today()
END_DATE = datetime(START_DATE.year + TIMEHORIZON_IN_YEARS, START_DATE.month + TIMEHORIZON_IN_MONTHS, START_DATE.day + TIMEHORIZON_IN_DAYS)
EXPORT_FOLDER = f"output/{RUN_NAME}"
os.makedirs(EXPORT_FOLDER, exist_ok=True)




# === HOOFDPROGRAMMA ===
print("📥 Inputdata laden...")
inputs = load_all_inputs(start_date=START_DATE, end_date=END_DATE, team_name=TEAM_NAME,GAMMA_SICK_DURATION = GAMMA_SICK_DURATION,
                    GAMMA_VACATION_DURATION = GAMMA_VACATION_DURATION,
                    GAMMA_TRAINING_DURATION = GAMMA_TRAINING_DURATION,
                    GAMMA_SICK_PROB = GAMMA_SICK_PROB,
                    GAMMA_VACATION_PROB = GAMMA_VACATION_PROB,
                    GAMMA_MUTATION_PROB = GAMMA_MUTATION_PROB)
mutation_prob = load_mutation_probability()

print("✅ Inputdata geladen.")

current_alfa_p = deepcopy(inputs["alfa_p"])
best_solution = None
best_alfa_p = deepcopy(current_alfa_p)
min_total_unqualified = float("inf")

overall_results = []
history = []
newcomer_counter = 1

for iteration in range(MAX_ITER):
    print(f"\n🚀 Iteratie {iteration+1}/{MAX_ITER}...")
    inputs["alfa_p"] = deepcopy(current_alfa_p)
    model, vars, _ = solve_workforce_model(prepared_inputs=inputs, MIPGAP= MIPGAP, NUM_SOLUTIONS = NUM_SOLUTIONS)
    solutions = parse_multiple_solutions(model, vars, inputs["r_p_k"], inputs["P"], inputs["K"])


    total_underqualified = defaultdict(int)
    solution_results = []

    for i, sol in enumerate(solutions):
        parsed_after_mutation, _ = simulate_mutation(
            parsed_inputs=deepcopy(inputs),
            solution=sol,
            mutation_probs={w: mutation_prob for w in inputs["W"]},
            start_date=START_DATE,
            end_date=END_DATE,
            export_folder=EXPORT_FOLDER,
            iteration=iteration,
            solution_index=i,
            employee_id_counter=newcomer_counter
        )
        newcomer_counter += 1

        ziekte_mapping = build_sickness_mapping(
            employee_file=r"C:\Users\olafs\OneDrive\Bureaublad\master's dissertation\variation and competences\AM_Data\Werknemers.xlsx",
            sim_file=r"C:\Users\olafs\OneDrive\Documenten\GitHub\Master-s-Dissertation-Olaf---Arcelor\Code\SimulatieResultaten.xlsx",
            employee_sheet="data",
            sickness_sheet="Ziekteanalyse",
            default_age=35
        )

        from Code.reality_module import simulate_full_sickness_effect  # voeg import toe indien nodig

        parsed_after_sickness, replacements, sickdays, missed_trainings, updated_fws = simulate_full_sickness_effect(
            parsed_after_mutation,
            inputs,
            ziekte_mapping,
            inputs["r_p_k"],
            inputs["Pm"],
            inputs["S"]
        )
        calendar_df = build_assignment_calendar(parsed_after_sickness)

        # 3️⃣ Kalender exporteren naar Excel
        export_styled_calendar_detailed(calendar_df, parsed_after_sickness, "output/sick_calendar.xlsx")
        calendar_df = generate_sickday_calendar_detailed(parsed_after_sickness, sickdays, replacements)
        export_styled_calendar_detailed(
            calendar_df,
            parsed_after_sickness,
            os.path.join(EXPORT_FOLDER, f"iter_{iteration + 1}_sol_{i}_calendar.xlsx")
        )

        export_sickness_logs(
            replacements_df=replacements,
            sickdays_dict=sickdays,
            missed_trainings_dict=missed_trainings,
            updated_fws_dict=updated_fws,
            export_path_prefix=os.path.join(EXPORT_FOLDER, f"iter_{iteration + 1}_sol_{i}")
        )

        underq_counts = extract_underqualified_counts(parsed_after_sickness, inputs["P_map"], sickdays)
        if "UnqualifiedCount" in underq_counts.columns:

            counts_dict = {
                row["Position"]: int(row["UnqualifiedCount"])
                for _, row in underq_counts.iterrows()
            }
        # --- Fix de mapping ---
            counts_dict_fixed = {}

            for pos_name, count in counts_dict.items():
                # Zoek bijbehorende numerieke positie
                p_found = None
                for p, name in inputs["P_map"].items():
                    if name == pos_name:
                        p_found = p
                        break
                if p_found is not None:
                    counts_dict_fixed[p_found] = count
                else:
                    print(f"⚠️ Positienaam '{pos_name}' niet gevonden in P_map!")

            # --- Gebruik daarna correcte dict ---
            for p, count in counts_dict_fixed.items():
                total_underqualified[p] += count



            solution_results.append({
                "iteration": iteration + 1,
                "solution_index": i,
                "solution": parsed_after_sickness,
                "underqualified_total": sum(counts_dict.values())
            })

        total_count = sum(total_underqualified.values())
    else:
        total_count = 0
    print(f"📉 Totale underqualified: {total_count}")

    if total_count == 0:
        print("⚠️ Geen enkele underqualified toewijzing gevonden. Gewichten worden niet aangepast.")
        new_alfa_p = deepcopy(current_alfa_p)
    else:
        # Genormaliseerd naar totaal 100
        # 1. Maak gewogen som (géén schaalvervorming!)

        print("\n🔎 Debugging alpha recalculation...")
        print("▶️ Current alpha_p before update:")
        for p in sorted(current_alfa_p):
            print(f"  Positie {p}: {current_alfa_p[p]:.4f}")

        raw_scores = {
            p: 0.5 * current_alfa_p.get(p, 0) + 0.5 * (total_underqualified.get(p, 0) / total_count) * 100
            for p in current_alfa_p
        }

        print("\n📈 Raw weighted scores (before normalization):")
        for p in sorted(raw_scores):
            print(f"  Positie {p}: {raw_scores[p]:.4f}")

        total_raw = sum(raw_scores.values())
        new_alfa_p = {p: (v / total_raw) * 100 for p, v in raw_scores.items()}

        print("\n🛠️ New normalized alpha_p after update:")
        for p in sorted(new_alfa_p):
            print(f"  Positie {p}: {new_alfa_p[p]:.4f}")
    print("\n🔎 Gedetailleerde recalculatie per positie:")
    for p in sorted(current_alfa_p.keys()):
        old = current_alfa_p.get(p, 0)
        unq = total_underqualified.get(p, 0)
        raw = 0.5 * old + 0.5 * (unq / total_count) * 100 if total_count > 0 else 0
        new = (raw / sum(
            0.5 * current_alfa_p.get(pp, 0) + 0.5 * (total_underqualified.get(pp, 0) / total_count) * 100 for pp in
            current_alfa_p)) * 100 if total_count > 0 else old
        print(f"  Positie {p:2d} | Underqualified: {unq:4d} | Vorige alfa: {old:6.2f} | Nieuwe alfa: {new:6.2f}")

    delta = sum(abs(new_alfa_p[p] - current_alfa_p[p]) for p in current_alfa_p)
    print(f"🔁 Gewichtsverandering: {delta:.4f}")
    history.append({
        "iteration": iteration + 1,
        "delta": delta,
        "alfa_p": deepcopy(current_alfa_p),
        "total": total_count  # ← deze extra regel zorgt dat de plot werkt
    })

    for res in solution_results:
        overall_results.append(res)
        if res["underqualified_total"] < min_total_unqualified:
            min_total_unqualified = res["underqualified_total"]
            best_solution = deepcopy(res["solution"])
            best_alfa_p = deepcopy(new_alfa_p)

    if delta < TOLERANCE:
        print("✅ Convergentie bereikt. Stoppen.")
        break

    current_alfa_p = new_alfa_p

# === EXPORT ===
print("\n💾 Resultaten exporteren...")

for res in overall_results:
    export_solution_to_excel(res["solution"], f"{EXPORT_FOLDER}/iter_{res['iteration']}_solution_{res['solution_index']}.xlsx")

pd.DataFrame(overall_results)[["iteration", "solution_index", "underqualified_total"]].to_excel(
    f"{EXPORT_FOLDER}/summary_underqualified_all_iterations.xlsx", index=False
)
pd.DataFrame(history).to_excel(f"{EXPORT_FOLDER}/optimization_history.xlsx", index=False)
export_solution_to_excel(best_solution, os.path.join(EXPORT_FOLDER, "best_overall_solution.xlsx"))
pd.DataFrame.from_dict(best_alfa_p, orient="index").to_excel(f"{EXPORT_FOLDER}/best_alfa_weights.xlsx")
from visualisation_module import plot_adaptive_optimization_progress

plot_adaptive_optimization_progress(export_folder=EXPORT_FOLDER)
from solution_utils import extract_position_qualification_timeline, export_qualification_timeline
if "K_p" not in inputs:
    inputs["K_p"] = {p: [k for (pp, k) in inputs["r_p_k"] if pp == p] for p in inputs["P"]}

timeline = extract_position_qualification_timeline(best_solution, inputs["P_map"], inputs["K_p"])

export_qualification_timeline(timeline, f"{EXPORT_FOLDER}/qualification_timeline.xlsx")

print("\n🏁 Optimalisatie met mutaties afgerond.")
print(f"🧠 Beste oplossing: totaal underqualified = {min_total_unqualified}")
'''
def adaptive_multi_solution_optimizer(
    start_date,
    end_date,
    initial_alfa_p,
    ziekte_mapping,
    r_p_k,
    P_map,
    Pm,
    S,
    max_iterations=10,
    tolerance=1,
    export_folder="output/adaptive",
    num_solutions=10,
    num_iterations=1000
):
    """
    Adaptive optimizer that generates multiple solutions per iteration and simulates them for sickness,
    then updates the alfa_p weights based on the aggregate underqualified assignments across all solutions.

    Parameters:
    - start_date, end_date: start and end dates for the simulation.
    - initial_alfa_p: initial weights per position (alfa_p).
    - ziekte_mapping: mapping of workers to sickness probabilities.
    - r_p_k: required skills per position.
    - P_map, Pm: position mappings.
    - S: set of shifts.
    - max_iterations: maximum number of iterations for optimization.
    - tolerance: convergence threshold for stopping optimization.
    - export_folder: folder to export results.
    - num_solutions: number of solutions to generate per iteration.
    - num_iterations: number of sickness simulations per solution.
    """

    os.makedirs(export_folder, exist_ok=True)

    best_solution = None
    best_alfa_p = deepcopy(initial_alfa_p)
    current_alfa_p = deepcopy(initial_alfa_p)
    min_total_unqualified = float("inf")
    previous_solution = None

    history = []

    for iteration in range(max_iterations):
        print(f"\n🔁 Iteration {iteration + 1} started...")

        inputs = load_all_inputs(start_date=start_date, end_date=end_date)
        inputs["alfa_p"] = deepcopy(current_alfa_p)

        model, vars, status = solve_workforce_model(
            prepared_inputs=inputs,
            warm_start_solution=previous_solution  # ✅ warm start toegevoegd
        )

        parsed_solutions = parse_multiple_solutions(model, vars)


        all_counts = []
        for i, solution in enumerate(parsed_solutions):
            print(f"🧬 Mutatiesimulatie + ziekte voor oplossing {i + 1}...")

            export_base = f"{export_folder}/sickness_solution_{iteration + 1}_{i + 1}"

            parsed_final, sickness_df, new_id = simulate_mutation(
                parsed_inputs=inputs,
                solution=solution,
                mutation_probs={w: 0.000057 for w in inputs["W"]},  # jouw mutatiekans
                start_date=start_date,
                end_date=end_date,
                export_folder=export_base,
                iteration=iteration,
                solution_index=i,
                employee_id_counter=i + 1
            )
            underqualified_counts = extract_underqualified_counts(solution, P_map, sickness_df)
            all_counts.append(underqualified_counts)

        total_underqualified = defaultdict(int)
        for counts in all_counts:
            for pos, count in counts.items():
                total_underqualified[pos] += int(count)

        total_count = sum(total_underqualified.values())
        print(f"📉 Total underqualified across all solutions: {total_count}")

        weight_previous = 0.5
        weight_underqualified = 1 - weight_previous
        new_alfa_p = {
            p: (weight_previous * current_alfa_p.get(p, 0) + weight_underqualified * (
                        total_underqualified.get(p, 0) / total_count) * 100)
            for p in current_alfa_p
        }

        delta = sum(abs(new_alfa_p[p] - current_alfa_p[p]) for p in current_alfa_p)
        print(f"Δ in weights = {delta:.4f}")

        if total_count < min_total_unqualified:
            min_total_unqualified = total_count
            best_solution = deepcopy(parsed_solutions[0])  # verbeter dit eventueel op basis van ranking
            best_alfa_p = deepcopy(new_alfa_p)

        if delta < tolerance:
            print("✅ Convergence reached. Stopping iterations.")
            break

        current_alfa_p = new_alfa_p

        if parsed_solutions:
            previous_solution = parsed_solutions[0]  # gebruik de eerste oplossing als warm start

        export_solution_to_excel(best_solution, os.path.join(export_folder, "best_solution.xlsx"))
        history.append({
            "iteration": iteration + 1,
            "delta": delta,
            "total": total_count,
            "alfa_p": str(current_alfa_p)  # naar string om opslaan in Excel mogelijk te maken
        })

        # Save progress after each iteration
        pd.DataFrame(history).to_excel(os.path.join(export_folder, "optimization_history.xlsx"), index=False)

    save_alfa_p_weights_to_excel(best_alfa_p, export_path="alfa_weights.xlsx")

    return best_solution, best_alfa_p
