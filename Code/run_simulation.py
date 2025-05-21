from datetime import datetime
import os
import pandas as pd
import numpy as np
import time
from collections import defaultdict
from copy import deepcopy
from Code.input_module import load_all_inputs, build_sickness_mapping
from Code.solver_module import solve_workforce_model
from Code.solution_utils import (
    parse_multiple_solutions,
    export_solution_to_excel,
    extract_underqualified_counts,
    extract_position_qualification_timeline,
    export_qualification_timeline,
    calculate_tfc_at_timepoints,
    generate_sickday_calendar_detailed,
    export_styled_calendar_detailed, check_training_feasibility, export_all_inputs_to_excel
)
from Code.simulator_module import build_assignment_calendar
from Code.reality_module import simulate_full_sickness_effect
from mutation_module import simulate_mutation
from visualisation_module import plot_adaptive_optimization_progress
from dateutil.relativedelta import relativedelta

def run_adaptive_simulation(
    team_name: str = "Ploeg D",
    mipgap: float = 0.2,
    sickness_iterations: int = 100,
    num_solutions: int = 200,
    beta: float = 100,
    recalibration_weight: float = 0.5,
    tolerance: float = 1.0,
    max_iter: int = 10,
    min_available: int = 8,
    gamma_sick_duration: float = 1.0,
    gamma_vacation_duration: float = 1.0,
    gamma_training_duration: float = 1.0,
    gamma_sick_prob: float = 1.0,
    gamma_vacation_prob: float = 1.0,
    gamma_mutation_prob: float = 1.0,
    timehorizon_years: int = 0,
    timehorizon_months: int = 6,
    timehorizon_days: int = 0,
    export_folder: str = None,
    run_name: str = None,
    export_enabled: bool = False
) -> dict:

    start_time = time.time()
    start_date = datetime.today()
    from datetime import timedelta
    from dateutil.relativedelta import relativedelta

    end_date = start_date + relativedelta(years=timehorizon_years, months=timehorizon_months) + timedelta(
        days=timehorizon_days)

    if run_name is None:
        run_name = f"simulation_{team_name}_{timehorizon_years}Y_{timehorizon_months}M_G{mipgap}_MA{min_available}_{start_date.strftime('d_%H%M')}"
    if export_folder is None:
        export_folder = f"output/{run_name}"
    os.makedirs(export_folder, exist_ok=True)

    print("\U0001F4E5 Inputdata laden...")
    inputs = load_all_inputs(
        start_date=start_date,
        end_date=end_date,
        team_name=team_name,
        GAMMA_SICK_DURATION=gamma_sick_duration,
        GAMMA_VACATION_DURATION=gamma_vacation_duration,
        GAMMA_TRAINING_DURATION=gamma_training_duration,
        GAMMA_SICK_PROB=gamma_sick_prob,
        GAMMA_VACATION_PROB=gamma_vacation_prob,
        GAMMA_MUTATION_PROB=gamma_mutation_prob
    )
    mutation_prob = 0.000180127
    print("✅ Inputdata geladen.")
    W = inputs["W"]
    current_alfa_p = deepcopy(inputs["alfa_p"])
    best_solution = None
    best_alfa_p = deepcopy(current_alfa_p)
    min_total_underqualified = float("inf")
    overall_results = []
    history = []
    newcomer_counter = 1
    avg_workers_per_shift = []
    all_training_progress = []
    all_rescheduling_types = []
    all_tfc_results = []
    all_uncaught_trainings = []
    export_all_inputs_to_excel(inputs)

    for iteration in range(max_iter):
        print(f"\n🚀 Iteratie {iteration+1}/{max_iter}...")
        inputs["alfa_p"] = deepcopy(current_alfa_p)
        model, vars, _ = solve_workforce_model(prepared_inputs=inputs, MIPGAP=mipgap, NUM_SOLUTIONS=num_solutions, beta=beta)
        solutions = parse_multiple_solutions(model, vars, inputs["r_p_k"], inputs["P"], inputs["K"])

        total_underqualified = defaultdict(int)
        solution_results = []

        for i, sol in enumerate(solutions):
            parsed_after_mutation, _ = simulate_mutation(
                parsed_inputs=deepcopy(inputs),
                solution=sol,
                mutation_probs={w: mutation_prob for w in inputs["W"]},
                start_date=start_date,
                end_date=end_date,
                export_folder=export_folder,
                iteration=iteration,
                solution_index=i,
                employee_id_counter=newcomer_counter
            )
            newcomer_counter += 1

            tfc_results = calculate_tfc_at_timepoints(
                parsed_solution=sol,
                P_map=inputs["P_map"],
                K_p=inputs["K_p"],
                S=inputs["S"],
                Pm=inputs["Pm"],
                f_w_s=inputs["f_w_s"],
                W=inputs["W"]
            )
            all_tfc_results.append({
                "iteration": iteration + 1,
                "solution_index": i,
                "tfc_results": tfc_results,
                "avg_tfc": np.mean([v["tfc"] for v in tfc_results.values()]) if tfc_results else 0
            })

            ziekte_mapping = build_sickness_mapping(
                employee_file=r"C:\\Users\\olafs\\OneDrive\\Bureaublad\\master's dissertation\\variation and competences\\AM_Data\\Werknemers.xlsx",
                sim_file=r"C:\\Users\\olafs\\OneDrive\\Documenten\\GitHub\\Master-s-Dissertation-Olaf---Arcelor\\Code\\SimulatieResultaten.xlsx",
                employee_sheet="data",
                sickness_sheet="Ziekteanalyse",
                default_age=35
            )

            sim_underq_per_position = defaultdict(int)
            sim_underq_per_worker = defaultdict(int)

            for _ in range(sickness_iterations):
                parsed_after_sickness, replacements, sickdays, missed_trainings, updated_fws, uncaught_trainings = simulate_full_sickness_effect(
                    parsed_after_mutation,
                    inputs,
                    ziekte_mapping,
                    inputs["r_p_k"],
                    inputs["Pm"],
                    inputs["S"]
                )

                all_training_progress.append(missed_trainings)
                all_uncaught_trainings.extend(uncaught_trainings)

                underq_counts = extract_underqualified_counts(parsed_after_sickness, inputs["P_map"], sickdays)
                if "UnqualifiedCount" in underq_counts.columns:
                    for _, row in underq_counts.iterrows():
                        pos_name = row["Position"]
                        count = int(row["UnqualifiedCount"])
                        for p, name in inputs["P_map"].items():
                            if name == pos_name:
                                sim_underq_per_position[p] += count

                    for (w, p, s), val in parsed_after_sickness["x"].items():
                        if val > 0.5 and w in sickdays and s in sickdays[w]:
                            sim_underq_per_worker[w] += 1

            # Speciale logica voor p=1 en p=2
            p1 = 1
            p2 = 2
            if p1 in sim_underq_per_position and p2 in sim_underq_per_position:
                avg_count = (sim_underq_per_position[p1] + sim_underq_per_position[p2]) / 2
                total_underqualified[p1] += avg_count
                total_underqualified[p2] += avg_count
            else:
                # fallback: als slechts één van beide bestaat, neem gewone waarde
                if p1 in sim_underq_per_position:
                    total_underqualified[p1] += sim_underq_per_position[p1]
                if p2 in sim_underq_per_position:
                    total_underqualified[p2] += sim_underq_per_position[p2]

            # Andere posities (behalve 1 en 2) normaal toevoegen
            for p, count in sim_underq_per_position.items():
                if p not in [p1, p2]:
                    total_underqualified[p] += count

            solution_results.append({
                "iteration": iteration + 1,
                "solution_index": i,
                "solution": sol,
                "underqualified_total": sum(sim_underq_per_position.values()),
                "underq_per_position": dict(sim_underq_per_position),
                "underq_per_worker": dict(sim_underq_per_worker)
            })

            worker_count = [sum(1 for w in W for p in inputs["Pm"] if sol["x"].get((w, p, s), 0) > 0) for s in inputs["S"]]
            avg_workers_per_shift.append(np.mean(worker_count))
            all_rescheduling_types.append(
                replacements["Type"].value_counts().to_dict() if not replacements.empty else {})

            if export_enabled:
                calendar_df = generate_sickday_calendar_detailed(
                    parsed_solution=parsed_after_sickness,
                    sickdays=sickdays,
                    replacements=replacements
                )
                export_styled_calendar_detailed(
                    calendar_df=calendar_df,
                    parsed_solution=parsed_after_sickness,
                    file_path=os.path.join(export_folder, f"calendar_iter{iteration+1}_sol{i+1}.xlsx")
                )

                qual_timeline = extract_position_qualification_timeline(
                    parsed_solution=parsed_after_sickness,
                    P_map=inputs["P_map"],
                    K_p=inputs["K_p"]
                )
                export_qualification_timeline(
                    timeline=qual_timeline,
                    output_path=os.path.join(export_folder, f"qual_timeline_iter{iteration+1}_sol{i+1}.xlsx")
                )

        total_count = sum(total_underqualified.values())
        print(f"📉 Totale underqualified: {total_count}")

        if total_count == 0:
            print("⚠️ Geen enkele underqualified toewijzing gevonden. Gewichten worden niet aangepast.")
            new_alfa_p = deepcopy(current_alfa_p)
        else:
            raw_scores = {
                p: recalibration_weight * current_alfa_p.get(p, 0) + (1 - recalibration_weight) * (total_underqualified.get(p, 0) / total_count) * 100
                for p in current_alfa_p
            }
            total_raw = sum(raw_scores.values())
            new_alfa_p = {p: (v / total_raw) * 100 for p, v in raw_scores.items()}

        delta = sum(abs(new_alfa_p[p] - current_alfa_p[p]) for p in current_alfa_p)
        print(f"🔁 Gewichtsverandering: {delta:.4f}")

        history.append({
            "iteration": iteration + 1,
            "delta": delta,
            "alfa_p": deepcopy(current_alfa_p),
            "total": total_count,
        })

        for res in solution_results:
            overall_results.append(res)
            if res["underqualified_total"] < min_total_underqualified:
                min_total_underqualified = res["underqualified_total"]
                best_solution = deepcopy(res["solution"])
                best_alfa_p = deepcopy(new_alfa_p)

        if delta < tolerance:
            print("✅ Convergentie bereikt. Stoppen.")
            break

        current_alfa_p = new_alfa_p

    total_runtime = time.time() - start_time

    if export_enabled:
        pd.DataFrame(history).to_excel(os.path.join(export_folder, "optimization_history.xlsx"), index=False)
        pd.DataFrame(overall_results)[["iteration", "solution_index", "underqualified_total"]].to_excel(
            os.path.join(export_folder, "summary_underqualified_all_iterations.xlsx"), index=False)
        export_solution_to_excel(best_solution, os.path.join(export_folder, "best_overall_solution.xlsx"))
        pd.DataFrame.from_dict(best_alfa_p, orient="index").to_excel(
            os.path.join(export_folder, "best_alfa_weights.xlsx"))
        plot_adaptive_optimization_progress(export_folder=export_folder)

    best_tfc = calculate_tfc_at_timepoints(
        parsed_solution=best_solution,
        P_map=inputs["P_map"],
        K_p=inputs["K_p"],
        S=inputs["S"],
        Pm=inputs["Pm"],
        f_w_s=inputs["f_w_s"],
        W=inputs["W"]
    ) if best_solution and "x" in best_solution else None

    print("\n🏁 Optimalisatie met mutaties afgerond.")
    print(f"🧠 Beste oplossing: totaal underqualified = {min_total_underqualified}")

    training_counts_per_worker_skill = defaultdict(int)
    for missed in all_training_progress:
        for w, missed_pairs in missed.items():
            for (p, s) in missed_pairs:
                training_counts_per_worker_skill[(w, p)] += 1

    underq_per_simulation_variance = np.var([
        sim["underqualified_total"]
        for sim in overall_results
        if isinstance(sim, dict) and "underqualified_total" in sim
    ]) if overall_results else None

    underq_totals_per_iter = [res["underqualified_total"] for res in overall_results]
    underq_iter_variance = np.var(underq_totals_per_iter) if underq_totals_per_iter else None

    # Bepaal eerste planning zonder mutaties of ziekte
    initial_underqualified = solution_results[0]["underqualified_total"] if solution_results else None

    # Bepaal gemiddelde underqualified na mutatiesimulatie (voor ziekte)
    underqualified_post_mutation = np.mean([
        res["underqualified_total"] for res in solution_results
    ]) if solution_results else None

    # Bepaal het aantal gevolgde trainingen per (w,p)
    trainings_followed_total = sum(
        1 for (w, p, s), val in best_solution.get("q", {}).items() if val >= 0.5
    ) if best_solution and "q" in best_solution else 0

    # Bepaal TFC na 3 en 6 maanden (in shifts = dag * 0.75 * 90 resp. 180)
    tfc_at_3_months = best_tfc.get(90, {}).get("tfc", 0)  # 3m ≈ 90d × 0.75 = 67.5 → 68
    tfc_at_6_months = best_tfc.get(135, {}).get("tfc", 0)  # 6m ≈ 180d × 0.75 = 135

    return {
        "beta": beta,
        "final_underqualified": min_total_underqualified,
        "iterations": iteration + 1,
        "converged": delta < tolerance,
        "final_alfa_p": best_alfa_p,
        "run_name": run_name,
        "export_folder": export_folder,
        "avg_workers_per_shift": np.mean(avg_workers_per_shift) if avg_workers_per_shift else None,
        "training_completion_log": all_training_progress,
        "rescheduling_types": all_rescheduling_types,
        "runtime_seconds": total_runtime,
        "uncaught_trainings": all_uncaught_trainings,
        "tfc_results": all_tfc_results,
        "best_solution_tfc": best_tfc,
        "training_counts_per_worker_skill": training_counts_per_worker_skill,
        "variance_underq_fixed": underq_per_simulation_variance,
        "variance_underq_dynamic": underq_iter_variance,
        "final_solution": best_solution,
        "avg_tfc_after_3months": tfc_at_3_months,
        "avg_tfc_after_6months": tfc_at_6_months,
        "avg_rescheduled_assignments": np.mean([
            sum(res_type.values()) for res_type in all_rescheduling_types
        ]) if all_rescheduling_types else None,
        "avg_underqualified_first_scheduling": initial_underqualified,
        "avg_underqualified_post_mutation": underqualified_post_mutation,
        "avg_underqualified_post_sickness": min_total_underqualified,
        "avg_trainings_followed": trainings_followed_total,
        "avg_runtime_seconds": total_runtime,
        "inputs": inputs
    }

if __name__ == "__main__":
    result = run_adaptive_simulation(
        team_name="Ploeg D",
        timehorizon_years=1,
        timehorizon_months=0,
        timehorizon_days=0,
        max_iter=10,
        tolerance=1.0,
        mipgap=0.1,
        beta=200,
        export_enabled=True,
        sickness_iterations=1000,
        min_available=8,
        num_solutions=200
    )

    export_folder = result["export_folder"]

    # 📈 Trainings per werknemer/skill (C)
    df_train = pd.DataFrame([
        {"Worker": w, "Position": p, "MissedTrainings": v}
        for (w, p), v in result["training_counts_per_worker_skill"].items()
    ])
    df_train.to_excel(os.path.join(export_folder, "metric_C_training_counts.xlsx"), index=False)

    # 📉 Variantie (D/E)
    pd.DataFrame([
        {"Metric": "Variance Under Sickness", "Value": result["variance_underq_fixed"]},
        {"Metric": "Variance Across Iterations", "Value": result["variance_underq_dynamic"]}
    ]).to_excel(os.path.join(export_folder, "metric_DE_variance.xlsx"), index=False)

    # 📌 TFC metrics (G)
    pd.DataFrame([
        {"Timepoint": "6_months", "TFC": result["tfc_at_90"]},
        {"Timepoint": "end", "TFC": result["tfc_at_end"]}
    ]).to_excel(os.path.join(export_folder, "metric_G_tfc_overview.xlsx"), index=False)

    print(f"\n✅ Alle output opgeslagen in: {export_folder}")