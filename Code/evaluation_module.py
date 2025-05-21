import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from reality_module import simulate_full_sickness_effect
from solution_utils import (
    extract_underqualified_counts,
    generate_sickday_calendar_detailed,
    export_styled_calendar_detailed,
    export_sickness_logs, analyze_training_recovery
)

def run_sickness_simulation(parsed_solution, P_map, ziekte_mapping, r_p_k, Pm, S, num_iterations=1000, export_path=None) -> pd.DataFrame:
    all_counts = []
    last_calendar = None

    for i in range(num_iterations):
        updated_solution, replacements, sickdays, missed_trainings, updated_fws = simulate_full_sickness_effect(
            parsed_solution, ziekte_mapping, r_p_k, Pm, S
        )

        df = extract_underqualified_counts(updated_solution, P_map, sickdays)

        if "Position" not in df.columns or "UnqualifiedCount" not in df.columns:
            continue

        count_dict = df.set_index("Position")["UnqualifiedCount"].to_dict()
        all_counts.append(count_dict)

        if i == num_iterations - 1 and export_path:
            after_calendar = generate_sickday_calendar_detailed(updated_solution, sickdays, replacements)
            export_styled_calendar_detailed(after_calendar, updated_solution, export_path + "_calendar.xlsx")
            export_styled_calendar_detailed(after_calendar, updated_solution,
                                            export_path + "_after_sickness_planning_detailed.xlsx")
        #  Analyseer of gemiste trainingen zijn ingehaald
        recovery_df = analyze_training_recovery(missed_trainings, updated_solution["y"])
        recovery_df.to_excel(export_path + "_recovery_analysis.xlsx", index=False)

    summary_df = pd.DataFrame(all_counts).fillna(0).astype(int)

    if export_path and not summary_df.empty:
        mean_per_position = summary_df.mean(axis=0).sort_values(ascending=False)
        plt.figure(figsize=(10, 5))
        mean_per_position.plot(kind='bar')
        plt.title("Gemiddeld aantal underqualified toewijzingen per positie")
        plt.ylabel("Gemiddeld aantal (over simulaties)")
        plt.tight_layout()
        plt.savefig(export_path + "_summary_chart.png")
        plt.close()
    return summary_df



def evaluate_multiple_solutions(parsed_solutions: list, P_map: dict, ziekte_mapping: dict, r_p_k: dict, Pm: list, S: list, num_iterations=1000) -> dict:
    results = {}
    for i, solution in enumerate(parsed_solutions):
        print(f"🧪 Simuleer ziektes voor oplossing {i+1}...")
        export_base = f"output/sickness_solution_{i+1}"
        df = run_sickness_simulation(solution, P_map, ziekte_mapping, r_p_k, Pm, S, num_iterations, export_path=export_base)
        results[i] = df
    return results

def summarize_underqualified_counts_across_runs(simulation_outputs: list[dict], P_map: dict) -> pd.DataFrame:
    counts = defaultdict(int)
    for parsed_sol in simulation_outputs:
        df = extract_underqualified_counts(parsed_sol, P_map)
        for _, row in df.iterrows():
            counts[row["Position"]] += row["UnqualifiedCount"]

    df_summary = pd.DataFrame([
        {"Position": p, "TotalUnqualified": c, "AveragePerRun": c / len(simulation_outputs)}
        for p, c in counts.items()
    ])
    df_summary.sort_values(by="TotalUnqualified", ascending=False, inplace=True)
    return df_summary

def export_simulation_results_to_excel(results_dict: dict, output_path: str):
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for i, df in results_dict.items():
            df.to_excel(writer, sheet_name=f"Solution_{i+1}", index=False)

def export_summary_per_solution(results_dict: dict, P_map: dict, output_path: str):
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for i, df in results_dict.items():
            total_per_pos = df.sum(axis=0).sort_values(ascending=False)
            summary_df = pd.DataFrame({
                "Position": total_per_pos.index,
                "TotalUnderqualified": total_per_pos.values,
                "AveragePerRun": (total_per_pos / len(df)).values
            })
            summary_df.to_excel(writer, sheet_name=f"Summary_{i+1}", index=False)
