from Code.reality_module import simulate_vacation_days, simulate_full_sickness_effect
from input_module import load_simulation_parameters, build_sickness_mapping, export_f_w_s_to_excel
from solver_module import solve_workforce_model
from solution_utils import parse_multiple_solutions
from copy import deepcopy
import numpy as np
from datetime import timedelta
import os
from Code.solution_utils import generate_sickday_calendar_detailed, export_styled_calendar_detailed


def simulate_mutation(parsed_inputs, solution, mutation_probs, start_date, end_date, export_folder, iteration, solution_index, employee_id_counter, max_mutations=1 , depth=0, min_available=8, lost_leavedays = 0):

  # 1. Mutatie bepalen
  W_full = parsed_inputs["W"]
  S_original = parsed_inputs["S"]
  shift_per_day = 0.75

  departing_worker = None
  mutation_shift = None
  mutation_probs = {w: 0.000180127 for w in W_full}

  for s in S_original:
      for w in W_full:
          if np.random.rand() < mutation_probs.get(w, 0.0):
              departing_worker = w
              mutation_shift = s
              break
      if departing_worker:
          break

  if departing_worker is None:
      print(f"\n🔁 Geen mutatie voor oplossing {solution_index + 1} – niemand vertrekt.")
      return solution, None

  mutation_day = int(mutation_shift / shift_per_day)
  mutation_date = start_date + timedelta(days=mutation_day)
  print(
      f"\n🔁 Mutatie voor oplossing {solution_index + 1} - werknemer {departing_worker} vertrekt op shift {mutation_shift} ({mutation_date.strftime('%Y-%m-%d')})")

  # 2. Inputs kopiëren
  inputs_mutated = deepcopy(parsed_inputs)

  # 3. Tijdshorizon aanpassen vanaf mutatieshift
  S_mut = [s for s in S_original if s >= mutation_shift]
  inputs_mutated["S"] = S_mut

  # 4. Vertrekkende werknemer onbeschikbaar maken
  for s in S_mut:
      inputs_mutated["f_w_s"][departing_worker][s] = 1

  # 5. Nieuwe werknemer toevoegen
  new_id = 10000000 + employee_id_counter
  days_later = 90
  new_start_shift = int((mutation_day + days_later) * shift_per_day)

  inputs_mutated["W"].append(new_id)
  inputs_mutated["I_wks"][new_id] = {k: {0: 0} for k in inputs_mutated["K"]}
  inputs_mutated.setdefault("I_expertise", {})[new_id] = {k: {0: 0} for k in inputs_mutated["K"]}

  sim_params = load_simulation_parameters("SimulatieResultaten.xlsx")
  verlof_targets = {new_id: round((end_date - start_date).days * 0.2)}
  combined_W = [w for w in inputs_mutated["W"] if w not in {departing_worker, new_id}]
  leave_targets_updated = {
      w: round((end_date - start_date).days * 0.2)  # of gebruik originele targets als beschikbaar
      for w in combined_W
  }
  _, _, shift_leave_map = simulate_vacation_days(
      W=combined_W,
      start_date=start_date,
      end_date=end_date,
      leave_targets=leave_targets_updated,
      month_probs=sim_params["month_probs"],
      mean_leave_duration=sim_params["verlofduur"],
      min_available=min_available,
      vacation_sampler=parsed_inputs["vacation_sampler"]
  )

  for w in combined_W:
      for s in S_mut:
          if s in shift_leave_map.get(w, []):
              inputs_mutated["f_w_s"][w][s] = 1  # op verlof
          else:
              inputs_mutated["f_w_s"][w][s] = 0  # beschikbaar

  # 8. Warm start opbouwen op basis van toestand vóór mutatie
  s0 = mutation_shift - 1
  warm_start = {
      "t": {(w, k, 0): solution["t"].get((w, k, s0), 0.0) for w in inputs_mutated["W"] for k in inputs_mutated["K"]},
      "v": {(w, k, 0): solution["v"].get((w, k, s0), 0.0) for w in inputs_mutated["W"] for k in inputs_mutated["K"]},
      "h": {(w, k, 0): 1.0 if solution["t"].get((w, k, s0), 0.0) >= inputs_mutated["L_k"].get(k, 1) else 0.0
            for w in inputs_mutated["W"] for k in inputs_mutated["K"]},
      "r": {(w, k, 0): 1.0 if solution["v"].get((w, k, s0), 0.0) >= 32 else 0.0
            for w in inputs_mutated["W"] for k in inputs_mutated["K"]},
  }
  # Zorg dat alle f_w_s[w] volledige shiftdekking heeft voor S
  for w in inputs_mutated["W"]:
      for s in inputs_mutated["S"]:
          if s not in inputs_mutated["f_w_s"].get(w, {}):
              inputs_mutated["f_w_s"].setdefault(w, {})[s] = 0

  inputs_mutated["S"] = [s for s in inputs_mutated["f_w_s"][inputs_mutated["W"][0]].keys()]
  for w in inputs_mutated["W"]:
      for s in inputs_mutated["S"]:
          inputs_mutated["f_w_s"].setdefault(w, {})[s] = inputs_mutated["f_w_s"][w].get(s, 0)
  for w in inputs_mutated["W"]:
      if set(inputs_mutated["S"]) - set(inputs_mutated["f_w_s"].get(w, {})) != set():
          print(f"⚠️ Waarschuwing: werknemer {w} heeft onvolledige f_w_s.")
          for w in inputs_mutated["W"]:
              missing = set(inputs_mutated["S"]) - set(inputs_mutated["f_w_s"].get(w, {}).keys())
              if missing:
                  print(f"⚠️ Werknemer {w} mist shifts in f_w_s: {sorted(missing)}")
                  raise ValueError("f_w_s coverage mismatch")
  assert len(inputs_mutated["W"]) == len(set(inputs_mutated["W"])), \
      f"❌ Duplicaat-ID's in W: {[(w, inputs_mutated['W'].count(w)) for w in inputs_mutated['W'] if inputs_mutated['W'].count(w) > 1]}"

  # 9. Oplossen
  model, vars, _ = solve_workforce_model(
      start_date=start_date + timedelta(days=mutation_day),
      end_date=end_date,
      prepared_inputs=inputs_mutated,
      warm_start_solution=warm_start
  )

  parsed_final = parse_multiple_solutions(
      model,
      vars,
      inputs_mutated["r_p_k"],
      inputs_mutated["P"],
      inputs_mutated["K"]
  )[0]
  if depth + 1 < max_mutations:
      employee_id_counter += 1

      return simulate_mutation(
          parsed_inputs=inputs_mutated,
          solution=parsed_final,
          mutation_probs=mutation_probs,
          start_date=start_date + timedelta(days=mutation_day),
          end_date=end_date,
          export_folder=export_folder,
          iteration=iteration,
          solution_index=solution_index,
          employee_id_counter=employee_id_counter,
          max_mutations=max_mutations,
          depth=depth + 1,
          lost_leavedays = lost_leavedays
      )
  else:
      return parsed_final, employee_id_counter
