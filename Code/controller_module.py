def should_trigger_timebased(
    last_trigger_shift: int,
    current_shift: int,
    frequency: int
) -> bool:
    """
    Bepaal of de solver opnieuw moet worden aangeroepen op basis van tijd (periodiek).

    Parameters:
    - last_trigger_shift: laatst geplande shift waarop solver werd aangeroepen
    - current_shift: huidige shift in simulatie
    - frequency: herplanfrequentie in shifts

    Returns:
    - True als herplanning nodig is, anders False
    """
    return (current_shift - last_trigger_shift) >= frequency
def evaluate_controller(
    sim_state: dict,
    last_trigger_shift: int,
    current_shift: int,
    config: dict
) -> bool:
    """
    Evaluatie of de solver moet worden herroepen, op basis van controllerconfiguratie.

    Parameters:
    - sim_state: dictionary met simulatiestatus (bv. {"understaffed_ratio": 0.18})
    - last_trigger_shift: laatst herplande shift
    - current_shift: huidige shift
    - config: dict met controllerparameters:
        {
            "mode": "time" | "event" | "hybrid",
            "freq": int,
            "threshold": float
        }

    Returns:
    - True indien herplanning noodzakelijk
    """
    def should_trigger_eventbased(sim_state: dict, threshold: float) -> bool:
        return sim_state.get("understaffed_ratio", 0) > threshold

    mode = config.get("mode", "time")
    freq = config.get("freq", 10)
    threshold = config.get("threshold", 0.2)

    if mode == "time":
        return should_trigger_timebased(last_trigger_shift, current_shift, freq)
    elif mode == "none":
        return False

    elif mode == "event":
        return should_trigger_eventbased(sim_state, threshold)
    elif mode == "hybrid":
        return (
            should_trigger_timebased(last_trigger_shift, current_shift, freq) or
            should_trigger_eventbased(sim_state, threshold)
        )
    else:
        raise ValueError(f"❌ Onbekende controller mode: {mode}")
from controller_module import evaluate_controller

def run_controlled_simulation(
    initial_solution: dict,
    controller_config: dict,
    simulator_fn,
    solver_fn,
    parsed_inputs: dict,
    verbose: bool = True
) -> dict:
    """
    Voert een gesimuleerde shift-per-shift planning uit met herplanning via controller.

    Parameters:
    - initial_solution: parsed oplossing van initieel model
    - controller_config: dict met controllerinstellingen (mode, freq, threshold)
    - simulator_fn: functie(parsed_solution, current_shift) → sim_state
    - solver_fn: functie(start_date, end_date, ...) → parsed_solution
    - parsed_inputs: oorspronkelijke modelinput (voor hergebruik bij herplanning)
    - verbose: indien True, toont logging

    Returns:
    - dict met:
        - "solutions": list van parsed oplossingen per herplanning
        - "triggers": lijst van shifts waarop herplanning gebeurde
        - "history": simulatiestatussen per shift
    """

    current_solution = initial_solution
    last_trigger_shift = 0
    solutions = [current_solution]
    triggers = [0]
    history = {}

    S = parsed_inputs["S"]

    for current_shift in S:
        sim_state = simulator_fn(current_solution, current_shift)
        history[current_shift] = sim_state

        if evaluate_controller(sim_state, last_trigger_shift, current_shift, controller_config):
            if verbose:
                print(f"🔁 Herplanning op shift {current_shift}...")

            new_model, new_vars, _ = solver_fn(
                parsed_inputs["start_date"],
                parsed_inputs["end_date"],
                parsed_inputs["training_data"],
                parsed_inputs["competence_data"],
                parsed_inputs["vacation_plan"]
            )

            from solution_utils import parse_solution
            current_solution = parse_solution(new_model, new_vars)

            solutions.append(current_solution)
            triggers.append(current_shift)
            last_trigger_shift = current_shift

    return {
        "solutions": solutions,
        "triggers": triggers,
        "history": history
    }
def basic_simulator_understaffing(
    parsed_solution: dict,
    current_shift: int,
    required_positions: list[tuple[int, int]]
) -> dict:
    """
    Simuleer status voor één shift: bereken onderbezetting op verplichte posities.

    Parameters:
    - parsed_solution: output van parse_solution(...)
    - current_shift: shift waarvoor simulatie gebeurt
    - required_positions: lijst van tuples (s, p) waarvoor bezetting vereist is

    Returns:
    - dict met simulatiestatus, bijv.:
        {
            "understaffed_count": 3,
            "total_required": 12,
            "understaffed_ratio": 0.25
        }
    """
    assigned = parsed_solution.get("x", {})
    staffed_positions = {
        (w, p, s) for (w, p, s) in assigned if s == current_shift and assigned[(w, p, s)] > 0.5
    }

    understaffed = 0
    for s, p in required_positions:
        if s != current_shift:
            continue
        if not any(pp == p and ss == s for (_, pp, ss) in staffed_positions):
            understaffed += 1

    total_required = sum(1 for s, _ in required_positions if s == current_shift)
    ratio = understaffed / total_required if total_required > 0 else 0.0

    return {
        "understaffed_count": understaffed,
        "total_required": total_required,
        "understaffed_ratio": ratio
    }

