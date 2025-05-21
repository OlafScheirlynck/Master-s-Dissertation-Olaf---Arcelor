from datetime import datetime, timedelta
from gurobipy import GRB, Model

from Code.input_module import build_f_w_s_from_vacation_matrix


def prepare_model_inputs(
    training_data: dict,
    competence_data: dict,
    vacation_matrix: dict,
    start_date,
    num_shifts: int,
    alfa_p: dict[int, float],
    f_w_s: dict,
    min_available: int = 8

) -> dict:


    W = training_data["W"]
    K_map = competence_data["K_map"]
    P_map = competence_data["P_map"]
    r_p_k = competence_data["r_p_k"]

    used_skills = sorted(set(k for (_, k) in r_p_k))
    K_map = {k: v for k, v in K_map.items() if k in used_skills}
    K_reverse_map = {v: k for k, v in K_map.items()}
    K = list(K_map.keys())

    I_wks = {
        w: {k: training_data["I_wks"][w][k] for k in training_data["I_wks"][w] if k in K}
        for w in training_data["I_wks"]
    }
    experts = {
        w: {k: v for k, v in training_data["experts"].get(w, {}).items() if k in K}
        for w in training_data["experts"]
    }
    L_k = {k: v for k, v in training_data["L_k"].items() if k in K}

    if "S" in training_data:
        S = training_data["S"]
    else:
        S = list(range(num_shifts))

    f_w_s = f_w_s

    P = list(P_map.keys())
    Pm = [p for p in P if not P_map[p].endswith("training") and P_map[p] != "reserve"]

    # Toevoegen extra mappings
    K_p = {p: [k for (pp, k) in r_p_k if pp == p] for p in P}
    P_k = {k: [p for (p, kk) in r_p_k if kk == k] for k in K}

    return {
        "W": W,
        "K": K,
        "K_map": K_map,
        "K_reverse_map": K_reverse_map,
        "P": P,
        "P_map": P_map,
        "Pm": Pm,
        "r_p_k": r_p_k,
        "I_wks": I_wks,
        "experts": experts,
        "L_k": L_k,
        "f_w_s": f_w_s,
        "S": S,
        "K_p": K_p,
        "P_k": P_k,
        "alfa_p": alfa_p
    }



def build_gurobi_model(inputs: dict, beta: float = 100) -> tuple[Model, dict]:
    # Als inputs geen 'K_p' en 'P_k' bevatten, moeten we voorbereiden

    inputs = prepare_model_inputs(
        training_data=inputs,
        competence_data=inputs,
        vacation_matrix=inputs["vacation_matrix"],
        start_date=inputs["start_date"],
        num_shifts=len(inputs["S"]),
        alfa_p=inputs["alfa_p"],
        f_w_s = inputs["f_w_s"]
    )


    W = inputs["W"]
    K = inputs["K"]
    K_map = inputs["K_map"]
    K_reverse_map = inputs["K_reverse_map"]
    P = inputs["P"]
    P_map = inputs["P_map"]
    Pm = inputs["Pm"]
    r_p_k = inputs["r_p_k"]
    I_wks = inputs["I_wks"]
    experts = inputs["experts"]
    L_k = inputs["L_k"]
    f_w_s = inputs["f_w_s"]
    S = inputs["S"]
    alfa_p = inputs["alfa_p"]
    K_p = inputs["K_p"]
    P_k = inputs["P_k"]

    model = Model("Workforce Planning")

    x = model.addVars(W, Pm, S, vtype=GRB.BINARY, name="x")
    y = model.addVars(W, Pm, S, vtype=GRB.BINARY, name="y")
    z = model.addVars(W, S, vtype=GRB.BINARY, name="z")
    q = model.addVars(W, Pm, S, vtype=GRB.BINARY, name="q")
    h = model.addVars(W, K, S, vtype=GRB.BINARY, name="h")
    t = model.addVars(W, K, S, vtype=GRB.CONTINUOUS, lb=0, name="t")
    r = model.addVars(W, K, S, vtype=GRB.BINARY, name="r")
    v = model.addVars(W, K, S, vtype=GRB.CONTINUOUS, lb=0, name="v")
    u = model.addVars(W, Pm, S, vtype=GRB.BINARY, name="u")


    for s in S:
        for p in Pm:
            model.addConstr(sum(x[w, p, s] for w in W) == 1)
            model.addConstr(sum(y[w, p, s] for w in W) <= 1)

    for w in W:
        for s in S:
            model.addConstr(sum(x[w, p, s] for p in Pm) +
                            sum(y[w, p, s] for p in Pm) +
                            z[w, s] + f_w_s[w][s] == 1)


    for w in W:
        for p in Pm:
            for s in S:
                #model.addConstr(y[w, p, s] <= 1 - q[w, p, s])
                if K_p[p]:
                    model.addConstr(q[w, p, s] <= sum(h[w, k, s] for k in K_p[p]) / len(K_p[p]))

    for w in W:
        for k in K:
            if k not in L_k:
                continue
            s0 = S[0]
            initial = I_wks[w].get(k, {}).get(0, 0)
            model.addConstr(t[w, k, s0] == initial)

            value = 1 if initial >= 0.99 * L_k[k] else 0
            model.addConstr(h[w, k, s0] == value)

            model.addConstr(r[w, k, s0] == experts.get(w, {}).get(k, 0))
            model.addConstr(v[w, k, s0] == (800 if experts.get(w, {}).get(k, 0) else 0))

            for s in S[1:] if len(S) > 1 else []:

                model.addConstr(t[w, k, s] == t[w, k, s - 1] +
                                8 * sum(y[w, p, s] for p in Pm if k in K_p.get(p, [])))
                model.addConstr(h[w, k, s] <= t[w, k, s] / L_k[k])
                model.addConstr(h[w, k, s] >= h[w, k, s - 1])
                model.addConstr(v[w, k, s] == v[w, k, s - 1] +
                                8 * sum(q[w, p, s] * x[w, p, s] for p in P_k.get(k, []) if p in Pm))
                model.addConstr(r[w, k, s] <= v[w, k, s] / 800)
                model.addConstr(r[w, k, s] >= r[w, k, s - 1])

    # Supervisieconstraint bij training
    for s in S:
        for p in Pm:  # Aantal 'echte' posities
            for w in W:
                for k in K_p[p]:
                    model.addConstr(y[w, p, s] <= sum(x[we, p, s] * r[we, k, s] for we in W))

    # Traininghiërarchie
    for w in W:
        for s in S:
            # Constraint (14): y[w, 0, s] ≤ q[w, 1, s] en q[w, 2, s]
            for p_other in [1, 2]:
                model.addConstr(y[w, 0, s] <= q[w, p_other, s])

            # Constraint (15): y[1]+y[2] ≤ q[4], q[5], q[6]
            for p_other in [4, 5, 6]:
                model.addConstr(y[w, 1, s] + y[w, 2, s] <= q[w, p_other, s])

            # Constraint (16): y[4] + y[5] ≤ q[3]
            model.addConstr(y[w, 4, s] + y[w, 5, s] <= q[w, 3, s])

            # Constraint (17): y[5] ≤ q[4] en q[6]
            for p_other in [4, 6]:
                model.addConstr(y[w, 5, s] <= q[w, p_other, s])

    for w in W:
        for p in Pm:
            for s in S:
                model.addConstr(u[w, p, s] >= x[w, p, s] - q[w, p, s])
                model.addConstr(u[w, p, s] <= x[w, p, s])
                model.addConstr(u[w, p, s] <= 1 - q[w, p, s])

    # OBJECTIVE
    model.setObjective(
        sum(alfa_p[p] * y[w, p, s]
            for w in W for p in Pm for s in S)
        -  beta * sum(u[w, p, s] for w in W for p in Pm for s in S),
        GRB.MAXIMIZE
    )

    model.setParam("OutputFlag", 0)
    return model, {
        "x": x, "y": y, "z": z, "q": q, "h": h, "t": t,
        "r": r, "v": v, "u": u
    }
from gurobipy import Model, GRB

def build_gurobi_model_efficient(inputs: dict, beta: float = 100) -> tuple[Model, dict]:
    inputs = prepare_model_inputs(
        training_data=inputs,
        competence_data=inputs,
        vacation_matrix=inputs["vacation_matrix"],
        start_date=inputs["start_date"],
        num_shifts=len(inputs["S"]),
        alfa_p=inputs["alfa_p"],
        f_w_s=inputs["f_w_s"]
    )

    W = inputs["W"]
    K = inputs["K"]
    P = inputs["P"]
    Pm = inputs["Pm"]
    r_p_k = inputs["r_p_k"]
    I_wks = inputs["I_wks"]
    experts = inputs["experts"]
    L_k = inputs["L_k"]
    f_w_s = inputs["f_w_s"]
    S = inputs["S"]
    alfa_p = inputs["alfa_p"]
    K_p = inputs["K_p"]
    P_k = inputs["P_k"]

    model = Model("Workforce Planning Efficient")

    x = model.addVars(W, Pm, S, vtype=GRB.BINARY, name="x")
    y = model.addVars(W, Pm, S, vtype=GRB.BINARY, name="y")
    z = model.addVars(W, S, vtype=GRB.BINARY, name="z")
    q = model.addVars(W, Pm, S, vtype=GRB.BINARY, name="q")
    h = model.addVars(W, K, S, vtype=GRB.BINARY, name="h")
    t = model.addVars(W, K, S, vtype=GRB.CONTINUOUS, lb=0, name="t")
    u = model.addVars(W, Pm, S, vtype=GRB.BINARY, name="u")

    for s in S:
        for p in Pm:
            model.addConstr(sum(x[w, p, s] for w in W) == 1)
            model.addConstr(sum(y[w, p, s] for w in W) <= 1)

    for w in W:
        for s in S:
            model.addConstr(sum(x[w, p, s] for p in Pm) +
                            sum(y[w, p, s] for p in Pm) +
                            z[w, s] + f_w_s[w][s] == 1)

    for w in W:
        for p in Pm:
            for s in S:
                if K_p[p]:
                    model.addConstr(q[w, p, s] <= sum(h[w, k, s] for k in K_p[p]) / len(K_p[p]))

    for w in W:
        for k in K:
            s0 = S[0]
            if k not in L_k:
                continue
            model.addConstr(t[w, k, s0] == I_wks[w].get(k, {}).get(0, 0))
            model.addConstr(h[w, k, s0] == int(I_wks[w].get(k, {}).get(0, 0) >= 0.99 * L_k[k]))

            for s in S[1:] if len(S) > 1 else []:
                model.addConstr(t[w, k, s] == t[w, k, s - 1] +
                                8 * sum(y[w, p, s] for p in Pm if k in K_p.get(p, [])))
                model.addConstr(h[w, k, s] <= t[w, k, s] / L_k[k])
                model.addConstr(h[w, k, s] >= h[w, k, s - 1])

    # Supervisieconstraint met statische experts
    for s in S:
        for p in Pm:
            for w in W:
                for k in K_p[p]:
                    model.addConstr(y[w, p, s] <= sum(x[we, p, s] * experts.get(we, {}).get(k, 0) for we in W))

    # Traininghiërarchie (optioneel behoud, maar computationeel zwaar)
    # Precompute q_any groepen
    group_0 = [1, 2]
    group_1_2 = [4, 5, 6]
    group_5 = [4, 6]

    q_any_0 = model.addVars(W, S, vtype=GRB.BINARY, name="q_any_0")
    q_any_1_2 = model.addVars(W, S, vtype=GRB.BINARY, name="q_any_1_2")
    q_any_5 = model.addVars(W, S, vtype=GRB.BINARY, name="q_any_5")

    for w in W:
        for s in S:
            model.addConstr(q_any_0[w, s] <= sum(q[w, p, s] for p in group_0))
            for p in group_0:
                model.addConstr(q_any_0[w, s] >= q[w, p, s])

            model.addConstr(q_any_1_2[w, s] <= sum(q[w, p, s] for p in group_1_2))
            for p in group_1_2:
                model.addConstr(q_any_1_2[w, s] >= q[w, p, s])

            model.addConstr(q_any_5[w, s] <= sum(q[w, p, s] for p in group_5))
            for p in group_5:
                model.addConstr(q_any_5[w, s] >= q[w, p, s])

            model.addConstr(y[w, 0, s] <= q_any_0[w, s])
            model.addConstr(y[w, 1, s] + y[w, 2, s] <= q_any_1_2[w, s])
            model.addConstr(y[w, 5, s] <= q_any_5[w, s])
            model.addConstr(y[w, 4, s] + y[w, 5, s] <= q[w, 3, s])  # dit blijft apart

    for w in W:
        for p in Pm:
            for s in S:
                model.addConstr(u[w, p, s] >= x[w, p, s] - q[w, p, s])
                model.addConstr(u[w, p, s] <= x[w, p, s])
                model.addConstr(u[w, p, s] <= 1 - q[w, p, s])

    model.setObjective(
        sum(alfa_p[p] * y[w, p, s] for w in W for p in Pm for s in S)
        - beta * sum(u[w, p, s] for w in W for p in Pm for s in S),
        GRB.MAXIMIZE
    )

    model.setParam("OutputFlag", 0)
    return model, {
        "x": x, "y": y, "z": z, "q": q, "h": h, "t": t,
        "r": {}, "v": {},  # lege dicts voor compatibiliteit
        "u": u
    }

def solve_workforce_model(
    start_date=None,
    end_date=None,
    prepared_inputs=None,
    beta=00,
    alfa_p=None,
    warm_start_solution=None,
    MIPGAP = 0.,
    NUM_SOLUTIONS = 20
):
    """
    Lost het workforce planningsmodel op.

    Parameters:
    - start_date, end_date: optioneel, voor logging/debug
    - prepared_inputs: dictionary met alle inputstructuren
    - beta: strafgewicht op foutinzet
    - alfa_p: optionele gewichten per skill
    - warm_start_solution: optionele parsed oplossing om als start te gebruiken

    Returns:
    - model, vars, status
    """

    if prepared_inputs is None:
        raise ValueError("❌ Er moeten 'prepared_inputs' worden meegegeven aan solve_workforce_model.")

    # Alfa_p overschrijven indien gewenst
    if alfa_p is not None:
        prepared_inputs = {**prepared_inputs, "alfa_p": alfa_p}

    # Bouw model
    model, vars = build_gurobi_model_efficient(prepared_inputs, beta=beta)

    # ✅ Warm start instellen als meegegeven
    if warm_start_solution is not None:
        for varname, varset in vars.items():
            if varname not in warm_start_solution:
                continue
            for key, var in varset.items():
                val = warm_start_solution[varname].get(key)
                if val is not None:
                    try:
                        var.start = val
                    except Exception as e:
                        print(f"⚠️ Kon warm-start niet instellen voor {varname}[{key}]: {e}")
    # === Gurobi solver parameters ===
    model.setParam("OutputFlag", 1)  # Log aan
    model.setParam("TimeLimit", 50000)  # Kortere tijdslimiet (30 min): sneller feedback
    model.setParam("MIPGap", MIPGAP)  # Minder strenge gap → sneller stoppen
    model.setParam("PoolSearchMode", 2)  # Zoek meerdere oplossingen, maar minder diepgaand dan mode 2
    model.setParam("PoolSolutions", NUM_SOLUTIONS)  # Max 5 oplossingen is vaak voldoende in iteraties
    #model.setParam("NodeLimit", 500000)  # Beperk de B&B-boom → sneller naar iets werkends
    #model.setParam("SolutionLimit", 9999999)  # Stop zodra 1 oplossing gevonden is (zekerheid dat je niet blijft hangen)
    model.setParam("Threads", 12)  # Gebruik alle cores
    model.setParam("Presolve", 2)  # Automatisch vooroptimaliseren
    model.setParam("Cuts", 1)  # Iets minder agressieve cuts (sneller en stabieler)
    model.setParam("Heuristics", 0.5)  # Meer focus op heuristieken (meer kans op snelle feasible oplossing)
    model.setParam("Method", -1)  # Laat Gurobi zelf kiezen (Barrier is default voor root)
    model.setParam("LogToConsole", 0)
    model.setParam("DisplayInterval", 20)

    model.optimize()

    status = model.Status

    if model.status == GRB.INFEASIBLE:
        print("⚡️ Model infeasible! Schrijf IIS...")
        model.computeIIS()
        model.write("model_infeasible.ilp")
        print("✅ IIS written to 'model_infeasible.ilp'")

    return model, vars, status


