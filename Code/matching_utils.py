# matching_utils.py

from collections import deque

def hopcroft_karp(lefts, rights, edges):
    """
    Voer Hopcroft-Karp matching uit.

    Parameters:
    - lefts: lijst van werknemers
    - rights: lijst van posities
    - edges: dict {werknemer: lijst van posities}

    Returns:
    - dict {werknemer: positie} indien gematcht
    """
    d = {}
    to_r = {l: "" for l in lefts}
    to_l = {r: "" for r in rights}

    def has_augmenting_path():
        q = deque()
        for l in lefts:
            d[l] = 0 if to_r[l] == "" else float("inf")
            if to_r[l] == "":
                q.append(l)
        d[""] = float("inf")
        while q:
            l = q.popleft()
            if d[l] < d[""]:
                for r in edges.get(l, []):
                    nl = to_l[r]
                    if d.get(nl, float("inf")) == float("inf"):
                        d[nl] = d[l] + 1
                        q.append(nl)
        return d[""] != float("inf")

    def try_matching(l):
        if l == "":
            return True
        for r in edges.get(l, []):
            nl = to_l[r]
            if d.get(nl, float("inf")) == d[l] + 1 and try_matching(nl):
                to_l[r] = l
                to_r[l] = r
                return True
        d[l] = float("inf")
        return False

    while has_augmenting_path():
        for l in (l for l in lefts if to_r[l] == ""):
            try_matching(l)

    return {l: r for l, r in to_r.items() if r != ""}
