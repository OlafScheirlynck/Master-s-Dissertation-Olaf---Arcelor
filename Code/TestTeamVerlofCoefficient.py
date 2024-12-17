import itertools
from collections import defaultdict, deque
import pandas as pd

class EmployabilityCalculationManager:
    def __init__(self, team):
        self.team = team




    def calculate(self):
            employee_results = {member.employee_id: EmployeeResult(member.employee_id, member.employee_positions)
                                for member in self.team.team_members}
            team_result = TeamResult(self.team.team_id, self.team.positions_to_fill)

            amount_of_positions_to_fill = len(self.team.positions_to_fill)
            team_size = len(self.team.team_members)

            if team_size < amount_of_positions_to_fill:
                print("Not enough team members to fill positions.")
                return list(employee_results.values()), team_result

            all_combinations = itertools.combinations(self.team.team_members, amount_of_positions_to_fill)

            nok = 0

            for drawn_team in all_combinations:
                rights = set(self.team.positions_to_fill)
                edges = {member.employee_id: set(member.employee_positions) for member in drawn_team}
                lefts = set(edges.keys())

                matches = EmployabilityCalculationManager.hopcroft_karp(lefts, rights, edges)

                if len(matches) != amount_of_positions_to_fill:
                    nok += 1
                    for key in team_result.missing_functions.keys():
                        if key not in matches.values():
                            team_result.missing_functions[key] += 1
                            break

                for employee_id, result in employee_results.items():
                    if employee_id not in edges:
                        result.result_positions["Rest"] += 1
                        if len(matches) != amount_of_positions_to_fill:
                            result.result_absent_nok += 1
                        else:
                            result.result_absent_ok += 1
                    elif employee_id not in matches:
                        result.result_positions["Unused"] += 1
                        result.result_present_nok += 1
                    else:
                        assigned_position = matches[employee_id]
                        result.result_positions[assigned_position] += 1
                        if len(matches) != amount_of_positions_to_fill:
                            result.result_present_nok += 1
                            # Check if underqualified
                            if assigned_position not in edges[employee_id]:
                                result.result_underqualified[assigned_position] += 1
                                team_result.underqualified_assignments[assigned_position] += 1
                        else:
                            result.result_present_ok += 1

            total_combinations = len(list(itertools.combinations(self.team.team_members, amount_of_positions_to_fill)))
            team_result.flexibility = (total_combinations - nok) / total_combinations * 100.0

            return list(employee_results.values()), team_result

    @staticmethod
    def has_augmenting_path(lefts, edges, to_matched_right, to_matched_left, distances):
        q = deque()
        for left in lefts:
            if to_matched_right[left] == "":
                distances[left] = 0
                q.append(left)
            else:
                distances[left] = float("inf")
        distances[""] = float("inf")

        while q:
            left = q.popleft()
            if distances[left] < distances[""]:
                for right in edges[left]:
                    next_left = to_matched_left[right]
                    if distances[next_left] == float("inf"):
                        distances[next_left] = distances[left] + 1
                        q.append(next_left)

        return distances[""] != float("inf")

    @staticmethod
    def try_matching(left, edges, to_matched_right, to_matched_left, distances):
        if left == "":
            return True

        for right in edges[left]:
            next_left = to_matched_left[right]
            if distances[next_left] == distances[left] + 1:
                if EmployabilityCalculationManager.try_matching(next_left, edges, to_matched_right, to_matched_left, distances):
                    to_matched_left[right] = left
                    to_matched_right[left] = right
                    return True

        distances[left] = float("inf")
        return False

    @staticmethod
    def hopcroft_karp(lefts, rights, edges):
        distances = {}
        to_matched_right = {left: "" for left in lefts}
        to_matched_left = {right: "" for right in rights}

        while EmployabilityCalculationManager.has_augmenting_path(lefts, edges, to_matched_right, to_matched_left, distances):
            for unmatched_left in (left for left in lefts if to_matched_right[left] == ""):
                EmployabilityCalculationManager.try_matching(unmatched_left, edges, to_matched_right, to_matched_left, distances)

        return {k: v for k, v in to_matched_right.items() if v != ""}


class EmployeeResult:
    def __init__(self, employee_id, employee_positions):
        self.employee_id = employee_id
        self.result_positions = defaultdict(int)
        self.result_present_ok = 0
        self.result_present_nok = 0
        self.result_absent_ok = 0
        self.result_absent_nok = 0
        self.result_underqualified = defaultdict(int)  # Tracks underqualified assignments


class TeamResult:
    def __init__(self, team_id, positions_to_fill):
        self.team_id = team_id
        self.positions_to_fill = positions_to_fill
        self.missing_functions = defaultdict(int)
        self.flexibility = 0.0
        self.underqualified_assignments = defaultdict(int)  # Tracks underqualified assignments


class TeamMember:
    def __init__(self, employee_id, availability, employee_positions):
        self.employee_id = employee_id
        self.availability = availability
        self.employee_positions = employee_positions


class Team:
    def __init__(self, team_id, team_members, positions_to_fill):
        self.team_id = team_id
        self.team_members = team_members
        self.positions_to_fill = positions_to_fill


# Test function with dynamic input
def test_employability_calculation():
    # Define all datasets as named teams
    teams_data = {
        "A": {
            "32010247": ["709352", "718714", "811071"],
            "32014541": ["711629", "741890", "718714", "811071"],
            "32020015": ["711629", "719556", "738713", "738649", "741890", "718714", "811071"],
            "32025797": ["711629", "719556", "738713", "738649", "741890", "718714", "811071"],
            "95560": ["993070", "719556", "738713", "741649", "981326", "990365", "741890", "718714", "970123"],
            "32010290": ["993070", "711629", "719556", "738713", "738649", "741649", "981326", "990365", "741890",
                         "718714", "970123"],
            "32010287": ["993070", "711629", "719556", "738713", "738649", "741649", "741890", "718714", "970123"],
            "32014250": ["993070", "711629", "719556", "738713", "738649", "741649", "981326", "990365", "741890",
                         "718714", "970123"],
        },
        "B": {
            "32010248": ["711325", "711336", "767017"],
            "32014538": ["711336", "767017", "122650", "116677"],
            "32010267": ["12614", "992106", "711325", "990622", "737597", "711336", "976794", "968344", "738802",
                         "767017", "995215", "122650", "116677", "576111", "741126"],
            "32020012": ["737597", "968344", "767017", "995215", "122650", "116677", "576111", "741126"],
            "32010253": ["12614", "992106", "990622", "737597", "976794", "968344", "767017", "995215", "122650",
                         "116677", "741126"],
            "32010275": ["12614", "992106", "990622", "737597", "976794", "968344", "738802", "767017", "995215",
                         "122650", "576111", "741126"],
            "32010277": ["12614", "990622", "737597", "976794", "968344", "738802", "767017", "995215", "116677",
                         "741126"],
            "32010276": ["12614", "990622", "737597", "976794", "968344", "738802", "767017", "995215", "122650",
                         "116677", "576111", "741126"],
        },
        "C": {
            "32010249": ["116647", "711330", "738624"],
            "32014539": ["706194", "116647", "711330", "738624", "738571", "712934", "712884"],
            "32025253": ["706194", "116647", "711330", "737597", "738624", "738571", "712934", "740937", "962237",
                         "707921", "712884"],
            "32020013": ["706194", "116647", "711330", "737597", "738624", "738571", "712934", "740937", "962237",
                         "707921", "712884"],
            "32010403": ["956852", "706194", "116647", "711330", "737597", "738624", "703650", "738571", "712934",
                         "740937", "996641", "976794", "962237", "707921", "712884", "997295", "964841"],
            "32010280": ["956852", "706194", "116647", "711330", "737597", "738624", "703650", "738571", "740937",
                         "996641", "976794", "962237", "707921", "712884", "997295", "964841"],
            "32010282": ["956852", "706194", "116647", "711330", "737597", "738624", "703650", "738571", "712934",
                         "740937", "996641", "976794", "962237", "707921", "712884", "997295", "964841"],
            "32010281": ["956852", "706194", "116647", "711330", "737597", "738624", "703650", "738571", "712934",
                         "740937", "996641", "976794", "962237", "707921", "712884", "997295", "964841"],
        },
        "D": {
            "32010250": ["117371", "7094"],
            "32014540": ["706194", "709428", "712884"],
            "32010286": ["706194", "964275", "893632", "740937", "709428", "712884", "967469", "730474", "820136"],
            "32010264": ["706194", "740937", "709428", "712884", "730474", "820136"],
            "32010257": ["706194", "964275", "893632", "740937", "997220", "712884", "967469", "730474", "991902",
                         "820136"],
            "32010278": ["706194", "964275", "893632", "740937", "709428", "712884", "967469", "730474", "991902",
                         "820136"],
            "32010285": ["706194", "964275", "893632", "740937", "709428", "712884", "967469", "730474", "820136"],
            "32010283": ["706194", "964275", "893632", "740937", "709428", "997220", "712884", "967469", "730474",
                         "991902", "820136"],
        },
    }

    # Function to choose and process a team
    def process_team(team_name):
        if team_name not in teams_data:
            print("Invalid team name. Please choose A, B, C, or D.")
            return

        # Get team data
        positions_and_members = teams_data[team_name]

        # Convert to TeamMember and Team objects for calculation
        team_members = []
        positions_to_fill = list(positions_and_members.keys())

        for position, members in positions_and_members.items():
            for member in members:
                if not any(tm.employee_id == member for tm in team_members):
                    team_members.append(TeamMember(member, 100, []))
                # Assign the position to the qualified team members
                for tm in team_members:
                    if tm.employee_id == member:
                        tm.employee_positions.append(position)

        # Create the team object
        team = Team(f"Team_{team_name}", team_members, positions_to_fill)

        # Initialize EmployabilityCalculationManager
        manager = EmployabilityCalculationManager(team)

        # Run the calculation
        employee_results, team_result = manager.calculate()

        # Print results
        print("Employee Results:")
        for result in employee_results:
            print(f"{result.employee_id}: {dict(result.result_positions)}")
            print(f"  Present OK: {result.result_present_ok}")
            print(f"  Present NOK: {result.result_present_nok}")
            print(f"  Absent OK: {result.result_absent_ok}")
            print(f"  Absent NOK: {result.result_absent_nok}")
            print(f"  Underqualified Assignments: {dict(result.result_underqualified)}")

        print("\nTeam Result:")
        print(f"Flexibility: {team_result.flexibility:.2f}%")
        print("Missing Functions:", dict(team_result.missing_functions))
        print("Underqualified Assignments:", dict(team_result.underqualified_assignments))

    # Choose a team to process
    team_choice = input("Choose a team to process (A, B, C, D): ").strip().upper()
    process_team(team_choice)


# Run test
test_employability_calculation()
