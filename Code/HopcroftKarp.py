from collections import defaultdict, deque

class HopcroftKarp:
    @staticmethod
    def has_augmenting_path(lefts, edges, to_matched_right, to_matched_left, distances):
        # Initialize distances
        q = deque()
        for left in lefts:
            if to_matched_right[left] == "":
                distances[left] = 0
                q.append(left)
            else:
                distances[left] = float("inf")
        distances[""] = float("inf")

        # Perform BFS
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
                if HopcroftKarp.try_matching(next_left, edges, to_matched_right, to_matched_left, distances):
                    to_matched_left[right] = left
                    to_matched_right[left] = right
                    return True

        # Mark as not matchable
        distances[left] = float("inf")
        return False

    @staticmethod
    def hopcroft_karp(lefts, rights, edges):
        # Initialize distances and matching dictionaries
        distances = {}
        to_matched_right = {left: "" for left in lefts}
        to_matched_left = {right: "" for right in rights}

        # Continue until no augmenting paths are found
        while HopcroftKarp.has_augmenting_path(lefts, edges, to_matched_right, to_matched_left, distances):
            for unmatched_left in (left for left in lefts if to_matched_right[left] == ""):
                HopcroftKarp.try_matching(unmatched_left, edges, to_matched_right, to_matched_left, distances)

        # Remove unmatched entries and return matches
        return {k: v for k, v in to_matched_right.items() if v != ""}


# Example usage
if __name__ == "__main__":
    # Example input
    lefts = {"L1", "L2", "L3"}
    rights = {"R1", "R2", "R3"}
    edges = {
        "L1": {"R1", "R2"},
        "L2": {"R2", "R3"},
        "L3": {"R1", "R3"}
    }

    # Perform matching
    matches = HopcroftKarp.hopcroft_karp(lefts, rights, edges)

    # Output matches
    print("Matches:", matches)
