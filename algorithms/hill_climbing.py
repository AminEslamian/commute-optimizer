"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Hill Climbing optimization.

        Guidelines:
        - Search through the neighborhood greedily for candidates that minimize total travel time.
        - Handle flat local maxima / shoulders by allowing up to `max_sideways` non-improving steps.
        - Apply up to `max_restarts` to explore different starting basins.
        - Track the best objective cost seen at each major step in self.history.

        Prohibited:
        - Looping indefinitely on flat plateaus.
        """
        best_state = None
        best_cost = float("inf")
        time_domain = self.problem.get_valid_time_domain()

        for restart in range(self.max_restarts + 1):
            # First run starts at the base Dijkstra state; restarts pick a random valid time
            if restart == 0:
                current_state = self.problem.generate_initial_state()
            else:
                base = self.problem.generate_initial_state()
                current_state = State(path=base.path, departure_time=random.choice(time_domain))
                # Apply a couple random neighbor moves to diverge from the base route
                for _ in range(random.randint(1, 4)):
                    nbrs = get_neighbors(current_state, self.problem)
                    if nbrs:
                        current_state = random.choice(nbrs)

            current_cost = self.problem.evaluate(current_state)

            if current_cost < best_cost:
                best_cost = current_cost
                best_state = current_state

            sideways_moves = 0

            while True:
                self.history.append(best_cost)

                neighbors = get_neighbors(current_state, self.problem)
                if not neighbors:
                    break

                # Steepest descent: find the neighbor with minimum travel time
                best_nbr = None
                best_nbr_cost = float("inf")
                for nbr in neighbors:
                    cost = self.problem.evaluate(nbr)
                    if cost < best_nbr_cost:
                        best_nbr_cost = cost
                        best_nbr = nbr

                # Better neighbor found
                if best_nbr_cost < current_cost:
                    current_state = best_nbr
                    current_cost = best_nbr_cost
                    sideways_moves = 0

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_state = current_state

                # Sideways move on a plateau
                elif best_nbr_cost == current_cost and sideways_moves < self.max_sideways:
                    current_state = best_nbr
                    sideways_moves += 1

                else:
                    break

        return best_state, best_cost
