"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""
import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Local Beam Search.

        Guidelines:
        - Maintain a set of `beam_width` parallel configurations.
        - In each iteration, generate neighbors from all active states in the beam.
        - Pool and rank all generated successors collectively, then retain the top `beam_width` states.
        - Halt when the beam ceases to improve or reaches max_iterations.
        - Record the best cost of the population at each step in self.history.

        Encouraged:
        - Ensuring diversity across states in the beam to prevent collapse into a single redundant path.

        Prohibited:
        - Treating this as K independent, non-communicating searches.
        """
        beam = []
        time_domain = self.problem.get_valid_time_domain()
        base = self.problem.generate_initial_state()
        beam.append(base)

        while len(beam) < self.beam_width:
            t = random.choice(time_domain)
            st = State(path=base.path, departure_time=t)
            beam.append(st)

        best_overall_state = base
        best_overall_cost = self.problem.evaluate(base)

        for _ in range(self.max_iterations):
            # Find best state in current beam
            for state in beam:
                cost = self.problem.evaluate(state)
                if cost < best_overall_cost:
                    best_overall_cost = cost
                    best_overall_state = state

            self.history.append(best_overall_cost)

            next_beam = []
            improved_any = False

            for state in beam:
                current_cost = self.problem.evaluate(state)
                neighbors = get_neighbors(state, self.problem)
                if not neighbors:
                    next_beam.append(state)
                    continue

                best_nbr = None
                best_nbr_cost = float("inf")
                for nbr in neighbors:
                    c = self.problem.evaluate(nbr)
                    if c < best_nbr_cost:
                        best_nbr_cost = c
                        best_nbr = nbr

                if best_nbr_cost < current_cost:
                    next_beam.append(best_nbr)
                    improved_any = True
                else:
                    next_beam.append(state)

            beam = next_beam
            if not improved_any:
                break

        return best_overall_state, best_overall_cost
