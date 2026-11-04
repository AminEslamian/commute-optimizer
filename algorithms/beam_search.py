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
        time_domain = self.problem.get_valid_time_domain()
        base = self.problem.generate_initial_state()

        # Initialize beam with diverse departure times
        beam = [base]
        while len(beam) < self.beam_width:
            rand_time = random.choice(time_domain)
            cand = State(path=base.path, departure_time=rand_time)
            if self.problem.is_valid_state(cand) and cand not in beam:
                beam.append(cand)
            elif len(beam) < self.beam_width:
                beam.append(base)

        best_state = base
        best_cost = self.problem.evaluate(base)

        stagnant_steps = 0

        for _ in range(self.max_iterations):
            # Find best state in the current beam
            current_best = min(beam, key=lambda s: self.problem.evaluate(s))
            current_best_cost = self.problem.evaluate(current_best)

            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_state = current_best
                stagnant_steps = 0
            else:
                stagnant_steps += 1

            self.history.append(best_cost)

            # Stop if beam makes no improvement for 15 consecutive iterations
            if stagnant_steps >= 15:
                break

            # Collect neighbors from all beam states
            pool = []
            seen = set()

            for state in beam:
                for nbr in get_neighbors(state, self.problem):
                    if nbr not in seen:
                        seen.add(nbr)
                        pool.append((nbr, self.problem.evaluate(nbr)))

            if not pool:
                break

            # Retain top K states across the entire pool
            pool.sort(key=lambda x: x[1])
            beam = [item[0] for item in pool[:self.beam_width]]

        return best_state, best_cost