"""
algorithms/genetic_algorithm.py
Population-based optimization using selection, recombination, and mutation.
"""
import random
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Genetic Algorithm optimization.

        Guidelines:
        - Initialize a population of valid candidate solutions.
        - Design a fitness metric suitable for minimizing commute travel time (lower cost -> higher fitness).
        - Use selection operators (such as tournament or roulette selection) to pick parents.
        - Pair parents to produce offspring via `crossover()`, and apply stochastic mutation.
        - Preserve the top candidates across generations if using elitism.
        - Track best population performance per generation in self.history.

        Encouraged:
        - Experimenting with population size, selection pressure, and operator rates to balance exploration vs. exploitation.
        """
        time_domain = self.problem.get_valid_time_domain()
        base = self.problem.generate_initial_state()

        # Initialize population with random valid departure times
        population = [base]
        for _ in range(self.population_size - 1):
            rand_time = random.choice(time_domain)
            ind = State(path=base.path, departure_time=rand_time)
            # Apply 1 or 2 neighbor steps to diversify routes
            nbrs = get_neighbors(ind, self.problem)
            if nbrs:
                ind = random.choice(nbrs)
            population.append(ind)

        best_state = base
        best_cost = self.problem.evaluate(base)

        for gen in range(self.generations):
            # Calculate cost and fitness (lower travel time -> higher fitness)
            evaluated = []
            for ind in population:
                cost = self.problem.evaluate(ind)
                fitness = 1000.0 / (cost + 1e-6)
                evaluated.append((ind, cost, fitness))

            # Sort by fitness descending (best first)
            evaluated.sort(key=lambda x: x[2], reverse=True)

            # Update best candidate
            current_best_ind, current_best_cost, _ = evaluated[0]
            if current_best_cost < best_cost:
                best_cost = current_best_cost
                best_state = current_best_ind

            self.history.append(best_cost)

            # carry over the top individuals directly
            new_population = [item[0] for item in evaluated[:self.elitism]]

            # Fill the rest of the generation with offspring
            while len(new_population) < self.population_size:
                # 3-way tournament selection for parent 1
                t1 = random.sample(evaluated, min(3, len(evaluated)))
                p1 = max(t1, key=lambda x: x[2])[0]

                # 3-way tournament selection for parent 2
                t2 = random.sample(evaluated, min(3, len(evaluated)))
                p2 = max(t2, key=lambda x: x[2])[0]

                # Crossover
                if random.random() < self.crossover_rate:
                    child = crossover(p1, p2, self.problem)
                else:
                    child = p1 if random.random() < 0.5 else p2

                # Mutation
                if random.random() < self.mutation_rate:
                    nbrs = get_neighbors(child, self.problem)
                    if nbrs:
                        child = random.choice(nbrs)

                if self.problem.is_valid_state(child):
                    new_population.append(child)
                else:
                    new_population.append(p1)

            population = new_population

        return best_state, best_cost