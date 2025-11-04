import random
import numpy as np
from deap import base, creator, tools, algorithms
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# --- Configuration ---
# The problem: Minimize total energy cost over a 24-hour period by scheduling 
# two machines (M001, M002) that must run for a minimum total duration.

# Machine power consumption (kW) - Simplified from data_simulator
M001_POWER = 150  # High power machine
M002_POWER = 80   # Medium power machine

# Constraints
TOTAL_RUN_TIME_M001 = 8  # M001 must run for 8 hours total
TOTAL_RUN_TIME_M002 = 12 # M002 must run for 12 hours total
TIME_SLOTS = 24  # 24 hours in a day

# Cost profile (simulated variable electricity cost per hour)
# Lower cost during off-peak (night) hours
COST_PROFILE = np.array([
    0.05, 0.05, 0.05, 0.05, 0.05, 0.05, # 00:00 - 05:00 (Off-peak)
    0.10, 0.15, 0.20, 0.25, 0.25, 0.20, # 06:00 - 11:00 (Morning peak)
    0.15, 0.10, 0.10, 0.10, 0.15, 0.20, # 12:00 - 17:00 (Mid-day)
    0.25, 0.30, 0.25, 0.20, 0.15, 0.10  # 18:00 - 23:00 (Evening peak)
])

# --- DEAP Setup ---

# 1. Define the fitness function (minimize cost)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# 2. Define the Individual (a schedule for 2 machines over 24 hours)
# Individual is a list of 48 integers (24 hours * 2 machines) where 1=ON, 0=OFF
creator.create("Individual", list, fitness=creator.FitnessMin)

# 3. Initialize the toolbox
toolbox = base.Toolbox()

# Attribute generator: 0 or 1 (machine ON or OFF for an hour)
toolbox.register("attr_bool", random.randint, 0, 1)

# Individual generator: 48 attributes
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, TIME_SLOTS * 2)

# Population generator
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_schedule(individual):
    """
    Fitness function: Calculates the total energy cost for a given schedule.
    Also applies a penalty if constraints are violated.
    """
    schedule_m001 = np.array(individual[:TIME_SLOTS])
    schedule_m002 = np.array(individual[TIME_SLOTS:])
    
    # 1. Calculate total energy cost
    energy_m001 = schedule_m001 * M001_POWER  # kWh for M001
    energy_m002 = schedule_m002 * M002_POWER  # kWh for M002
    total_cost = np.sum((energy_m001 + energy_m002) * COST_PROFILE)
    
    # 2. Apply penalty for constraint violation
    penalty = 0.0
    
    # Constraint 1: Total run time for M001
    run_time_m001 = np.sum(schedule_m001)
    if run_time_m001 != TOTAL_RUN_TIME_M001:
        penalty += abs(run_time_m001 - TOTAL_RUN_TIME_M001) * 1000 # High penalty
        
    # Constraint 2: Total run time for M002
    run_time_m002 = np.sum(schedule_m002)
    if run_time_m002 != TOTAL_RUN_TIME_M002:
        penalty += abs(run_time_m002 - TOTAL_RUN_TIME_M002) * 1000 # High penalty
        
    # Constraint 3: M001 and M002 cannot run at the same time (simplified collision avoidance)
    # This is a strong constraint for a small workshop
    collision_hours = np.sum(schedule_m001 * schedule_m002)
    penalty += collision_hours * 500 # Medium penalty for each collision hour
    
    return total_cost + penalty,

# Register GA operators
toolbox.register("evaluate", evaluate_schedule)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga_optimization(
    pop_size: int = 100, 
    generations: int = 50, 
    cx_prob: float = 0.7, 
    mut_prob: float = 0.2
) -> Tuple[float, List[int]]:
    """Runs the Genetic Algorithm to find the optimal schedule."""
    logger.info("Starting Genetic Algorithm optimization...")
    
    # Initialize population
    pop = toolbox.population(n=pop_size)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run the GA
    pop, log = algorithms.eaSimple(
        pop, 
        toolbox, 
        cxpb=cx_prob, 
        mutpb=mut_prob, 
        ngen=generations, 
        stats=stats, 
        halloffame=tools.HallOfFame(1), 
        verbose=False
    )
    
    best_individual = tools.HallOfFame(1).get()[0]
    best_cost = best_individual.fitness.values[0]
    
    logger.info(f"Optimization finished. Best cost found: {best_cost:.2f}")
    
    return best_cost, best_individual

def format_schedule(individual: List[int]) -> List[dict]:
    """Converts the best individual (schedule) into a human-readable format."""
    schedule_m001 = individual[:TIME_SLOTS]
    schedule_m002 = individual[TIME_SLOTS:]
    
    formatted = []
    for h in range(TIME_SLOTS):
        formatted.append({
            "hour": f"{h:02d}:00-{h+1:02d}:00",
            "cost_factor": COST_PROFILE[h],
            "M001_status": "ON" if schedule_m001[h] == 1 else "OFF",
            "M002_status": "ON" if schedule_m002[h] == 1 else "OFF",
            "M001_power_kw": M001_POWER if schedule_m001[h] == 1 else 0,
            "M002_power_kw": M002_POWER if schedule_m002[h] == 1 else 0,
        })
    return formatted

class GADigitalTwinOptimizer:
    """A class to wrap the GA logic for easy integration."""
    
    def get_optimal_schedule(self) -> Tuple[float, List[dict]]:
        """Public method to run the optimization and return the result."""
        try:
            best_cost, best_individual = run_ga_optimization()
            formatted_schedule = format_schedule(best_individual)
            return best_cost, formatted_schedule
        except Exception as e:
            logger.error(f"GA Optimization failed: {e}", exc_info=True)
            return 0.0, []

if __name__ == "__main__":
    # Example usage
    optimizer = GADigitalTwinOptimizer()
    cost, schedule = optimizer.get_optimal_schedule()
    
    print("\n--- Optimal Schedule Recommendation ---")
    print(f"Total Optimized Cost: ${cost:.2f}")
    print(pd.DataFrame(schedule).to_markdown(index=False))
