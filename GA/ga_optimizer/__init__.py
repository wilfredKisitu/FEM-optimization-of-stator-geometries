"""ga_optimizer — Multi-objective NSGA-II optimisation of stator geometries.

Public API::

    from ga_optimizer import run_ga, ParetoArchive

    archive = run_ga(
        config_path="ga_optimizer/configs/default_ga.yaml",
        fea_config_path="FEA/configs/default.yaml",
        output_dir="ga_results/",
    )
    print(f"Pareto front: {archive.size} solutions")
"""

from .orchestrator import run_ga, GAState, check_termination
from .pareto.archive import ParetoArchive
from .chromosome import decode_chromosome, random_individual, GENE_DEFINITIONS, N_GENES
from .objectives import ObjectiveVector, extract_objectives, INFEASIBLE_OBJECTIVES
from .population import Individual, Population, initialise_population

__all__ = [
    "run_ga",
    "GAState",
    "check_termination",
    "ParetoArchive",
    "decode_chromosome",
    "random_individual",
    "GENE_DEFINITIONS",
    "N_GENES",
    "ObjectiveVector",
    "extract_objectives",
    "INFEASIBLE_OBJECTIVES",
    "Individual",
    "Population",
    "initialise_population",
]
