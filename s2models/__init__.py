"""s2models – topological scale framework for hypergraphs.

Python implementation of

    H. Molina-Abril, M.J. Morón-Fernández, M. Benito-Marimón, F. Díaz-del-Río,
    P. Real. "Topological scale framework for hypergraphs".
    Applied Mathematics and Computation 485 (2025) 128989.
    https://doi.org/10.1016/j.amc.2024.128989

Main entry points
-----------------
* :class:`Hypergraph` – cellular hypergraphs (Definitions 1–3).
* :class:`ChainComplex`, :class:`ChainMap` – algebraic hypergraphs and their
  homology through the Smith Normal Form (Appendix A).
* :class:`BoundaryScaleModel`, :class:`ColorRefinementModel`, ... –
  scale-space models (s²-models, Definitions 4–7).
* :class:`HomologyS2Model` – the homology s²-model (Definition 8).
* :mod:`s2models.invariants` – the certificates ω_d, ω_β, ω_i, ω_ρ+, ω_ρ- used
  for graph/hypergraph isomorphism testing (Section 6).
"""

from .chain import ChainComplex, ChainMap, Homology, identity_map
from .homology_model import HomologyS2Model
from .hypergraph import Hypergraph
from .invariants import (
    METRICS,
    certificate,
    cr_inter_degree_certificate,
    hybrid_certificate,
    isomorphism_test,
    level_features,
    model_features,
    omega_beta,
    omega_d,
    omega_i,
    omega_rho_minus,
    omega_rho_plus,
    unique_fraction,
)
from .linalg import elementary_divisors, rank, smith_normal_form, snf_multiset
from .models import (
    BasisChangeModel,
    BoundaryScaleModel,
    ColorRefinementModel,
    IteratedLaplacianModel,
    S2Model,
    TrivialAdjacencyModel,
)
from .rings import Z, Z2, Ring, Zp

__version__ = "0.1.0"

__all__ = [
    "Hypergraph",
    "ChainComplex",
    "ChainMap",
    "Homology",
    "identity_map",
    "S2Model",
    "BoundaryScaleModel",
    "BasisChangeModel",
    "ColorRefinementModel",
    "IteratedLaplacianModel",
    "TrivialAdjacencyModel",
    "HomologyS2Model",
    "Ring",
    "Z",
    "Z2",
    "Zp",
    "smith_normal_form",
    "elementary_divisors",
    "snf_multiset",
    "rank",
    "METRICS",
    "level_features",
    "model_features",
    "certificate",
    "omega_d",
    "omega_beta",
    "omega_i",
    "omega_rho_plus",
    "omega_rho_minus",
    "unique_fraction",
    "isomorphism_test",
    "cr_inter_degree_certificate",
    "hybrid_certificate",
]
