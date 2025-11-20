"""
FX analysis examples.

This module contains example scripts demonstrating various analyses
of the LRW FX model including volatility smile analysis, Greeks analysis,
and basket pricing examples.
"""

# Import main analysis functions for easy access
from .volatility_smile_analysis import (
    create_base_fx_model,
    compute_implied_volatilities,
    analyze_correlation_impact,
    analyze_term_structure,
    analyze_vol_of_vol_impact,
    analyze_interest_rate_differential
)

__all__ = [
    'create_base_fx_model',
    'compute_implied_volatilities',
    'analyze_correlation_impact',
    'analyze_term_structure',
    'analyze_vol_of_vol_impact',
    'analyze_interest_rate_differential'
]
