# sensitivities/results.py
"""Structured results and logging for sensitivity calculations."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from enum import Enum
import json
import logging
from datetime import datetime

import numpy as np
import jax.numpy as jnp

logger = logging.getLogger(__name__)


class InstrumentType(Enum):
    """Type of hedging instrument."""
    ZERO_COUPON = "ZC"
    SWAP = "SWAP"
    BOND = "BOND"


class LegType(Enum):
    """Type of swap leg."""
    FLOATING = "FLOATING"
    FIXED = "FIXED"


class GreekType(Enum):
    """Type of Greek sensitivity."""
    DELTA = "DELTA"
    GAMMA = "GAMMA"
    VEGA = "VEGA"
    ALPHA = "ALPHA"
    OMEGA = "OMEGA"
    M = "M"


# =============================================================================
# Base Result Classes
# =============================================================================

@dataclass
class LegResult:
    """Result for a single leg component."""
    maturity: float
    value: float
    leg_type: LegType
    
    def to_dict(self) -> Dict:
        return {
            "maturity": self.maturity,
            "value": self.value,
            "leg_type": self.leg_type.value
        }


@dataclass
class MatrixResult:
    """Result for matrix-valued sensitivities (Vega, Omega, M)."""
    greek_type: GreekType
    strike: float
    values: np.ndarray
    component_details: Dict[tuple, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.values, jnp.ndarray):
            self.values = np.array(self.values)
    
    def to_dict(self) -> Dict:
        return {
            "greek_type": self.greek_type.value,
            "strike": self.strike,
            "matrix": self.values.tolist(),
            "components": {f"({i},{j})": v for (i, j), v in self.component_details.items()}
        }
    
    def summary(self, precision: int = 6) -> str:
        lines = [
            f"\n{'='*60}",
            f"{self.greek_type.value} Sensitivity Matrix (Strike: {self.strike:.4f})",
            f"{'='*60}",
        ]
        
        n = self.values.shape[0]
        
        # Header row
        header = "     " + "".join([f"   [{j}]    " for j in range(n)])
        lines.append(header)
        lines.append("-" * len(header))
        
        # Matrix rows
        for i in range(n):
            row_values = "  ".join([f"{self.values[i,j]:+.{precision}f}" for j in range(n)])
            lines.append(f"[{i}]  {row_values}")
        
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# =============================================================================
# Delta Hedging Results
# =============================================================================

@dataclass
class DeltaHedgingResult:
    """Structured delta hedging result."""
    strike: float
    instrument_type: InstrumentType
    price: float
    floating_leg: Dict[float, float] = field(default_factory=dict)
    fixed_leg: Dict[float, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "greek": "DELTA",
            "strike": self.strike,
            "instrument_type": self.instrument_type.value,
            "price": self.price,
            "floating_leg": {f"T={k:.2f}": v for k, v in self.floating_leg.items()},
            "fixed_leg": {f"T={k:.2f}": v for k, v in self.fixed_leg.items()},
            "timestamp": self.timestamp
        }
    
    def to_flat_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for backward compatibility."""
        result = {}
        for t, v in self.floating_leg.items():
            result[f"DELTA.{self.instrument_type.value}.FLOATING.T{t:.2f}"] = v
        for t, v in self.fixed_leg.items():
            result[f"DELTA.{self.instrument_type.value}.FIXED.T{t:.2f}"] = v
        result[f"DELTA.{self.instrument_type.value}.PRICE"] = self.price
        return result
    
    def summary(self, precision: int = 6) -> str:
        lines = [
            f"\n{'='*60}",
            f"Delta Hedging Strategy ({self.instrument_type.value})",
            f"{'='*60}",
            f"  Strike:     {self.strike:.4f}",
            f"  Price:      {self.price:+.{precision}f}",
            f"{'-'*60}",
            f"  {'Leg':<12} {'Maturity':<12} {'Delta':>16}",
            f"{'-'*60}",
        ]
        
        for t, delta in sorted(self.floating_leg.items()):
            lines.append(f"  {'Floating':<12} {t:<12.4f} {delta:>+16.{precision}f}")
        
        for t, delta in sorted(self.fixed_leg.items()):
            lines.append(f"  {'Fixed':<12} {t:<12.4f} {delta:>+16.{precision}f}")
        
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# =============================================================================
# Vega Hedging Results
# =============================================================================

@dataclass
class VegaHedgingResult:
    """Structured Vega hedging result."""
    strike: float
    component_i: int
    component_j: int
    instrument_type: InstrumentType
    vega_value: float
    floating_leg: Dict[float, float] = field(default_factory=dict)
    fixed_leg: Dict[float, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def component(self) -> str:
        return f"({self.component_i},{self.component_j})"
    
    def to_dict(self) -> Dict:
        return {
            "greek": "VEGA",
            "strike": self.strike,
            "component": self.component,
            "instrument_type": self.instrument_type.value,
            "vega_value": self.vega_value,
            "floating_leg": {f"T={k:.2f}": v for k, v in self.floating_leg.items()},
            "fixed_leg": {f"T={k:.2f}": v for k, v in self.fixed_leg.items()},
            "timestamp": self.timestamp
        }
    
    def to_flat_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for backward compatibility."""
        result = {}
        comp = f"{self.component_i}_{self.component_j}"
        for t, v in self.floating_leg.items():
            result[f"VEGA.{comp}.{self.instrument_type.value}.FLOATING.T{t:.2f}"] = v
        for t, v in self.fixed_leg.items():
            result[f"VEGA.{comp}.{self.instrument_type.value}.FIXED.T{t:.2f}"] = v
        result[f"VEGA.{comp}.{self.instrument_type.value}.VALUE"] = self.vega_value
        return result
    
    def summary(self, precision: int = 6) -> str:
        lines = [
            f"\n{'='*60}",
            f"Vega Hedging Strategy ({self.instrument_type.value})",
            f"{'='*60}",
            f"  Strike:     {self.strike:.4f}",
            f"  Component:  σ{self.component}",
            f"  Vega:       {self.vega_value:+.{precision}f}",
            f"{'-'*60}",
            f"  {'Leg':<12} {'Maturity':<12} {'Sensitivity':>16}",
            f"{'-'*60}",
        ]
        
        for t, val in sorted(self.floating_leg.items()):
            lines.append(f"  {'Floating':<12} {t:<12.4f} {val:>+16.{precision}f}")
        
        for t, val in sorted(self.fixed_leg.items()):
            lines.append(f"  {'Fixed':<12} {t:<12.4f} {val:>+16.{precision}f}")
        
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# =============================================================================
# Gamma Results
# =============================================================================

@dataclass
class GammaResult:
    """Structured Gamma result."""
    strike: float
    instrument_type: InstrumentType
    first_component: Union[int, str]
    second_component: Union[int, str]
    gamma_value: float
    first_maturity: Optional[float] = None
    second_maturity: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def component_pair(self) -> str:
        if isinstance(self.first_component, str):
            return f"{self.first_component}-{self.second_component}"
        return f"T{self.first_maturity:.2f}-T{self.second_maturity:.2f}"
    
    def to_dict(self) -> Dict:
        return {
            "greek": "GAMMA",
            "strike": self.strike,
            "instrument_type": self.instrument_type.value,
            "first_component": self.first_component,
            "second_component": self.second_component,
            "first_maturity": self.first_maturity,
            "second_maturity": self.second_maturity,
            "gamma_value": self.gamma_value,
            "timestamp": self.timestamp
        }
    
    def to_flat_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for backward compatibility."""
        return {
            f"GAMMA.{self.instrument_type.value}.{self.component_pair}": self.gamma_value
        }
    
    def summary(self, precision: int = 6) -> str:
        lines = [
            f"\n{'='*60}",
            f"Gamma Sensitivity ({self.instrument_type.value})",
            f"{'='*60}",
            f"  Strike:          {self.strike:.4f}",
            f"  Component Pair:  {self.component_pair}",
            f"  Gamma Value:     {self.gamma_value:+.{precision}f}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# =============================================================================
# Alpha Sensitivity Result
# =============================================================================

@dataclass
class AlphaSensitivityResult:
    """Structured Alpha sensitivity result."""
    strike: float
    alpha_sensitivity: float
    option_price: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "greek": "ALPHA",
            "strike": self.strike,
            "alpha_sensitivity": self.alpha_sensitivity,
            "option_price": self.option_price,
            "timestamp": self.timestamp
        }
    
    def to_flat_dict(self) -> Dict[str, float]:
        return {
            "ALPHA.SENSITIVITY": self.alpha_sensitivity,
            "ALPHA.OPTION_PRICE": self.option_price
        }
    
    def summary(self, precision: int = 6) -> str:
        lines = [
            f"\n{'='*60}",
            f"Alpha Sensitivity",
            f"{'='*60}",
            f"  Strike:            {self.strike:.4f}",
            f"  Alpha Sensitivity: {self.alpha_sensitivity:+.{precision}f}",
            f"  Option Price:      {self.option_price:+.{precision}f}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


# =============================================================================
# Comprehensive Sensitivity Report
# =============================================================================

@dataclass
class SensitivityReport:
    """Comprehensive report containing all computed sensitivities."""
    strike: float
    maturity: float
    tenor: float
    delta_results: List[DeltaHedgingResult] = field(default_factory=list)
    vega_results: List[VegaHedgingResult] = field(default_factory=list)
    gamma_results: List[GammaResult] = field(default_factory=list)
    matrix_results: List[MatrixResult] = field(default_factory=list)
    alpha_result: Optional[AlphaSensitivityResult] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def add_delta(self, result: DeltaHedgingResult):
        self.delta_results.append(result)
    
    def add_vega(self, result: VegaHedgingResult):
        self.vega_results.append(result)
    
    def add_gamma(self, result: GammaResult):
        self.gamma_results.append(result)
    
    def add_matrix(self, result: MatrixResult):
        self.matrix_results.append(result)
    
    def set_alpha(self, result: AlphaSensitivityResult):
        self.alpha_result = result
    
    def to_dict(self) -> Dict:
        return {
            "metadata": {
                "strike": self.strike,
                "maturity": self.maturity,
                "tenor": self.tenor,
                "timestamp": self.timestamp
            },
            "delta": [r.to_dict() for r in self.delta_results],
            "vega": [r.to_dict() for r in self.vega_results],
            "gamma": [r.to_dict() for r in self.gamma_results],
            "matrix_sensitivities": [r.to_dict() for r in self.matrix_results],
            "alpha": self.alpha_result.to_dict() if self.alpha_result else None
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_flat_dict(self) -> Dict[str, float]:
        """Convert all results to flat dictionary for backward compatibility."""
        result = {}
        for delta in self.delta_results:
            result.update(delta.to_flat_dict())
        for vega in self.vega_results:
            result.update(vega.to_flat_dict())
        for gamma in self.gamma_results:
            result.update(gamma.to_flat_dict())
        if self.alpha_result:
            result.update(self.alpha_result.to_flat_dict())
        return result
    
    def summary(self) -> str:
        lines = [
            f"\n{'#'*70}",
            f"#  SENSITIVITY REPORT",
            f"#  Strike: {self.strike:.4f}  |  Maturity: {self.maturity:.2f}Y  |  Tenor: {self.tenor:.2f}Y",
            f"#  Generated: {self.timestamp}",
            f"{'#'*70}",
        ]
        
        if self.alpha_result:
            lines.append(self.alpha_result.summary())
        
        for delta in self.delta_results:
            lines.append(delta.summary())
        
        for matrix in self.matrix_results:
            lines.append(matrix.summary())
        
        for vega in self.vega_results:
            lines.append(vega.summary())
        
        for gamma in self.gamma_results:
            lines.append(gamma.summary())
        
        lines.append(f"\n{'#'*70}")
        lines.append(f"#  END OF REPORT")
        lines.append(f"{'#'*70}\n")
        
        return "\n".join(lines)


# =============================================================================
# Logging Utilities
# =============================================================================

class SensitivityLogger:
    """Professional logging for sensitivity calculations."""
    
    def __init__(self, name: str = "sensitivities", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_calculation_start(self, greek: str, strike: float, **kwargs):
        extra_info = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        self.logger.info(f"Starting {greek} calculation | Strike={strike:.4f} | {extra_info}")
    
    def log_calculation_complete(self, result: Union[DeltaHedgingResult, VegaHedgingResult, 
                                                     GammaResult, MatrixResult, AlphaSensitivityResult]):
        if isinstance(result, DeltaHedgingResult):
            self.logger.info(f"Delta complete | Price={result.price:+.6f} | "
                           f"Floating legs={len(result.floating_leg)} | Fixed legs={len(result.fixed_leg)}")
        elif isinstance(result, VegaHedgingResult):
            self.logger.info(f"Vega complete | Component={result.component} | "
                           f"Value={result.vega_value:+.6f}")
        elif isinstance(result, GammaResult):
            self.logger.info(f"Gamma complete | Pair={result.component_pair} | "
                           f"Value={result.gamma_value:+.6f}")
        elif isinstance(result, MatrixResult):
            self.logger.info(f"{result.greek_type.value} matrix complete | "
                           f"Shape={result.values.shape} | "
                           f"Frobenius norm={np.linalg.norm(result.values):.6f}")
        elif isinstance(result, AlphaSensitivityResult):
            self.logger.info(f"Alpha complete | Sensitivity={result.alpha_sensitivity:+.6f}")
    
    def log_iteration(self, greek: str, i: int, j: int, value: float):
        self.logger.debug(f"{greek}[{i},{j}] = {value:+.6f}")
    
    def log_report(self, report: SensitivityReport):
        self.logger.info(f"Full report generated | "
                        f"Delta={len(report.delta_results)} | "
                        f"Vega={len(report.vega_results)} | "
                        f"Gamma={len(report.gamma_results)} | "
                        f"Matrix={len(report.matrix_results)}")


# =============================================================================
# Factory Functions for Easy Result Creation
# =============================================================================

def create_delta_result(model, instrument_type: str, price: float,
                       floating_leg: Dict[float, float],
                       fixed_leg: Dict[float, float]) -> DeltaHedgingResult:
    """Factory function to create DeltaHedgingResult from model."""
    return DeltaHedgingResult(
        strike=model.strike,
        instrument_type=InstrumentType(instrument_type),
        price=price,
        floating_leg=floating_leg,
        fixed_leg=fixed_leg
    )


def create_vega_result(model, i: int, j: int, instrument_type: str,
                      vega_value: float, floating_leg: Dict[float, float],
                      fixed_leg: Dict[float, float]) -> VegaHedgingResult:
    """Factory function to create VegaHedgingResult from model."""
    return VegaHedgingResult(
        strike=model.strike,
        component_i=i,
        component_j=j,
        instrument_type=InstrumentType(instrument_type),
        vega_value=vega_value,
        floating_leg=floating_leg,
        fixed_leg=fixed_leg
    )


def create_gamma_result(model, instrument_type: str, gamma_value: float,
                       first_component: Union[int, str],
                       second_component: Union[int, str],
                       first_maturity: Optional[float] = None,
                       second_maturity: Optional[float] = None) -> GammaResult:
    """Factory function to create GammaResult from model."""
    return GammaResult(
        strike=model.strike,
        instrument_type=InstrumentType(instrument_type),
        first_component=first_component,
        second_component=second_component,
        gamma_value=gamma_value,
        first_maturity=first_maturity,
        second_maturity=second_maturity
    )


def create_matrix_result(greek_type: str, strike: float,
                        values: np.ndarray,
                        component_details: Optional[Dict] = None) -> MatrixResult:
    """Factory function to create MatrixResult."""
    return MatrixResult(
        greek_type=GreekType(greek_type),
        strike=strike,
        values=values,
        component_details=component_details or {}
    )
