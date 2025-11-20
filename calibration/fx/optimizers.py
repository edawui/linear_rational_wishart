"""
Optimization methods for FX model calibration.

This module provides various optimization algorithms for calibrating FX models,
including local and global optimization methods.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Dict, Any
import numpy as np
import time
from scipy import optimize

from .base import ParameterBounds, CalibrationResult
from .objectives import ObjectiveFunction


class OptimizerBase(ABC):
    """Abstract base class for optimizers."""
    
    def __init__(
        self,
        objective: ObjectiveFunction,
        bounds: ParameterBounds,
        **kwargs
    ):
        """
        Initialize optimizer.
        
        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function to minimize
        bounds : ParameterBounds
            Parameter bounds
        **kwargs
            Additional optimizer-specific parameters
        """
        self.objective = objective
        self.bounds = bounds
        self.options = kwargs
    
    @abstractmethod
    def optimize(
        self,
        initial_parameters: np.ndarray,
        **kwargs
    ) -> CalibrationResult:
        """
        Perform optimization.
        
        Parameters
        ----------
        initial_parameters : np.ndarray
            Starting point for optimization
        **kwargs
            Additional optimization parameters
            
        Returns
        -------
        CalibrationResult
            Optimization results
        """
        pass
    
    def _create_result(
        self,
        scipy_result: Any,
        initial_parameters: np.ndarray,
        parameter_names: list,
        start_time: float
    ) -> CalibrationResult:
        """
        Create CalibrationResult from scipy optimization result.
        
        Parameters
        ----------
        scipy_result : scipy.optimize.OptimizeResult
            Result from scipy optimizer
        initial_parameters : np.ndarray
            Initial parameter values
        parameter_names : list
            Names of parameters
        start_time : float
            Optimization start time
            
        Returns
        -------
        CalibrationResult
            Standardized calibration result
        """
        optimization_time = time.time() - start_time
        
        # Calculate initial error
        self.objective.reset_counter()
        initial_errors = self.objective(initial_parameters)
        if isinstance(initial_errors, np.ndarray):
            initial_error = np.sqrt(np.mean(initial_errors**2))
        else:
            initial_error = float(initial_errors)
        
        # Calculate final error
        final_parameters = scipy_result.x
        final_errors = self.objective(final_parameters)
        if isinstance(final_errors, np.ndarray):
            final_error = np.sqrt(np.mean(final_errors**2))
        else:
            final_error = float(final_errors)
        
        return CalibrationResult(
            success=scipy_result.success,
            final_parameters=final_parameters,
            initial_parameters=initial_parameters,
            parameter_names=parameter_names,
            final_error=final_error,
            initial_error=initial_error,
            num_iterations=getattr(scipy_result, 'nit', 0),
            num_function_evaluations=self.objective.call_counter,
            optimization_time=optimization_time,
            optimizer_message=getattr(scipy_result, 'message', ''),
        )


class LeastSquaresOptimizer(OptimizerBase):
    """Trust Region Reflective least squares optimizer."""
    
    def __init__(
        self,
        objective: ObjectiveFunction,
        bounds: ParameterBounds,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        max_nfev: int = 1000,
        verbose: int = 2,
        min_cost_reduction: float = 1e-8,
        consecutive_small_steps: int = 2,
        early_stopping: bool = True,
        **kwargs
    ):
        """
        Initialize least squares optimizer.
        
        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function (should return array of residuals)
        bounds : ParameterBounds
            Parameter bounds
        ftol : float
            Function tolerance
        xtol : float
            Parameter tolerance
        gtol : float
            Gradient tolerance
        max_nfev : int
            Maximum function evaluations
        verbose : int
            Verbosity level (0, 1, or 2)
        **kwargs
            Additional options
        """
        super().__init__(objective, bounds, **kwargs)

        ftol = kwargs.get('ftol', ftol)
        xtol = kwargs.get('xtol', xtol)
        gtol = kwargs.get('gtol', gtol)
        max_nfev = kwargs.get('max_nfev', max_nfev)
        verbose = kwargs.get('verbose', verbose)


        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.max_nfev = max_nfev
        self.verbose = verbose

        self.min_cost_reduction = kwargs.get('min_cost_reduction', min_cost_reduction)
        self.consecutive_small_steps = kwargs.get('consecutive_small_steps', consecutive_small_steps)
        self.early_stopping = kwargs.get('early_stopping', early_stopping)
        
        # Tracking variables
        self._cost_history = []
        self._small_step_count = 0

    
    def optimize(
        self,
        initial_parameters: np.ndarray,
        parameter_names: Optional[list] = None,
        **kwargs
    ) -> CalibrationResult:
        """
        Perform least squares optimization.
        
        Parameters
        ----------
        initial_parameters : np.ndarray
            Starting point
        parameter_names : list, optional
            Names of parameters
        **kwargs
            Additional options
            
        Returns
        -------
        CalibrationResult
            Optimization results
        """
        if parameter_names is None:
            parameter_names = [f"param_{i}" for i in range(len(initial_parameters))]
        
        print(f"\n{'='*60}")
        print("LEAST SQUARES OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Initial parameters: {initial_parameters}")
        print(f"Bounds: {self.bounds.lower} to {self.bounds.upper}")
        
        if self.early_stopping:
            print(f"Early stopping: ON (min reduction: {self.min_cost_reduction:.0e}, "
                  f"max consecutive: {self.consecutive_small_steps})")
        
        start_time = time.time()
        self.objective.reset_counter()
        
        # Reset tracking variables
        self._cost_history = []
        self._small_step_count = 0
        
        if self.early_stopping:
            # Use a custom wrapper that checks stopping criteria
            result = self._optimize_with_early_stopping(initial_parameters, **kwargs)
        else:
            # Run optimization
            result = optimize.least_squares(
                self.objective,
                initial_parameters,
                bounds=self.bounds.to_scipy_bounds(),
                method='trf',
                ftol=self.ftol,
                xtol=self.xtol,
                gtol=self.gtol,
                max_nfev=self.max_nfev,
                verbose=self.verbose,
                **kwargs
            )
        
        print(f"\nOptimization completed:")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Function evaluations: {result.nfev}")
        print(f"Final parameters: {result.x}")
        
        return self._create_result(result, initial_parameters, parameter_names, start_time)

    def _optimize_with_early_stopping_old(self, initial_parameters, **kwargs):
        """Run optimization with early stopping by reducing max_nfev iteratively."""
        
        # Strategy: Run optimization in chunks and check progress
        chunk_size = max(5, self.max_nfev // 10)  # 10% chunks, minimum 5
        total_nfev = 0
        current_x = initial_parameters.copy()
        best_result = None

        # nb_chunck = self.max_nfev // chunk_size + (1 if self.max_nfev % chunk_size > 0 else 0)
        nb_chunck = len(list(range(0, self.max_nfev, chunk_size)))

        for chunk in range(0, self.max_nfev, chunk_size):
            remaining_evals = min(chunk_size, self.max_nfev - total_nfev)
            
            if remaining_evals <= 0:
                break
            
            print(f"\n nb_chunck = {nb_chunck} and running optimization chunk {chunk} with {remaining_evals} number of function evaluations ...")  
            # Run optimization chunk
            result = optimize.least_squares(
                self.objective,
                current_x,
                bounds=self.bounds.to_scipy_bounds(),
                method='trf',
                ftol=self.ftol,
                xtol=self.xtol,
                gtol=self.gtol,
                max_nfev=remaining_evals,
                # verbose=max(0, self.verbose - 1),  # Reduce verbosity for chunks
                verbose=  self.verbose,
                **kwargs
            )
            
            total_nfev += result.nfev
            best_result = result
            success=True    # result.success ##Sometimes result.success is false but we want to continue
            # Check early stopping
            # if success: # result.success:
            if not result.success:
                current_cost = result.cost
                self._cost_history.append(current_cost)
                
                # Check for small improvements
                if len(self._cost_history) > 1:
                    improvement = self._cost_history[-2] - self._cost_history[-1]
                    print(f"self._cost_history[-2] {self._cost_history[-2]}: self._cost_history[-1] = {self._cost_history[-1]:.8f}, Improvement = {improvement:.8f}, self.min_cost_reduction={self.min_cost_reduction}")
                    if improvement < self.min_cost_reduction:
                        self._small_step_count += 1
                    else:
                        self._small_step_count = 0
                    
                    print( f"Consecutive small steps: {self._small_step_count} (threshold: {self.consecutive_small_steps})")
                    # Early stopping
                    if self._small_step_count >= self.consecutive_small_steps:
                        if self.verbose >= 1:
                            print(f"Early stopping after {total_nfev} evaluations: "
                                  f"{self._small_step_count} consecutive small improvements")
                        break
                
                current_x = result.x
            else:
                # If optimization failed, stop
                break
        
        # Update result with total evaluations
        if best_result:
            best_result.nfev = total_nfev
            
        return best_result
    
    def _optimize_with_early_stopping(self, initial_parameters, **kwargs):
        """Recommended version fixing your specific issues."""
        """Run optimization with early stopping by reducing max_nfev iteratively."""
        max_chunk_size, max_optimizer_run=10, 5
        max_chunk_size, max_optimizer_run=5, 10

        # Increase chunk size to allow more progress per chunk
        # chunk_size = max(10, self.max_nfev // 5)  # Fewer, larger chunks
        chunk_size = max(max_chunk_size, self.max_nfev // max_optimizer_run)  # Fewer, larger chunks
        total_nfev = 0
        current_x = initial_parameters.copy()
        best_result = None
    
        # Relax early stopping criteria
        min_cost_reduction = max(self.min_cost_reduction, 1e-10)  # Less strict
        consecutive_small_steps = 2 #max(self.consecutive_small_steps, 3)  # More patience
    
        chunk_number = 0
        total_chunk = len(list(range(0, self.max_nfev, chunk_size)))
        for chunk_start in range(0, self.max_nfev, chunk_size):
            remaining_evals = min(chunk_size, self.max_nfev - total_nfev)
        
            if remaining_evals <= 0:
                break
        
            chunk_number += 1
            print(f"\nChunk {chunk_number}: out of {total_chunk} total chunks, and with {remaining_evals} max evaluations...")
        
            result = optimize.least_squares(
                self.objective,
                current_x,
                bounds=self.bounds.to_scipy_bounds(),
                method='trf',
                ftol=self.ftol,
                xtol=self.xtol,
                gtol=self.gtol,
                max_nfev=remaining_evals,
                verbose=self.verbose,
                **kwargs
            )
        
            total_nfev += result.nfev
            best_result = result
        
            print(f"Chunk {chunk_number}: {result.nfev} evals, cost: {result.cost:.8e}")
        
            # Early stopping logic
            self._cost_history.append(result.cost)
        
            if len(self._cost_history) > 1:
                improvement = self._cost_history[-2] - self._cost_history[-1]
            
                if improvement < min_cost_reduction:
                    self._small_step_count += 1
                    print(f"Small improvement: {improvement:.2e} < {min_cost_reduction:.2e}")
                else:
                    self._small_step_count = 0
                    print(f"Good improvement: {improvement:.2e}")
            
                print(f"Small steps: {self._small_step_count}/{consecutive_small_steps}")
            
                if self._small_step_count >= consecutive_small_steps:
                    print(f"Early stopping after {total_nfev} evaluations")
                    break
        
            current_x = result.x
    
        if best_result:
            best_result.nfev = total_nfev
    
        return best_result

class DifferentialEvolutionOptimizer(OptimizerBase):
    """Global optimization using Differential Evolution."""
    
    def __init__(
        self,
        objective: ObjectiveFunction,
        bounds: ParameterBounds,
        maxiter: int = 300,
        popsize: int = 15,
        atol: float = 1e-8,
        tol: float = 1e-6,
        seed: int = 42,
        updating: str = 'immediate',
        polish: bool = True,
        workers: int = 1,
        **kwargs
    ):
        """
        Initialize differential evolution optimizer.
        
        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function (should return scalar)
        bounds : ParameterBounds
            Parameter bounds
        maxiter : int
            Maximum iterations
        popsize : int
            Population size multiplier
        atol : float
            Absolute tolerance
        tol : float
            Relative tolerance
        seed : int
            Random seed
        updating : str
            Population updating strategy
        polish : bool
            Whether to polish result with local optimizer
        workers : int
            Number of parallel workers
        **kwargs
            Additional options
        """
        super().__init__(objective, bounds, **kwargs)

        maxiter = kwargs.get('maxiter', maxiter)
        popsize = kwargs.get('popsize', popsize)
        atol = kwargs.get('atol', atol)
        tol = kwargs.get('tol', tol)
        seed = kwargs.get('seed', seed)
        updating = kwargs.get('updating', updating)
        polish = kwargs.get('polish', polish)
        workers = kwargs.get('workers', workers)
        

        self.maxiter = maxiter
        self.popsize = popsize
        self.atol = atol
        self.tol = tol
        self.seed = seed
        self.updating = updating
        self.polish = polish
        self.workers = workers
    
    def optimize(
        self,
        initial_parameters: np.ndarray,
        parameter_names: Optional[list] = None,
        **kwargs
    ) -> CalibrationResult:
        """
        Perform differential evolution optimization.
        
        Parameters
        ----------
        initial_parameters : np.ndarray
            Initial guess (used for comparison)
        parameter_names : list, optional
            Names of parameters
        **kwargs
            Additional options
            
        Returns
        -------
        CalibrationResult
            Optimization results
        """
        if parameter_names is None:
            parameter_names = [f"param_{i}" for i in range(len(initial_parameters))]
        
        print(f"\n{'='*60}")
        print("DIFFERENTIAL EVOLUTION OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Population size: {self.popsize * len(initial_parameters)}")
        print(f"Max iterations: {self.maxiter}")
        print(f"Polish: {self.polish}")
        
        start_time = time.time()
        self.objective.reset_counter()
        
        # Objective wrapper for scalar output
        def objective_wrapper(params):
            try:
                errors = self.objective(params)
                if isinstance(errors, np.ndarray):
                    return np.sum(errors**2)
                return float(errors)
            except Exception as e:
                print(f"Error in objective evaluation: {e}")
                return 1e10
        
        # Run optimization
        result = optimize.differential_evolution(
            objective_wrapper,
            self.bounds.to_list_bounds(),
            maxiter=self.maxiter,
            popsize=self.popsize,
            atol=self.atol,
            tol=self.tol,
            seed=self.seed,
            updating=self.updating,
            polish=self.polish,
            workers=self.workers,
            **kwargs
        )
        
        print(f"\nOptimization completed:")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Function evaluations: {result.nfev}")
        print(f"Final parameters: {result.x}")
        print(f"Final objective: {result.fun:.8f}")
        
        return self._create_result(result, initial_parameters, parameter_names, start_time)


class HybridOptimizer(OptimizerBase):
    """Hybrid optimizer: global search followed by local refinement."""
    
    def __init__(
        self,
        objective: ObjectiveFunction,
        bounds: ParameterBounds,
        global_maxiter: int = 100,
        global_popsize: int = 10,
        local_ftol: float = 1e-6,#10,
        local_xtol: float = 1e-6,#10,
        local_gtol: float = 1e-6,#10,
        local_max_nfev: int = 500,
        **kwargs
    ):
        """
        Initialize hybrid optimizer.
        
        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function
        bounds : ParameterBounds
            Parameter bounds
        global_maxiter : int
            Max iterations for global search
        global_popsize : int
            Population size for global search
        local_ftol : float
            Function tolerance for local search
        local_xtol : float
            Parameter tolerance for local search
        local_gtol : float
            Gradient tolerance for local search
        local_max_nfev : int
            Max evaluations for local search
        **kwargs
            Additional options
        """
        super().__init__(objective, bounds, **kwargs)

        global_maxiter = kwargs.get('global_maxiter', global_maxiter)
        global_popsize = kwargs.get('global_popsize', global_popsize)
        local_ftol = kwargs.get('local_ftol', local_ftol)
        local_xtol = kwargs.get('local_xtol', local_xtol)
        local_gtol = kwargs.get('local_gtol', local_gtol)
        local_max_nfev = kwargs.get('local_max_nfev', local_max_nfev)
        

        self.global_maxiter = global_maxiter
        self.global_popsize = global_popsize
        self.local_ftol = local_ftol
        self.local_xtol = local_xtol
        self.local_gtol = local_gtol
        self.local_max_nfev = local_max_nfev
    
    def optimize(
        self,
        initial_parameters: np.ndarray,
        parameter_names: Optional[list] = None,
        **kwargs
    ) -> CalibrationResult:
        """
        Perform hybrid optimization.
        
        Parameters
        ----------
        initial_parameters : np.ndarray
            Initial guess
        parameter_names : list, optional
            Names of parameters
        **kwargs
            Additional options
            
        Returns
        -------
        CalibrationResult
            Optimization results
        """
        if parameter_names is None:
            parameter_names = [f"param_{i}" for i in range(len(initial_parameters))]
        
        print(f"\n{'='*60}")
        print("HYBRID OPTIMIZATION")
        print(f"{'='*60}")
        
        start_time = time.time()
        self.objective.reset_counter()
        
        # Phase 1: Global search
        print("\nPhase 1: Global search with Differential Evolution")
        
        def objective_wrapper(params):
            try:
                errors = self.objective(params)
                if isinstance(errors, np.ndarray):
                    return np.sum(errors**2)
                return float(errors)
            except Exception as e:
                return 1e10
        
        def print_progress(xk, convergence):
                """Print progress during optimization."""
                # xk: best solution vector so far
                # convergence: fractional value (0 to 1)
                global iteration_count
                iteration_count += 1
    
                # Evaluate objective to show current best
                current_best = objective_wrapper(xk)
    
                print(f"Iteration {iteration_count}: "
                      f"Best value = {current_best:.6e}, "
                      f"Convergence = {convergence:.6f}")
    
                return False  # Return True to stop early

        # Initialize counter
        iteration_count = 0
        global_result = optimize.differential_evolution(
            objective_wrapper,
            self.bounds.to_list_bounds(),
            maxiter=self.global_maxiter,
            popsize=self.global_popsize,
            atol=1e-6,
            tol=1e-4,
            seed=42,
            polish=False,
            disp=True   # Basic display
            # ,  callback=print_progress  # Add this!
        )
        
        print(f"Global search completed. Best objective: {global_result.fun:.8f}")
        
        # Phase 2: Local refinement
        print("\nPhase 2: Local refinement with Trust Region")
        
        try:
            # For least squares, use the array-returning objective
            local_result = optimize.least_squares(
                self.objective,
                global_result.x,
                bounds=self.bounds.to_scipy_bounds(),
                method='trf',
                ftol=self.local_ftol,
                xtol=self.local_xtol,
                gtol=self.local_gtol,
                max_nfev=self.local_max_nfev,
                verbose=2  # 0=silent, 1=termination, 2=progress
            )
            
            # Create combined result
            final_result = type('obj', (object,), {
                'x': local_result.x,
                'success': local_result.success,
                'message': f"Global: {global_result.message}; Local: {local_result.message}",
                'nfev': global_result.nfev + local_result.nfev,
                'nit': getattr(global_result, 'nit', 0) + getattr(local_result, 'nit', 0)
            })()
            
        except Exception as e:
            print(f"Local refinement failed: {e}. Using global result.")
            final_result = global_result
        
        print(f"\nHybrid optimization completed")
        print(f"Total function evaluations: {final_result.nfev}")
        
        return self._create_result(final_result, initial_parameters, parameter_names, start_time)


class DualAnnealingOptimizer(OptimizerBase):
    """Dual Annealing global optimizer."""
    
    def __init__(
        self,
        objective: ObjectiveFunction,
        bounds: ParameterBounds,
        maxiter: int = 100,#1000,
        seed: int = 42,
        no_local_search: bool = False,
        **kwargs
    ):
        """
        Initialize dual annealing optimizer.
        
        Parameters
        ----------
        objective : ObjectiveFunction
            Objective function
        bounds : ParameterBounds
            Parameter bounds
        maxiter : int
            Maximum iterations
        seed : int
            Random seed
        no_local_search : bool
            If True, skip local search
        **kwargs
            Additional options
        """
        super().__init__(objective, bounds, **kwargs)
        maxiter = kwargs.get('maxiter', maxiter)
        seed = kwargs.get('seed', seed)
        no_local_search = kwargs.get('no_local_search', no_local_search)
      
        self.maxiter = maxiter
        self.seed = seed
        self.no_local_search = no_local_search
    
    def optimize(
        self,
        initial_parameters: np.ndarray,
        parameter_names: Optional[list] = None,
        **kwargs
    ) -> CalibrationResult:
        """
        Perform dual annealing optimization.
        
        Parameters
        ----------
        initial_parameters : np.ndarray
            Initial guess
        parameter_names : list, optional
            Names of parameters
        **kwargs
            Additional options
            
        Returns
        -------
        CalibrationResult
            Optimization results
        """
        if parameter_names is None:
            parameter_names = [f"param_{i}" for i in range(len(initial_parameters))]
        
        print(f"\n{'='*60}")
        print("DUAL ANNEALING OPTIMIZATION")
        print(f"{'='*60}")
        print(f"Max iterations: {self.maxiter}")
        print(f"Local search: {not self.no_local_search}")
        
        start_time = time.time()
        self.objective.reset_counter()
        
        # Objective wrapper for scalar output
        def objective_wrapper(params):
            try:
                errors = self.objective(params)
                if isinstance(errors, np.ndarray):
                    return np.sum(errors**2)
                return float(errors)
            except Exception as e:
                return 1e10
        
        # Simple verbose callback
        def verbose_callback(x, f, context):
            if context % 100 == 0:  # Print every 100 iterations
                print(f"Iteration {context}: f = {f:.6e}")
            return False

        # Run optimization
        result = optimize.dual_annealing(
            objective_wrapper,
            self.bounds.to_list_bounds(),
            maxiter=self.maxiter,
            seed=self.seed,
            no_local_search=self.no_local_search,
            # callback=verbose_callback,  # Add this line
            callback=lambda x, f, context: print(f"Iter {context}: {f:.6e}") if context % 100 == 0 else None,
 
            **kwargs
        )
        
        print(f"\nOptimization completed:")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Function evaluations: {result.nfev}")
        print(f"Final parameters: {result.x}")
        print(f"Final objective: {result.fun:.8f}")
        
        return self._create_result(result, initial_parameters, parameter_names, start_time)


# Factory function for creating optimizers
def create_optimizer(
    optimizer_type: str,
    objective: ObjectiveFunction,
    bounds: ParameterBounds,
    **kwargs
) -> OptimizerBase:
    """
    Factory function for creating optimizers.
    
    Parameters
    ----------
    optimizer_type : str
        Type of optimizer ('least_squares', 'differential_evolution', 'hybrid', 'dual_annealing')
    objective : ObjectiveFunction
        Objective function
    bounds : ParameterBounds
        Parameter bounds
    **kwargs
        Additional optimizer-specific parameters
        
    Returns
    -------
    OptimizerBase
        Created optimizer
    """
    optimizers = {
        'least_squares': LeastSquaresOptimizer,
        'differential_evolution': DifferentialEvolutionOptimizer,
        'hybrid': HybridOptimizer,
        'dual_annealing': DualAnnealingOptimizer
    }
    
    if optimizer_type not in optimizers:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Available: {list(optimizers.keys())}")
    
    return optimizers[optimizer_type](objective, bounds, **kwargs)
