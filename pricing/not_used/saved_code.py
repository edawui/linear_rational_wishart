 def  compute_swaption_exposure_profile_vectorized_0_toremove(
        self,
        exposure_dates: List[float],
        initial_swaption_config,
        nb_mc: int,
        dt: float,
        main_date: float = 0.0,
        schema: str = "EULER_FLOORED",
        pricing_method: str = "fft",

        batch_size: int = 500  # Increased batch size for better performance
        ) -> np.ndarray:
        """
        Compute swaption exposure profile with ultra-fast JAX vectorization.
        """
    
        exposure_profile = np.zeros((len(exposure_dates), nb_mc), dtype=np.float64)
    
        intial_maturity= initial_swaption_config.maturity
        
        # if schema == "ALFONSI":
        #     dt = np.min(floating_schedule_trade)
    
        # MC simulation
        start_time = time.perf_counter()
        time_list = exposure_dates #jnp.array(exposure_dates)
        mc_simulator= WishartMonteCarloPricer(self.model)
        sim_results_dict = mc_simulator.simulate(time_list, nb_mc, dt, schema)
        mc_time = time.perf_counter() - start_time
        print(f"MC simulation time: {mc_time:.4f} seconds")

        # Convert dict to array
        swap_start = time.perf_counter()
        sim_results = np.zeros((nb_mc, len(time_list), self.model.x0.shape[0], self.model.x0.shape[1]))
        for path_idx in range(nb_mc):
            for t_idx, t in enumerate(time_list):
                sim_results[path_idx, t_idx] = sim_results_dict[path_idx][t]
        swap_time = time.perf_counter() - swap_start
        print(f"Simulation swapping time: {swap_time:.4f} seconds")
    
        # Pricing
        pricing_start = time.perf_counter()
        for i, valuation_date in enumerate(exposure_dates):
            current_maturity = intial_maturity - valuation_date
            if current_maturity <= 0:
                continue
            
            print(f"  Date {valuation_date} ")

            for path in range(nb_mc):
                current_wishart = sim_results[path, i]
                # print(f" Path {path}, Date {valuation_date}, wishart {current_wishart}")
                
                new_lrw_model_config = self.model.model_config
                new_lrw_model_config.x0 = current_wishart
                new_swaption_config = initial_swaption_config
                new_swaption_config.maturity=current_maturity 
                # new_swaption_config = initial_swaption_config.replace(maturity=current_maturity)

                new_lrw_model = LRWModel(new_lrw_model_config,new_swaption_config)
                new_lrw_model.is_spread=self.model.is_spread
                new_lrw_model.set_swaption_config(new_swaption_config)
                new_lrw_model.set_weight_matrices(self.model.u1,self.model.u2)
                new_pricer = LRWSwaptionPricer(new_lrw_model)

                current_price=new_pricer.price_swaption(
                                            method = pricing_method,
                                            num_paths = nb_mc,
                                            dt=dt,
                                            return_implied_vol= False
                                            )
                exposure_profile[i, path] = current_price
                # exposure_profile[i, path] = 0.0 # Placeholder for current_price computation 
  
        pricing_time = time.perf_counter() - pricing_start
        print(f"Pricing time: {pricing_time:.4f} seconds")
        # print(f"exposure_dates: {len(exposure_dates)}, {exposure_dates}")
        # print(f"exposure_dates: {len(exposure_profile)},{exposure_profile}")
    
        return exposure_profile

    def compute_swaption_exposure_profile_vectorized_toremove(
            self,
            exposure_dates: List[float],
            initial_swaption_config,
            nb_mc: int,
            dt: float,
            main_date: float = 0.0,
            schema: str = "EULER_FLOORED",
            pricing_method: str = "fft",
            batch_size: int = 500
        ) -> np.ndarray:
        """
        Compute swaption exposure profile with ultra-fast JAX vectorization.
        """
        
    
        exposure_profile = np.zeros((len(exposure_dates), nb_mc), dtype=np.float64)
        initial_maturity = initial_swaption_config.maturity
    
        # MC simulation
        start_time = time.perf_counter()
        time_list = exposure_dates
        mc_simulator = WishartMonteCarloPricer(self.model)
        sim_results_dict = mc_simulator.simulate(time_list, nb_mc, dt, schema)
        mc_time = time.perf_counter() - start_time
        print(f"MC simulation time: {mc_time:.4f} seconds")
    
        # OPTIMIZED: More efficient dict to array conversion
        swap_start = time.perf_counter()
        sim_results = np.array([
            [sim_results_dict[path_idx][t] for t in time_list]
            for path_idx in range(nb_mc)
        ])
        swap_time = time.perf_counter() - swap_start
        print(f"Simulation swapping time: {swap_time:.4f} seconds")
    
        # OPTIMIZED: Reuse model config and swaption config objects
        pricing_start = time.perf_counter()
    
        # Pre-create reusable objects outside the loop
        new_lrw_model_config = self.model.model_config
        new_swaption_config = initial_swaption_config
    
        for i, valuation_date in enumerate(exposure_dates):
            current_maturity = initial_maturity - valuation_date
            if current_maturity <= 0:
                continue
        
            print(f"  Date {valuation_date}")
        
            # Update swaption config ONCE per date (not per path)
            new_swaption_config.maturity = current_maturity
        
            # OPTIMIZED: Batch processing of paths
            for batch_start in range(0, nb_mc, batch_size):
                batch_end = min(batch_start + batch_size, nb_mc)
            
                for path in range(batch_start, batch_end):
                    current_wishart = sim_results[path, i]
                
                    # Update model state
                    new_lrw_model_config.x0 = current_wishart
                
                    # Reuse model and pricer (or create if first iteration)
                    if path == batch_start:
                        # Create once per batch
                        new_lrw_model = LRWModel(new_lrw_model_config, new_swaption_config)
                        new_lrw_model.is_spread = self.model.is_spread
                        new_lrw_model.set_swaption_config(new_swaption_config)
                        new_lrw_model.set_weight_matrices(self.model.u1, self.model.u2)
                        new_pricer = LRWSwaptionPricer(new_lrw_model)
                    else:
                        # Just update the state
                        new_lrw_model.model_config.x0 = current_wishart
                
                    current_price = new_pricer.price_swaption(
                        # method="fft",
                        method = pricing_method,
                        num_paths=nb_mc,
                        dt=dt,
                        return_implied_vol=False
                    )
                    exposure_profile[i, path] = current_price
            
                # Clean up batch objects
                del new_lrw_model, new_pricer
                gc.collect()
    
        pricing_time = time.perf_counter() - pricing_start
        print(f"Pricing time: {pricing_time:.4f} seconds")
    
        return exposure_profile

    def compute_swaption_exposure_profile_vectorized_2_1_toremove(
            self,
            exposure_dates: List[float],
            initial_swaption_config,
            nb_mc: int,
            dt: float,
            main_date: float = 0.0,
            schema: str = "EULER_FLOORED",
            pricing_method: str = "fft",

            batch_size: int = 500
        ) -> np.ndarray:
            """
            Compute swaption exposure profile with ultra-fast vectorization.
            """
          
    
            exposure_profile = np.zeros((len(exposure_dates), nb_mc), dtype=np.float64)
            initial_maturity = initial_swaption_config.maturity
    
            # MC simulation
            start_time = time.perf_counter()
            mc_simulator = WishartMonteCarloPricer(self.model)
            sim_results_dict = mc_simulator.simulate(exposure_dates, nb_mc, dt, schema)
            mc_time = time.perf_counter() - start_time
            print(f"MC simulation time: {mc_time:.4f} seconds")
    
            # Vectorized conversion
            swap_start = time.perf_counter()
            sim_results = np.array([
                [sim_results_dict[path][t] for t in exposure_dates]
                for path in range(nb_mc)
            ])
            print(f"Simulation swapping time: {time.perf_counter() - swap_start:.4f} seconds")
    
            # Vectorized pricing
            pricing_start = time.perf_counter()
    
            for i, valuation_date in enumerate(exposure_dates):
                current_maturity = initial_maturity - valuation_date
                if current_maturity <= 0:
                    continue
        
                print(f"  Date {valuation_date}")
        
                # Process in batches to manage memory
                for batch_start in range(0, nb_mc, batch_size):
                    batch_end = min(batch_start + batch_size, nb_mc)
                    batch_wishart = sim_results[batch_start:batch_end, i]
            
                    # Vectorized pricing for the batch
                    batch_prices = self._price_swaption_batch(
                        batch_wishart,
                        current_maturity,
                        initial_swaption_config,
                        nb_mc,
                        dt,
                        pricing_method
                    )
            
                    exposure_profile[i, batch_start:batch_end] = batch_prices
            
                    # Memory cleanup
                    del batch_wishart, batch_prices
                    gc.collect()
    
            pricing_time = time.perf_counter() - pricing_start
            print(f"Pricing time: {pricing_time:.4f} seconds")
    
            return exposure_profile

    def _price_swaption_batch_toremove(
                self,
                wishart_batch: np.ndarray,
                maturity: float,
                initial_swaption_config,
                nb_mc: int,
                dt: float,
                pricing_method
            ) -> np.ndarray:
            """
            Price a batch of swaptions with different Wishart states.
    
            This should be implemented using vectorized operations if your pricer supports it.
            Otherwise, use the loop but with object reuse.
            """
            batch_size = wishart_batch.shape[0]
            prices = np.zeros(batch_size, dtype=np.float64)
    
            # Create config objects once
            model_config = self.model.model_config
            swaption_config = initial_swaption_config
            swaption_config.maturity = maturity
    
            # Reuse model and pricer
            model_config.x0 = wishart_batch[0]
            lrw_model = LRWModel(model_config, swaption_config)
            lrw_model.is_spread = self.model.is_spread
            lrw_model.set_weight_matrices(self.model.u1, self.model.u2)
            # pricer = LRWSwaptionPricer(lrw_model)
    
            for i, wishart in enumerate(wishart_batch):
                # Only update the state
                lrw_model.model_config.x0 = wishart
                lrw_model.set_wishart_parameter(lrw_model.model_config)
                pricer = LRWSwaptionPricer(lrw_model)

                prices[i] = pricer.price_swaption(
                    method = pricing_method,
                    # method="fft",
                    num_paths=nb_mc,
                    dt=dt,
                    return_implied_vol=False
                )
    
            # Clean up
            del lrw_model, pricer
    
            return prices

    def compute_swaption_exposure_profile_vectorized_2_2_toremove(
        self,
        exposure_dates: List[float],
        initial_swaption_config,
        nb_mc: int,
        dt: float,
        main_date: float = 0.0,
        schema: str = "EULER_FLOORED",
        pricing_method: str = "fft",
        batch_size: int = 50  # REDUCED from 500
    ) -> np.ndarray:
        """
        Memory-optimized swaption exposure profile computation.
        """
        # import time
        # import gc
    
        # Force garbage collection settings
        gc.set_threshold(100, 5, 5)  # Aggressive GC
    
        exposure_profile = np.zeros((len(exposure_dates), nb_mc), dtype=np.float64)
        initial_maturity = initial_swaption_config.maturity
    
        # MC simulation
        print("Starting MC simulation...")
        start_time = time.perf_counter()
        mc_simulator = WishartMonteCarloPricer(self.model)
        sim_results_dict = mc_simulator.simulate(exposure_dates, nb_mc, dt, schema)
        mc_time = time.perf_counter() - start_time
        print(f"MC simulation time: {mc_time:.4f} seconds")
    
        # Free the simulator immediately
        del mc_simulator
        gc.collect()
    
        # Convert dict to array in chunks to avoid memory spike
        swap_start = time.perf_counter()
        print("Converting simulation results...")
        sim_results = self._convert_sim_results_memory_efficient(
            sim_results_dict, exposure_dates, nb_mc
        )
    
        # Free the dict immediately
        del sim_results_dict
        gc.collect()
    
        swap_time = time.perf_counter() - swap_start
        print(f"Simulation conversion time: {swap_time:.4f} seconds")
    
        # Pricing with aggressive memory management
        pricing_start = time.perf_counter()
    
        # Pre-create ONE reusable model/pricer
        print("Creating reusable model...")
        reusable_config = self._create_reusable_config(initial_swaption_config)
    
        for i, valuation_date in enumerate(exposure_dates):
            current_maturity = initial_maturity - valuation_date
            if current_maturity <= 0:
                continue
        
            print(f"  Processing date {i+1}/{len(exposure_dates)}: {valuation_date:.4f}")
        
            # Update maturity once per date
            reusable_config['swaption_config'].maturity = current_maturity
        
            # Process in small batches
            for batch_start in range(0, nb_mc, batch_size):
                batch_end = min(batch_start + batch_size, nb_mc)
            
                # Price the batch
                exposure_profile[i, batch_start:batch_end] = self._price_batch_memory_safe(
                    sim_results[batch_start:batch_end, i],
                    reusable_config,
                    nb_mc,
                    dt
                )
            
                # Force garbage collection every batch
                if batch_start % (batch_size * 5) == 0:
                    gc.collect()
        
            # Clean up after each date
            gc.collect()
    
        # Final cleanup
        del sim_results
        gc.collect()
    
        pricing_time = time.perf_counter() - pricing_start
        print(f"Pricing time: {pricing_time:.4f} seconds")
    
        return exposure_profile

    def _convert_sim_results_memory_efficient_toremove(
            self,
            sim_results_dict: dict,
            time_list: List[float],
            nb_mc: int
        ) -> np.ndarray:
            """
            Convert dict to array without massive memory spike.
            """
            # import gc
    
            # Pre-allocate result
            sim_results = np.zeros(
                (nb_mc, len(time_list), self.model.x0.shape[0], self.model.x0.shape[1]),
                dtype=np.float64
            )
    
            # Convert in chunks
            chunk_size = 100
            for chunk_start in range(0, nb_mc, chunk_size):
                chunk_end = min(chunk_start + chunk_size, nb_mc)
        
                for path_idx in range(chunk_start, chunk_end):
                    for t_idx, t in enumerate(time_list):
                        sim_results[path_idx, t_idx] = sim_results_dict[path_idx][t]
        
                # Periodically clean up
                if chunk_start % (chunk_size * 5) == 0:
                    gc.collect()
    
            return sim_results

    def _create_reusable_config_toremove(self, initial_swaption_config):
        """
        Create reusable configuration objects.
        """
        # print(f"_create_reusable_config with u1={self.model.u1}, u2={self.model.u2}")

        return {
            'model_config': self.model.model_config,
            'swaption_config': initial_swaption_config,
            'u1': self.model.u1,
            'u2': self.model.u2,
            'is_spread': self.model.is_spread
        }

    def _price_batch_memory_safe_toremove(
            self,
            wishart_batch: np.ndarray,
            reusable_config: dict,
            nb_mc: int,
            dt: float
        ) -> np.ndarray:
            """
            Price batch with single model/pricer reuse.
            """
            # import gc
    
            batch_size = wishart_batch.shape[0]
            prices = np.zeros(batch_size, dtype=np.float64)
    
            # Create model and pricer ONCE for the batch
            model_config = reusable_config['model_config']
            swaption_config = reusable_config['swaption_config']
    
            # Update initial state
            model_config.x0 = wishart_batch[0]
    
            # Create reusable objects
            lrw_model = LRWModel(model_config, swaption_config)
            lrw_model.is_spread = reusable_config['is_spread']
            lrw_model.set_weight_matrices(reusable_config['u1'], reusable_config['u2'])
            pricer = LRWSwaptionPricer(lrw_model)
    
            # Price each path by updating state only
            for i, wishart in enumerate(wishart_batch):
                lrw_model.model_config.x0 = wishart
                ##These are important because withoutht this, the code seems to have some issue mainly on u1 and u2
                lrw_model.set_wishart_parameter(lrw_model.model_config)
                lrw_model.set_weight_matrices(reusable_config['u1'], reusable_config['u2'])

                try:
                    prices[i] = pricer.price_swaption(
                        method="fft",
                        num_paths=nb_mc,
                        dt=dt,
                        return_implied_vol=False
                    )
                except Exception as e:
                    print(f"    Warning: Pricing failed for path {i}: {e}")
                    prices[i] = 0.0
    
            # Explicit cleanup
            del lrw_model, pricer
            gc.collect()
    
            return prices

    def compute_swaption_exposure_profile_vectorized_2_toremove(
        self,
        exposure_dates: List[float],
        initial_swaption_config,
        nb_mc: int,
        dt: float,
        main_date: float = 0.0,
        schema: str = "EULER_FLOORED",
        pricing_method: str = "fft"
        ,  batch_size: int = 50  # REDUCED from 500
    ) -> np.ndarray:
        """
        Simplified memory-optimized version.
        Single model/pricer reused for all computations.
        """
        # import time
        # import gc
        # import psutil
    
        gc.set_threshold(100, 5, 5)
    
        exposure_profile = np.zeros((len(exposure_dates), nb_mc), dtype=np.float64)
        initial_maturity = initial_swaption_config.maturity
    
        # MC simulation
        print("MC simulation...")
        start = time.perf_counter()
        mc_simulator = WishartMonteCarloPricer(self.model)
        sim_results_dict = mc_simulator.simulate(exposure_dates, nb_mc, dt, schema)
        print(f"  Time: {time.perf_counter() - start:.2f}s")
    
        del mc_simulator
        gc.collect()
    
        # Convert
        print("Converting results...")
        start = time.perf_counter()
        sim_results = np.array([
            [sim_results_dict[path][t] for t in exposure_dates]
            for path in range(nb_mc)
        ])
        print(f"  Time: {time.perf_counter() - start:.2f}s")
    
        del sim_results_dict
        gc.collect()
    
        # Create SINGLE reusable model/pricer
        print("Pricing...")
        start = time.perf_counter()
    
        model_config = self.model.model_config
        swaption_config = initial_swaption_config
    
        model_config.x0 = sim_results[0, 0]
        swaption_config.maturity = initial_maturity
    
        lrw_model = LRWModel(model_config, swaption_config)
        lrw_model.is_spread = self.model.is_spread
 
        pricer = LRWSwaptionPricer(lrw_model)
        if exposure_dates[0]==0.0:
            initial_price=pricer.price_swaption(
                    method=pricing_method,
                    num_paths=nb_mc,
                    dt=dt,
                    return_implied_vol=False
                )
            for path in range(nb_mc):
                exposure_profile[0, path] = initial_price
            enumerate_start=1
        else:
            enumerate_start=0
        # Main pricing loop - simple and memory-efficient
        for i, valuation_date in enumerate(exposure_dates[enumerate_start:], start=enumerate_start):
            current_maturity = initial_maturity - valuation_date
            if current_maturity <= 0:
                continue
            print(f"  Processing date {i+1}/{len(exposure_dates)}: {valuation_date:.4f}")
        
            # Update maturity
            swaption_config.maturity = current_maturity
            lrw_model.set_swaption_config(swaption_config)
        
            # Price all paths
            for path in range(nb_mc):
                model_config.x0 = sim_results[path, i]
                lrw_model.set_wishart_parameter(model_config)
                # lrw_model.set_weight_matrices(self.model.u1, self.model.u2)
            
                exposure_profile[i, path] = pricer.price_swaption(
                    method=pricing_method,
                    num_paths=nb_mc,
                    dt=dt,
                    return_implied_vol=False
                )
                # Periodic cleanup
                if (path + 1) %50==0:# 10 == 0:
                    mem_pct_before = psutil.virtual_memory().percent
                    gc.collect()
                    mem_pct = psutil.virtual_memory().percent
                    print(f" Path {path},  Date {i+1}/{len(exposure_dates)}: mem_pct_before {mem_pct_before:.1f}%, and after {mem_pct:.1f}% RAM")
            # Periodic cleanup
            if (i + 1) %10==0:# 10 == 0:
                gc.collect()
                mem_pct = psutil.virtual_memory().percent
                print(f"  Date {i+1}/{len(exposure_dates)}: {mem_pct:.1f}% RAM")
    
        print(f"  Time: {time.perf_counter() - start:.2f}s")
    
        del lrw_model, pricer, sim_results
        gc.collect()
    
        return exposure_profile

    def compute_swaption_exposure_profile_vectorized_2_to_remove_anduseWithworkers_toremove(
        self,
        exposure_dates: List[float],
        initial_swaption_config,
        nb_mc: int,
        dt: float,
        main_date: float = 0.0,
        schema: str = "EULER_FLOORED",
        pricing_method: str = "fft",
        batch_size: int = 50
    ) -> np.ndarray:
        """
        Memory-optimized with pricer recreation to prevent leaks.
        """
        import time
        import gc
        import psutil
        import jax  # Import to check settings
        print("Starting swaption exposure profile computation...")
        # config.constants.NMAX=5
        
        print(f"NMAX={NMAX}, UR={UR}")
        # NMAX=5
        # WishartProcess.
        
        # print(f"NMAX={NMAX}, UR={UR}")
        # Verify JAX settings
        print("\nJAX Configuration Check:")
        print(f"  JIT disabled: {os.environ.get('JAX_DISABLE_JIT', 'NOT SET')}")
        print(f"  Memory fraction: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'NOT SET')}")
    
        gc.set_threshold(100, 5, 5)
    
        exposure_profile = np.zeros((len(exposure_dates), nb_mc), dtype=np.float64)
        initial_maturity = initial_swaption_config.maturity
    
        # MC simulation
        print("\nMC simulation...")
        start = time.perf_counter()
        mc_simulator = WishartMonteCarloPricer(self.model)
        sim_results_dict = mc_simulator.simulate(exposure_dates, nb_mc, dt, schema)
        print(f"  Time: {time.perf_counter() - start:.2f}s")
    
        del mc_simulator
        gc.collect()
    
        # Convert
        print("Converting results...")
        start = time.perf_counter()
        sim_results = np.array([
            [sim_results_dict[path][t] for t in exposure_dates]
            for path in range(nb_mc)
        ])
        print(f"  Time: {time.perf_counter() - start:.2f}s")
    
        del sim_results_dict
        gc.collect()
    
        # Pricing
        print("Pricing...")
        start = time.perf_counter()
    
        model_config = self.model.model_config
        swaption_config = initial_swaption_config
    
        # Handle initial date
        if exposure_dates[0] == 0.0:
            model_config.x0 = sim_results[0, 0]
            swaption_config.maturity = initial_maturity
        
            lrw_model = LRWModel(model_config, swaption_config)
            lrw_model.is_spread = self.model.is_spread
            pricer = LRWSwaptionPricer(lrw_model)
        
            initial_price = pricer.price_swaption(
                method=pricing_method,
                num_paths=nb_mc,
                dt=dt,
                return_implied_vol=False
            )
            exposure_profile[0, :] = initial_price
        
            del lrw_model, pricer
            gc.collect()
        
            enumerate_start = 1
        else:
            enumerate_start = 0
    
        # Create initial pricer
        lrw_model = None
        pricer = None
    
        # Main pricing loop with periodic pricer recreation
        for i, valuation_date in enumerate(exposure_dates[enumerate_start:], start=enumerate_start):
            current_maturity = initial_maturity - valuation_date
            if current_maturity <= 0:
                continue
        
            print(f"  Processing date {i+1}/{len(exposure_dates)}: {valuation_date:.4f}")
        
            # Update maturity
            swaption_config.maturity = current_maturity
        
            for path in range(nb_mc):
                # ? RECREATE PRICER EVERY 50 PATHS to prevent memory accumulation
                if path % batch_size == 0:
                    # Clean up old pricer
                    if pricer is not None:
                        del lrw_model, pricer
                        gc.collect()
                    
                        # Clear JAX cache if available
                        if hasattr(jax, 'clear_caches'):
                            jax.clear_caches()
                
                    # Create fresh pricer
                    model_config.x0 = sim_results[path, i]
                    lrw_model = LRWModel(model_config, swaption_config)
                    lrw_model.is_spread = self.model.is_spread
                    pricer = LRWSwaptionPricer(lrw_model)
                
                    mem_pct = psutil.virtual_memory().percent
                    print(f"    Recreated pricer at path {path}: {mem_pct:.1f}% RAM")
                else:
                    # Just update state
                    model_config.x0 = sim_results[path, i]
                    lrw_model.set_wishart_parameter(model_config)
            
                exposure_profile[i, path] = pricer.price_swaption(
                    method=pricing_method,
                    num_paths=nb_mc,
                    dt=dt,
                    return_implied_vol=False
                )
        
            # Cleanup after each date
            if pricer is not None:
                del lrw_model, pricer
                pricer = None
                lrw_model = None
        
            gc.collect()
            mem_pct = psutil.virtual_memory().percent
            print(f"  After date {i+1}: {mem_pct:.1f}% RAM")
    
        print(f"  Pricing time: {time.perf_counter() - start:.2f}s")
    
        del sim_results
        gc.collect()
    
        return exposure_profile
  
