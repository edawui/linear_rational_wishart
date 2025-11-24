import jax
# jax.config.update("jax_enable_x64", False)  # Use float32 instead of float64

from ast import Try
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import time
import matplotlib
try:
    from ..models.interest_rate.config import SwaptionConfig, LRWModelConfig
    from ..models.interest_rate.lrw_model import LRWModel
    from ..pricing.swaption_pricer import LRWSwaptionPricer
    from ..utils.reporting import print_pretty
    from ..components.jump  import JumpComponent
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from linear_rational_wishart.components.jump  import JumpComponent
    from linear_rational_wishart.models.interest_rate.config import SwaptionConfig, LRWModelConfig
    from linear_rational_wishart.models.interest_rate.lrw_model import LRWModel
    from linear_rational_wishart.pricing.swaption_pricer import LRWSwaptionPricer
    from linear_rational_wishart.utils.reporting import print_pretty


from lrw_jump_basic_examples import *
from lrw_basic_examples import * 


matplotlib.use('TkAgg')  # or 'Qt5Agg'

# Enable interactive mode globally
plt.ion()



if __name__ == "__main__":
    print( "Runing test examples for LRW model and pricer")
    # Run basic examples
    print("Linear Rational Wishart Model Basic Examples")
    # Run all examples
    example_basic_lrw_setup()     #ok working
    example_swaption_pricing()    #ok working
    example_term_structure()        #ok working
    example_parameter_comparison() #ok working
    example_wishart_properties() #ok working
    

    print("Linear Rational Wishart Model with Jump  Basic Examples")
    # Run all examples
    example_jump_basic_lrw_setup()     #ok working
    example_jump_swaption_pricing()    #ok working
    example_jump_term_structure()        #ok working
    example_jump_parameter_comparison() #ok working
    example_jump_wishart_properties() #ok working
    # Keep all plots open
    plt.ioff()  # Turn off interactive mode
    plt.show()  # This wi


    # Keep all plots open
    plt.ioff()  # Turn off interactive mode
    plt.show()  # This will
        
