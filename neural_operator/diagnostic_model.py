import numpy as np
import jax.numpy as jnp

# Load model and test
from linear_rational_wishart.neural_operator.inference import WishartPINNInference
from linear_rational_wishart.neural_operator.model import WishartPINNModel

model_path = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results\neural_operator\saved_models\models_complex\final"
norm_path = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results\neural_operator\saved_models\normalization_stats_complex.npz"

# Load
inference = WishartPINNInference.from_saved_model(
    model_path,
    normalization_stats_path=norm_path
)

# Test with complex theta (as used in pricing)
d = 2
T = 1.0
ur = 0.5
ui = 5.0  # Mid-range value

a3 = np.eye(d)
theta = complex(ur, ui) * a3  # Complex theta!

m = np.array([[-0.5, 0.0], [0.0, -0.5]])
omega = np.array([[0.3, 0.05], [0.05, 0.3]])
sigma = np.array([[0.5, 0.1], [0.1, 0.5]])
x0 = np.eye(d)

print("Testing NN vs Numerical...")
print(f"theta = {complex(ur, ui)} * I")

try:
    # NN prediction
    A_nn, B_nn = inference.compute_A_B(T, theta, m, omega, sigma)
    phi_nn = np.exp(np.trace(A_nn @ x0) + B_nn)
    
    # Numerical ground truth
    A_num, B_num = inference.numerical_computer.compute_A_B(T, theta, m, omega, sigma, x0)
    phi_num = np.exp(np.trace(A_num @ x0) + B_num)
    
    # Errors
    err_A = np.linalg.norm(A_nn - A_num) / (np.linalg.norm(A_num) + 1e-10)
    err_B = np.abs(B_nn - B_num) / (np.abs(B_num) + 1e-10)
    err_phi = np.abs(phi_nn - phi_num) / (np.abs(phi_num) + 1e-10)
    
    print(f"\nResults:")
    print(f"  A_nn  = \n{A_nn}")
    print(f"  A_num = \n{A_num}")
    print(f"  Error A: {err_A:.2e}")
    print(f"  Error B: {err_B:.2e}")
    print(f"  Error Φ: {err_phi:.2e}")
    
    if err_phi < 0.01:
        print("\n✅ Model is working! No retrain needed.")
    elif err_phi < 0.1:
        print("\n⚠️  Model works but accuracy is marginal. Consider retraining.")
    else:
        print("\n❌ Model has high error. Need to retrain.")
        
except Exception as e:
    print(f"\n❌ Error during inference: {e}")
    print("   This usually means model/data mode mismatch. Need to retrain.")