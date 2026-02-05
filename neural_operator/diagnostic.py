import numpy as np

# Paths - adjust as needed
train_path = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results\neural_operator\saved_models\train_data.npz"
train_path = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results\neural_operator\saved_models\train_data_complex.npz"
train_path = r"E:\OneDrive\Dropbox\LinearRationalWishart_Work\Code\ED\LinearRationalWishart\LinearRationalWishart_NewCode\Output_results\neural_operator\saved_models\val_data_complex.npz"
data = np.load(train_path)

theta_dim = data['theta'].shape[1]
B_dim = data['B'].shape[1] if data['B'].ndim > 1 else 1

print("=" * 50)
if theta_dim == 6 and B_dim == 2:
    print("✅ Your data is COMPLEX mode")
    print("   No need to regenerate for Fourier pricing!")
    print("\n   Just ensure:")
    print("   1. ur used in generation matches pricing ur (0.5)")
    print("   2. ui_max covers your integration range (typically 25+)")
    print("   3. Model was trained with mode='complex'")
elif theta_dim == 3 and B_dim == 1:
    print("❌ Your data is REAL mode")
    print("   Must regenerate with mode='complex' for pricing!")
else:
    print(f"⚠️  Unexpected shapes: theta={theta_dim}, B={B_dim}")
print("=" * 50)
