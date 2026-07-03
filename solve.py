import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import logging

# Configure logging for professional output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ParametricCurveFitter:
    """
    A mathematical solver to find the optimal parameters (theta, M, X) for the given curve.
    Uses algebraic inverse rotation to decouple time (t) and ensure O(N) evaluation complexity.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.x = None
        self.y = None
        self.optimal_params = None
        self._load_data()
        
    def _load_data(self):
        """Loads and prepares the dataset."""
        try:
            df = pd.read_csv(self.data_path)
            self.x = df['x'].values
            self.y = df['y'].values
            logging.info(f"Successfully loaded {len(self.x)} points from {self.data_path}.")
        except FileNotFoundError:
            logging.error(f"Could not find {self.data_path}. Please ensure it is in the same directory.")
            raise
        except Exception as e:
            logging.error(f"Failed to load data: {e}")
            raise

    def _inverse_rotation_loss(self, params: list) -> float:
        """
        Calculates the mean squared error based on geometric inverse rotation.
        params = [theta_deg, X, M]
        """
        theta_rad = np.radians(params[0])
        X = params[1]
        M = params[2]
        
        # Analytically extract 't' and 'A' via inverse rotation matrix
        # This completely avoids O(N*M) distance calculations
        t = (self.x - X) * np.cos(theta_rad) + (self.y - 42) * np.sin(theta_rad)
        A_actual = -(self.x - X) * np.sin(theta_rad) + (self.y - 42) * np.cos(theta_rad)
        
        # Expected A given the mathematical definition
        A_expected = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
        
        # Mean Squared Error
        return float(np.mean((A_actual - A_expected)**2))

    def optimize(self) -> dict:
        """
        Runs global optimization to find the exact hidden variables.
        """
        # Bounds: Theta (0, 50), X (0, 100), M (-0.05, 0.05)
        bounds = [(0, 50), (0, 100), (-0.05, 0.05)]
        logging.info("Starting Differential Evolution optimization...")
        
        result = differential_evolution(
            self._inverse_rotation_loss, 
            bounds=bounds, 
            seed=42, 
            popsize=50, 
            tol=1e-6
        )
        
        if result.success:
            self.optimal_params = {
                'theta_deg': result.x[0],
                'X': result.x[1],
                'M': result.x[2],
                'error': result.fun
            }
            logging.info(f"Optimization Converged! Error: {result.fun:.2e}")
            return self.optimal_params
        else:
            logging.error("Optimization failed to converge.")
            return {}

    


if __name__ == "__main__":
    # Ensure correct file path (assumes data is in the same directory as script)
    fitter = ParametricCurveFitter(data_path="d:/Project/kamaladhi/Dataset/xy_data.csv")
    
    # 1. Run optimization
    results = fitter.optimize()
    
    # 2. Print finalized results neatly
    print("\n" + "="*40)
    print("FINAL EXTRACTED VARIABLES:")
    print("="*40)
    print(f"Theta: {results['theta_deg']:.4f} degrees")
    print(f"X:     {results['X']:.4f}")
    print(f"M:     {results['M']:.4f}")
    print(f"MSE:   {results['error']:.2e}")
    print("="*40 + "\n")
    
    # 3. Generate Visual Proof
    fitter.plot_results()
