import numpy as np
import pandas as pd
import os
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
    def plot_results(self):
        """
        Generates a visual proof of the mathematical model by plotting the raw data 
        against the predicted parametric curve.
        """
        if not self.optimal_params:
            logging.warning("Please run optimization before plotting.")
            return
            
        theta = np.radians(self.optimal_params['theta_deg'])
        X = self.optimal_params['X']
        M = self.optimal_params['M']
        
        # Generate the predicted curve smoothly
        t_vals = np.linspace(6, 60, 1000)
        x_pred = t_vals * np.cos(theta) - np.exp(M * np.abs(t_vals)) * np.sin(0.3 * t_vals) * np.sin(theta) + X
        y_pred = 42 + t_vals * np.sin(theta) + np.exp(M * np.abs(t_vals)) * np.sin(0.3 * t_vals) * np.cos(theta)

        # Plotting styling (Standard Academic / Matplotlib Default)
        plt.figure(figsize=(10, 6))
        
        # Use standard blue dots and a solid red line for a more traditional look
        plt.scatter(self.x, self.y, color='blue', alpha=0.5, label='Original Data (xy_data.csv)', s=15)
        plt.plot(x_pred, y_pred, color='red', linewidth=2, label='Predicted Parametric Curve')
        
        plt.title('Parametric Curve Fitting via Inverse Rotation')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save output and show interactively
        output_dir = 'plots'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}/visual_proof.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logging.info(f"Saved standard visualization as '{output_file}'.")
        print(f"\n[!] Success: Visual proof saved to {output_file}")
        
        # Display the interactive plot window
        plt.show()



    


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
