"""
Simple Logistic Equation: x' = rx(1-x)

The logistic equation models population growth with carrying capacity.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random
import timeit
from numba import njit, jit


@jit(fastmath=True, nopython=True)
def logistic_equation(x, r):
    """
    Calculate the logistic equation derivative x' = rx(1-x).
    
    Parameters:
    -----------
    x : float
        Current population size (normalized, 0 < x < 1)
    r : float
        Growth rate parameter
        
    Returns:
    --------
    float
        The derivative x' = rx(1-x)
    """
    return r * x * (1 - x)


def find_common_values(r, n_initial=10000, n_steps=10000):
    """
    Find common values reached by iterating the logistic equation
    with many initial guesses (vectorized version).
    
    Parameters:
    -----------
    r : float
        Growth rate parameter
    n_initial : int
        Number of initial guesses (default: 10000)
    n_steps : int
        Number of iteration steps (default: 10000)
        
    Returns:
    --------
    np.ndarray
        Array of final values (rounded to avoid floating point precision issues)
    """
    # Generate random initial conditions
    x = np.random.random(n_initial)
    
    # Vectorized iteration: x_{n+1} = r * x_n * (1 - x_n)
    for _ in range(n_steps):
        x = logistic_equation(x, r)

    # Filter out values that escaped to infinity or are outside [0,1]
    valid_values = x[np.isfinite(x) & (x >= 0) & (x <= 1)]
    
    # Round to avoid floating point precision issues
    rounded_values = np.round(valid_values, 6)
    
    # Count frequencies and analyze stability
    unique_values, counts = np.unique(rounded_values, return_counts=True)
    
    # Sort by frequency (most common first)
    sorted_indices = np.argsort(counts)[::-1]
    unique_values = unique_values[sorted_indices]
    counts = counts[sorted_indices]
    
    # Calculate relative frequencies
    total_count = len(rounded_values)
    relative_freqs = counts / total_count
    
    # Check if there are stable attractors (values that appear frequently)
    # A value is "stable" if it appears in at least 10% of iterations
    stable_threshold = 0.1
    stable_mask = relative_freqs >= stable_threshold
    
    if np.any(stable_mask):
        # There are stable attractors - only keep those
        stable_values = unique_values[stable_mask]
        stable_counts = counts[stable_mask]
        
        result_values = []
        for i in range(len(stable_values)):
            # Add each stable value multiple times based on its frequency
            for _ in range(stable_counts[i]):
                result_values.append(stable_values[i])
        
        return np.array(result_values)
    else:
        # No stable attractors (chaotic regime) - keep more values
        # Keep top 20% of values or at least 10 values
        # n_keep = max(10, len(unique_values) // 10)
        n_keep = len(unique_values)
        result_values = []
        
        for i in range(min(n_keep, len(unique_values))):
            # Add each value multiple times based on its frequency
            for _ in range(counts[i]):
                result_values.append(unique_values[i])
        
        return np.array(result_values)


def plot_common_values_vs_r(r_min=0.1, r_max=2.0, r_step=0.01, n_initial=100, n_steps=100):
    """
    Plot common values as a function of r for the logistic equation.
    
    Parameters:
    -----------
    r_min : float
        Minimum r value (default: 0.1)
    r_max : float
        Maximum r value (default: 2.0)
    r_step : float
        Step size for r (default: 0.01)
    n_initial : int
        Number of initial conditions (default: 100)
    n_steps : int
        Number of iteration steps (default: 1000)
    """
    r_values = np.arange(r_min, r_max + r_step, r_step)
    all_r = []
    all_common_values = []
    
    print(f"Sweeping r from {r_min} to {r_max} in steps of {r_step}")
    print(f"Total r values: {len(r_values)}")
    
    for i, r in enumerate(r_values):
        print(f"Running for r = {r}")
        # Find common values for this r
        common_values = find_common_values(r, n_initial, n_steps)
        
        # Store ALL values for this r directly (skip if empty)
        if len(common_values) > 0:
            for value in common_values:
                all_r.append(r)
                all_common_values.append(value)
    
    # Create the plot with dark background
    # plt.style.use('dark_background')
    # plt.figure(figsize=(12, 8))
    # plt.scatter(all_r, all_common_values, alpha=0.6, s=0.05, color='white')
    # plt.xlabel('r', fontsize=12, fontfamily='Comic Sans MS')
    # plt.ylabel('$x_{eq}$', fontsize=12, fontfamily='Comic Sans MS')
    # plt.xlim(r_min, r_max)
    
    # # Add equation text
    # plt.text(0.1, 0.9, 'x\' = rx(1-x)', transform=plt.gca().transAxes, 
    #          fontsize=20, fontfamily='Comic Sans MS', color='yellow',
    #          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    # plt.ylim(0, 1)
    # plt.tight_layout()
    # plt.show()
    
    # Create clean plot without labels, ticks, or equations - save as high-quality SVG
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 8))
    plt.scatter(all_r, all_common_values, alpha=0.6, s=0.05, color='white')
    plt.xlim(r_min, r_max)
    plt.ylim(0, 1)
    
    # Remove all ticks and labels
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    
    # Remove spines
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('logistic_bifurcation_clean.svg', format='svg', dpi=150, bbox_inches='tight', 
                facecolor='none', edgecolor='none', transparent=True)
    plt.savefig('logistic_bifurcation_clean.png', format='png', dpi=600, bbox_inches='tight',
                facecolor='none', edgecolor='none', transparent=True)
    plt.show()
    
    return all_r, all_common_values


# Example usage
if __name__ == "__main__":
    # Plot common values vs r with timing
    print("\nPlotting common values vs growth rate...")
    
    # Time the computation
    start_time = timeit.default_timer()
    plot_common_values_vs_r(r_min=2.8, r_max=4.0, r_step=0.0001, n_initial=100, n_steps=1000)
    end_time = timeit.default_timer()
    
    print(f"\nTotal computation time: {end_time - start_time:.2f} seconds")