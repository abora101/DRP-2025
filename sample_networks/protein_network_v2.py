import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.patches import FancyArrowPatch

@dataclass
class HillInteraction:
    """Represents a Hill function-based interaction between proteins"""
    n: float  # Hill coefficient
    K: float  # Half-maximal activation constant
    type: str  # 'activation' or 'repression'

class ProteinNetwork:
    # [Previous __init__ and other methods remain the same until visualize_results]

    def visualize_results(self, 
                         simulation_results: Dict[str, np.ndarray], 
                         phase_plot_proteins: List[str] = None, 
                         use_tight_axes: bool = True,
                         time_points: Optional[List[float]] = None):
        """
        Create network and time series visualizations with optional phase plot and time point markers
        
        Parameters:
        - simulation_results: Simulation results dictionary
        - phase_plot_proteins: Optional list of two proteins for phase plot
        - use_tight_axes: If True, adjusts phase plot axes to focus on the trajectory range
        - time_points: Optional list of time points to highlight in the visualizations
        """
        # Determine subplot layout based on phase plot
        if phase_plot_proteins:
            fig = plt.figure(figsize=(18, 12))
            subplot_layout = (2, 2)
        else:
            fig = plt.figure(figsize=(18, 8))
            subplot_layout = (1, 2)
        
        # Network subplot
        ax1 = plt.subplot(subplot_layout[0], subplot_layout[1], 1)
        
        # [Network visualization code remains the same...]
        pos = nx.spring_layout(self.graph, k=1.5, iterations=50)
        
        nx.draw_networkx_nodes(self.graph, pos, 
                        node_color='#87CEFA',
                        node_size=3000,
                        alpha=0.8,
                        ax=ax1)
        
        nx.draw_networkx_labels(self.graph, pos, font_size=14, font_weight='bold')
        
        for source, target in self.graph.edges():
            interaction = self.interactions[(source, target)]
            edge_color = '#228B22' if interaction.type == 'activation' else '#B22222'
        
            mid_x = (pos[source][0] + pos[target][0]) / 2
            mid_y = (pos[source][1] + pos[target][1]) / 2
        
            edge = FancyArrowPatch(
                pos[source],
                pos[target],
                arrowstyle='-|>' if interaction.type == 'activation' else '-[',
                color=edge_color,
                mutation_scale=25,
                shrinkA=20,
                shrinkB=20,
                connectionstyle="arc3,rad=0.1"
            )
            ax1.add_patch(edge)
        
            label = f'n={interaction.n:.1f}, K={interaction.K:.1f}'
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
            ax1.text(mid_x, mid_y, label, ha='center', va='center', fontsize=6, bbox=bbox_props)
        
        for node, (x, y) in pos.items():
            plt.text(x, y + 0.15, 
                f'({self.aggregation_types[node]})\nβ₀={self.beta_naughts[node]:.1f}',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.3'))
        
        ax1.set_title(f"{self.name} Regulatory Network", fontsize=16, fontweight='bold', pad=10)
        ax1.axis('off')
        
        ax1.set_xlim(min(pos[node][0] for node in pos) - 0.3, max(pos[node][0] for node in pos) + 0.3)
        ax1.set_ylim(min(pos[node][1] for node in pos) - 0.3, max(pos[node][1] for node in pos) + 0.3)
        
        # Time series subplot with time point markers
        ax2 = plt.subplot(subplot_layout[0], subplot_layout[1], 2)
        t = simulation_results['t']
        
        # Plot time series
        for protein in self.graph.nodes:
            line = ax2.plot(t, simulation_results[protein], 
                          label=f'{protein}',
                          linewidth=2)
            
            # Add markers for specific time points
            if time_points:
                color = line[0].get_color()  # Get the color of the current line
                for time_point in time_points:
                    # Find the closest time index
                    idx = np.abs(t - time_point).argmin()
                    ax2.plot(t[idx], simulation_results[protein][idx], 'o', 
                            color=color, markersize=8, 
                            label=f'{protein} at t={time_point}')
        
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Concentration', fontsize=12)
        ax2.set_title('Protein Concentrations Over Time', fontsize=16, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3)
        
        # Add vertical lines for time points
        if time_points:
            for time_point in time_points:
                ax2.axvline(x=time_point, color='gray', linestyle='--', alpha=0.5)
        
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Optional Phase Plot with time point markers
        if phase_plot_proteins and len(phase_plot_proteins) == 2:
            ax3 = plt.subplot(subplot_layout[0], subplot_layout[1], 3)
            
            protein_x, protein_y = phase_plot_proteins
            x_traj = simulation_results[protein_x]
            y_traj = simulation_results[protein_y]
            
            # Prepare initial conditions for other proteins
            fixed_levels = {p: simulation_results[p][0] for p in self.graph.nodes if p not in phase_plot_proteins}
            
            # Determine axis limits based on trajectory
            if use_tight_axes:
                x_min, x_max = min(x_traj), max(x_traj)
                y_min, y_max = min(y_traj), max(y_traj)
                x_padding = (x_max - x_min) * 0.1
                y_padding = (y_max - y_min) * 0.1
                x_range = np.linspace(x_min - x_padding, x_max + x_padding, 20)
                y_range = np.linspace(y_min - y_padding, y_max + y_padding, 20)
            else:
                x_range = np.linspace(0, max(x_traj)*1.2, 20)
                y_range = np.linspace(0, max(y_traj)*1.2, 20)
            
            # Vector field calculation
            X, Y = np.meshgrid(x_range, y_range)
            U = np.zeros_like(X)
            V = np.zeros_like(Y)
            
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    current_levels = fixed_levels.copy()
                    current_levels[protein_x] = X[i, j]
                    current_levels[protein_y] = Y[i, j]
                    
                    beta_x = self.get_production_rate(protein_x, current_levels, 0)
                    beta_y = self.get_production_rate(protein_y, current_levels, 0)
                    
                    U[i, j] = beta_x - self.removal_rates[protein_x] * X[i, j]
                    V[i, j] = beta_y - self.removal_rates[protein_y] * Y[i, j]
            
            # Plot vector field
            ax3.quiver(X, Y, U, V, color='black', alpha=0.3, width=0.002, scale=20)
            
            # Plot trajectory
            ax3.plot(x_traj, y_traj, 'b-', linewidth=2)
            ax3.plot(x_traj[0], y_traj[0], 'ro', label='Initial Point')
            ax3.plot(x_traj[-1], y_traj[-1], 'go', label='Final Point')
            
            # Add markers for specific time points in phase plot
            if time_points:
                for time_point in time_points:
                    idx = np.abs(t - time_point).argmin()
                    ax3.plot(x_traj[idx], y_traj[idx], 'mo', markersize=8,
                            label=f't={time_point}')
            
            ax3.set_xlabel(f'{protein_x} Concentration', fontsize=12)
            ax3.set_ylabel(f'{protein_y} Concentration', fontsize=12)
            ax3.set_title(f'Phase Portrait: {protein_x} vs {protein_y}', fontsize=16)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            if use_tight_axes:
                ax3.set_xlim(x_min - x_padding, x_max + x_padding)
                ax3.set_ylim(y_min - y_padding, y_max + y_padding)
        
        plt.tight_layout()
        plt.show()