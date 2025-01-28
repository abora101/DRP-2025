import networkx as nx
import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, List
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
    def __init__(self, name, noise_level: float = 0.0):
        """Initialize an empty protein regulatory network"""
        self.name = name
        self.graph = nx.DiGraph()
        self.protein_levels = {}
        self.interactions = {}
        self.removal_rates = {}  # alpha values for each protein
        self.aggregation_types = {}  # 'AND' or 'OR' for each protein
        self.beta_naughts = {}  # Production rates for each protein
        self.external_signals = {}  # External signals for each protein
        self.noise_level = noise_level #noise amt is random normal * noise_level

    def get_noise_sample(self):
        return self.noise_level * np.random.normal(0, 1)
        
    def add_protein(self, name: str, initial_level: float = 1.0, removal_rate: float = 1.0, beta_naught: float = 1.0, aggregation_type: str = 'AND'):
        """Add a protein to the network"""
        if aggregation_type not in ['AND', 'OR']:
            raise ValueError(f"aggregation_type for protein {name} must be either 'AND' or 'OR'")
            
        self.graph.add_node(name)
        self.protein_levels[name] = initial_level
        self.removal_rates[name] = removal_rate
        self.beta_naughts[name] = beta_naught
        self.aggregation_types[name] = aggregation_type
        self.external_signals[name] = lambda t: True  # Default: always ON
    
    def add_interaction(self, source: str, target: str, n: float, K: float, interaction_type: str = 'activation'):
        """Add a directed interaction between proteins using Hill function regulation"""
        if not (source in self.graph and target in self.graph):
            raise ValueError("Both proteins must be added to network first")
            
        if self.graph.has_edge(source, target):
            raise ValueError(f"Interaction from {source} to {target} already exists")
            
        self.graph.add_edge(source, target)
        self.interactions[(source, target)] = HillInteraction(
            n=n,
            K=K,
            type=interaction_type
        )
    
    def set_external_signal(self, protein: str, signal_function):
        """Set the external signal function for a protein"""
        if protein not in self.graph:
            raise ValueError(f"Protein {protein} not in network")
        self.external_signals[protein] = signal_function
    
    def calculate_hill_regulation(self, source: str, source_level: float, interaction: HillInteraction, t: float) -> float:
        """Calculate regulatory effect using Hill function and external signal"""
        if not self.external_signals[source](t):
            hill_term = 0
        else:
            hill_term = (source_level ** interaction.n) / \
                   (interaction.K ** interaction.n + source_level ** interaction.n)
        
        if interaction.type == 'activation':
            return hill_term
        else:  # repression
            return 1 - hill_term
    
    def get_production_rate(self, protein: str, current_levels: Dict[str, float], t: float) -> float:
        """Calculate the total production rate (beta) for a protein"""
        predecessors = list(self.graph.predecessors(protein))
        
        if not predecessors:
            return self.beta_naughts[protein]  # baseline rate
        
        # Calculate individual regulatory effects
        regulations = []
        for source in predecessors:
            interaction = self.interactions[(source, protein)]
            source_level = current_levels[source]
            hill_term = self.calculate_hill_regulation(source, source_level, interaction, t)
            regulations.append(hill_term)
            
        # Combine effects based on aggregation type
        if self.aggregation_types[protein] == 'AND':
            # Product of all terms
            combined_effect = np.prod(regulations)
        else:  # 'OR'
            # In high n limit, this approaches logical OR
            combined_effect = max(regulations)
            
        return self.beta_naughts[protein] * combined_effect
    
    def system_equations(self, t: float, y: np.ndarray, protein_order: List[str], max_step: float = np.inf) -> np.ndarray:
        """Define the system of differential equations"""
        current_levels = {protein: level for protein, level in zip(protein_order, y)}
        derivatives = np.zeros_like(y)
        
        for i, protein in enumerate(protein_order):
            beta = self.get_production_rate(protein, current_levels, t)
            alpha = self.removal_rates[protein]
            derivatives[i] = beta - alpha * y[i] + self.get_noise_sample()

            max_derivative = -y[i]/max_step
            derivatives[i] = max(derivatives[i], max_derivative)
            
        return derivatives
    
    def simulate(self, t_span: Tuple[float, float], method: str = 'RK45', rtol: float = 1e-3, atol: float = 1e-6, max_step: float = np.inf) -> Dict[str, np.ndarray]:
        """Simulate the system using scipy.integrate.solve_ivp"""
        protein_order = list(self.graph.nodes)
        y0 = [self.protein_levels[protein] for protein in protein_order]
        
        solution = solve_ivp(
            fun=lambda t, y: self.system_equations(t, y, protein_order, max_step),
            t_span=t_span,
            y0=y0,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=max_step
        )
        
        results = {}
        for i, protein in enumerate(protein_order):
            results[protein] = solution.y[i]
            
        results['t'] = solution.t
        return results

    def visualize_results(self, simulation_results: Dict[str, np.ndarray], phase_plot_proteins: List[str] = None):
        """
        Create network and time series visualizations with optional phase plot
        
        Parameters:
        - simulation_results: Simulation results dictionary
        - phase_plot_proteins: Optional list of two proteins for phase plot
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
        
        # Existing network visualization code...
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
        
        ax1.set_title(f"{self.name} Protein Regulatory Network", fontsize=16, fontweight='bold', pad=10)
        ax1.axis('off')
        
        ax1.set_xlim(min(pos[node][0] for node in pos) - 0.3, max(pos[node][0] for node in pos) + 0.3)
        ax1.set_ylim(min(pos[node][1] for node in pos) - 0.3, max(pos[node][1] for node in pos) + 0.3)
        
        # Time series subplot
        ax2 = plt.subplot(subplot_layout[0], subplot_layout[1], 2)
        t = simulation_results['t']
        
        for protein in self.graph.nodes:
            ax2.plot(t, simulation_results[protein], 
                label=f'{protein} ({self.aggregation_types[protein]})',
                linewidth=2)
        
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Concentration', fontsize=12)
        ax2.set_title('Protein Concentrations Over Time', fontsize=16, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Optional Phase Plot
        if phase_plot_proteins and len(phase_plot_proteins) == 2:
            ax3 = plt.subplot(subplot_layout[0], subplot_layout[1], 3)
            
            protein_x, protein_y = phase_plot_proteins
            t = simulation_results['t']
            x_traj = simulation_results[protein_x]
            y_traj = simulation_results[protein_y]
            
            # Prepare initial conditions for other proteins
            fixed_levels = {p: simulation_results[p][0] for p in self.graph.nodes if p not in phase_plot_proteins}
            
            # Create vector field
            x_range = np.linspace(0, max(x_traj)*1.2, 20)
            y_range = np.linspace(0, max(y_traj)*1.2, 20)
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
            
            ax3.set_xlabel(f'{protein_x} Concentration', fontsize=12)
            ax3.set_ylabel(f'{protein_y} Concentration', fontsize=12)
            ax3.set_title(f'Phase Portrait: {protein_x} vs {protein_y}', fontsize=16)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        plt.tight_layout()
        plt.show()