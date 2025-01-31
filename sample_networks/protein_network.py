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
    
    def add_interaction(self, source: str, target: str, n: float = 10.0, K: float = 1.0, interaction_type: str = 'activation'):
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

    def visualize_results(self, 
                        simulation_results: Dict[str, np.ndarray], 
                        phase_plot_proteins: List[str] = None, 
                        use_tight_axes: bool = True,
                        time_points: List[float] = None,
                        display_until: float = None,
                        show_network: bool = True):
        """
        Create network and time series visualizations with optional phase plot and time point markers
        
        Parameters:
        - simulation_results: Simulation results dictionary
        - phase_plot_proteins: Optional list of two proteins for phase plot
        - use_tight_axes: If True, adjusts phase plot axes to focus on the trajectory range
        - time_points: Optional list of time points to highlight in the visualizations
        - display_until: Optional time point until which to display the trajectory
        - show_network: If True, displays the network diagram (default: True)
        """
        # Determine layout based on what we're showing
        if show_network:
            if phase_plot_proteins:
                fig = plt.figure(figsize=(24, 6))
                gs = plt.GridSpec(1, 3, width_ratios=[1.2, 1, 1])
                start_idx = 0
            else:
                fig = plt.figure(figsize=(16, 6))
                gs = plt.GridSpec(1, 2, width_ratios=[1.2, 1])
                start_idx = 0
        else:
            if phase_plot_proteins:
                fig = plt.figure(figsize=(16, 6))
                gs = plt.GridSpec(1, 2, width_ratios=[1, 1])
                start_idx = 0
            else:
                fig = plt.figure(figsize=(10, 6))
                gs = plt.GridSpec(1, 1)
                start_idx = 0

        # Network subplot (if shown)
        if show_network:
            ax1 = plt.subplot(gs[0])
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
            
            ax1.set_title(f"{self.name} Regulatory Network", fontsize=14, pad=10)
            ax1.axis('off')
            
            ax1.set_xlim(min(pos[node][0] for node in pos) - 0.3, max(pos[node][0] for node in pos) + 0.3)
            ax1.set_ylim(min(pos[node][1] for node in pos) - 0.3, max(pos[node][1] for node in pos) + 0.3)
            
            curr_pos = 1
        else:
            curr_pos = 0
        
        # Time series subplot
        ax2 = plt.subplot(gs[curr_pos])
        t = simulation_results['t']
        
        if display_until is not None:
            display_idx = np.searchsorted(t, display_until)
            for protein in self.graph.nodes:
                ax2.plot(t, simulation_results[protein], 
                        alpha=0.2, linestyle='--', color='gray')
        else:
            display_idx = len(t)
        
        for protein in self.graph.nodes:
            line = ax2.plot(t[:display_idx], simulation_results[protein][:display_idx], 
                        label=f'{protein}',
                        linewidth=2)
            
            if time_points:
                color = line[0].get_color()
                for time_point in time_points:
                    if time_point <= (display_until if display_until is not None else t[-1]):
                        idx = np.abs(t - time_point).argmin()
                        ax2.plot(t[idx], simulation_results[protein][idx], 'o', 
                                color=color, markersize=8, 
                                label=f'{protein} at t={time_point}')
        
        if display_until is not None:
            ax2.axvline(x=display_until, color='red', linestyle='--', alpha=0.7,
                    label=f'Current time (t={display_until})')
        
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Concentration', fontsize=12)
        ax2.set_title('Protein Concentrations Over Time', fontsize=14, pad=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
        
        # Phase Plot (if requested)
        if phase_plot_proteins and len(phase_plot_proteins) == 2:
            curr_pos += 1
            ax3 = plt.subplot(gs[curr_pos])
            protein_x, protein_y = phase_plot_proteins
            x_traj = simulation_results[protein_x]
            y_traj = simulation_results[protein_y]
            
            fixed_levels = {p: simulation_results[p][0] for p in self.graph.nodes if p not in phase_plot_proteins}
            
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
            
            ax3.quiver(X, Y, U, V, color='black', alpha=0.3, width=0.002, scale=20)
            
            if display_until is not None:
                ax3.plot(x_traj, y_traj, 'b-', linewidth=1, alpha=0.2)
            
            ax3.plot(x_traj[:display_idx], y_traj[:display_idx], 'b-', linewidth=2)
            ax3.plot(x_traj[0], y_traj[0], 'ro', label='Initial Point')
            
            if display_until is not None:
                current_x = x_traj[display_idx-1]
                current_y = y_traj[display_idx-1]
                ax3.plot(current_x, current_y, 'mo', markersize=8, label=f'Current (t={display_until})')
            else:
                ax3.plot(x_traj[-1], y_traj[-1], 'go', label='Final Point')
            
            if time_points:
                for time_point in time_points:
                    if time_point <= (display_until if display_until is not None else t[-1]):
                        idx = np.abs(t - time_point).argmin()
                        ax3.plot(x_traj[idx], y_traj[idx], 'ko', markersize=8,
                                label=f't={time_point}')
            
            ax3.set_xlabel(f'{protein_x} Concentration', fontsize=12)
            ax3.set_ylabel(f'{protein_y} Concentration', fontsize=12)
            ax3.set_title(f'Phase Portrait: {protein_x} vs {protein_y}', fontsize=14, pad=10)
            ax3.grid(True, alpha=0.3)
            ax3.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
            
            if use_tight_axes:
                ax3.set_xlim(x_min - x_padding, x_max + x_padding)
                ax3.set_ylim(y_min - y_padding, y_max + y_padding)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Leave room for legends
        plt.show()

    def visualize_results(self, 
                        simulation_results: Dict[str, np.ndarray], 
                        phase_plot_proteins: List[str] = None, 
                        use_tight_axes: bool = True,
                        time_points: List[float] = None,
                        display_until: float = None,
                        show_network: bool = True):
        """
        Create network and time series visualizations with optional phase plot and time point markers
        
        Parameters:
        - simulation_results: Simulation results dictionary
        - phase_plot_proteins: Optional list of two proteins for phase plot
        - use_tight_axes: If True, adjusts phase plot axes to focus on the trajectory range
        - time_points: Optional list of time points to highlight in the visualizations
        - display_until: Optional time point until which to display the trajectory
        - show_network: If True, displays the network diagram (default: True)
        """
        # Determine layout based on what we're showing
        if show_network:
            if phase_plot_proteins:
                fig = plt.figure(figsize=(24, 6))
                gs = plt.GridSpec(1, 3, width_ratios=[1.2, 1, 1])
                start_idx = 0
            else:
                fig = plt.figure(figsize=(16, 6))
                gs = plt.GridSpec(1, 2, width_ratios=[1.2, 1])
                start_idx = 0
        else:
            if phase_plot_proteins:
                fig = plt.figure(figsize=(16, 6))
                gs = plt.GridSpec(1, 2, width_ratios=[1, 1])
                start_idx = 0
            else:
                fig = plt.figure(figsize=(10, 6))
                gs = plt.GridSpec(1, 1)
                start_idx = 0

        # Network subplot (if shown)
        if show_network:
            ax1 = plt.subplot(gs[0])
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
            
            ax1.set_title(f"{self.name} Regulatory Network", fontsize=14, pad=10)
            ax1.axis('off')
            
            ax1.set_xlim(min(pos[node][0] for node in pos) - 0.3, max(pos[node][0] for node in pos) + 0.3)
            ax1.set_ylim(min(pos[node][1] for node in pos) - 0.3, max(pos[node][1] for node in pos) + 0.3)
            
            curr_pos = 1
        else:
            curr_pos = 0
        
        # Time series subplot
        ax2 = plt.subplot(gs[curr_pos])
        t = simulation_results['t']
        
        if display_until is not None:
            display_idx = np.searchsorted(t, display_until)
            for protein in self.graph.nodes:
                ax2.plot(t, simulation_results[protein], 
                        alpha=0.2, linestyle='--', color='gray')
        else:
            display_idx = len(t)
        
        # Plot time series
        protein_lines = {}
        for protein in self.graph.nodes:
            line = ax2.plot(t[:display_idx], simulation_results[protein][:display_idx], 
                        label=f'{protein}',
                        linewidth=2)
            protein_lines[protein] = line[0].get_color()
        
        # Add vertical lines for time points
        if time_points:
            for time_point in time_points:
                if time_point <= (display_until if display_until is not None else t[-1]):
                    ax2.axvline(x=time_point, color='black', linestyle=':', alpha=0.5)
        
        # Add vertical lines for initial and final points
        ax2.axvline(x=t[0], color='black', linestyle=':', alpha=0.5)
        if display_until is not None:
            ax2.axvline(x=display_until, color='red', linestyle='--', alpha=0.7)
        else:
            ax2.axvline(x=t[-1], color='black', linestyle=':', alpha=0.5)
        
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Concentration', fontsize=12)
        ax2.set_title('Protein Concentrations Over Time', fontsize=14, pad=10)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
        
        # Phase Plot (if requested)
        if phase_plot_proteins and len(phase_plot_proteins) == 2:
            curr_pos += 1
            ax3 = plt.subplot(gs[curr_pos])
            protein_x, protein_y = phase_plot_proteins
            x_traj = simulation_results[protein_x]
            y_traj = simulation_results[protein_y]
            
            fixed_levels = {p: simulation_results[p][0] for p in self.graph.nodes if p not in phase_plot_proteins}
            
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
            
            ax3.quiver(X, Y, U, V, color='black', alpha=0.3, width=0.002, scale=20)
            
            if display_until is not None:
                ax3.plot(x_traj, y_traj, 'b-', linewidth=1, alpha=0.2)
            
            ax3.plot(x_traj[:display_idx], y_traj[:display_idx], 'b-', linewidth=2)
            
            # Plot all time points in black except current time in red
            ax3.plot(x_traj[0], y_traj[0], 'ko', markersize=8, label='Initial Point')
            
            if time_points:
                for time_point in time_points:
                    if time_point <= (display_until if display_until is not None else t[-1]):
                        idx = np.abs(t - time_point).argmin()
                        ax3.plot(x_traj[idx], y_traj[idx], 'ko', markersize=8,
                                label=f't={time_point}')
            
            if display_until is not None:
                current_x = x_traj[display_idx-1]
                current_y = y_traj[display_idx-1]
                ax3.plot(current_x, current_y, 'ro', markersize=8, label=f'Current (t={display_until})')
            else:
                ax3.plot(x_traj[-1], y_traj[-1], 'ko', markersize=8, label='Final Point')
            
            ax3.set_xlabel(f'{protein_x} Concentration', fontsize=12)
            ax3.set_ylabel(f'{protein_y} Concentration', fontsize=12)
            ax3.set_title(f'Phase Portrait: {protein_y} vs {protein_x}', fontsize=14, pad=10)
            ax3.grid(True, alpha=0.3)
            ax3.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
            
            if use_tight_axes:
                ax3.set_xlim(x_min - x_padding, x_max + x_padding)
                ax3.set_ylim(y_min - y_padding, y_max + y_padding)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Leave room for legends
        plt.show()


    def get_latex_equations(self) -> str:
        """Generate LaTeX equations representing the system's differential equations"""
        equations = []
        
        for protein in self.graph.nodes:
            # Start equation for this protein
            equation = f"\\frac{{d{protein}}}{{dt}} &= "
            
            # Get predecessors (regulatory proteins)
            predecessors = list(self.graph.predecessors(protein))
            
            if not predecessors:
                # Only basal production and decay
                equation += f"\\beta_{{{protein}}} - \\alpha {protein}"
            else:
                # Add basal production rate
                equation += f"\\beta_{{{protein}}} "
                
                # Handle regulation terms based on aggregation type
                if self.aggregation_types[protein] == 'AND':
                    equation += "\\left("
                    regulation_terms = []
                    
                    for source in predecessors:
                        interaction = self.interactions[(source, protein)]
                        if interaction.type == 'activation':
                            regulation_terms.append(
                                f"\\frac{{{source}^{{n_{{{source}, {protein}}}}}}}"
                                f"{{K_{{{source}, {protein}}}^{{n_{{{source}, {protein}}}}} + {source}^{{n_{{{source}, {protein}}}}}}}"
                            )
                        else:  # repression
                            regulation_terms.append(
                                f"\\frac{{K_{{{source}, {protein}}}^{{n_{{{source}, {protein}}}}}}}"
                                f"{{K_{{{source}, {protein}}}^{{n_{{{source}, {protein}}}}} + {source}^{{n_{{{source}, {protein}}}}}}})"
                            )
                    
                    equation += " \\cdot ".join(regulation_terms)
                    equation += "\\right)"
                    
                else:  # OR aggregation
                    regulation_terms = []
                    
                    for source in predecessors:
                        interaction = self.interactions[(source, protein)]
                        if interaction.type == 'activation':
                            regulation_terms.append(
                                f"\\frac{{{source}^{{{interaction.n}}}}}"
                                f"{{{interaction.K}^{{{interaction.n}}} + {source}^{{{interaction.n}}}}}"
                            )
                        else:  # repression
                            regulation_terms.append(
                                f"\\left(1 - \\frac{{{source}^{{{interaction.n}}}}}"
                                f"{{{interaction.K}^{{{interaction.n}}} + {source}^{{{interaction.n}}}}}\\right)"
                            )
                    
                    equation += "\\max\\left(" + ", ".join(regulation_terms) + "\\right)"
                
                # Add decay term
                equation += f" - \\alpha {protein}"
            
            equations.append(equation)
        
        # Join all equations with line breaks and align properly
        latex = "\\begin{align*}\n"
        latex += "\\\\\n".join(equations)
        latex += "\n\\end{align*}"
        
        return latex