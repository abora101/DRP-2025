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
    def __init__(self):
        """Initialize an empty protein regulatory network"""
        self.graph = nx.DiGraph()
        self.protein_levels = {}
        self.interactions = {}
        self.removal_rates = {}  # alpha values for each protein
        self.aggregation_types = {}  # 'AND' or 'OR' for each protein
        self.beta_naughts = {}  # Production rates for each protein
        
    def add_protein(self, name: str, initial_level: float = 1.0, 
                   removal_rate: float = 0.1, 
                   beta_naught: float = 1.0,
                   aggregation_type: str = 'OR'):
        """
        Add a protein to the network
        
        Parameters:
        -----------
        beta_naught: float
            Base production rate for this protein
        aggregation_type: str
            'AND' or 'OR' - how to combine multiple regulatory inputs
        """
        if aggregation_type not in ['AND', 'OR']:
            raise ValueError("aggregation_type must be either 'AND' or 'OR'")
            
        self.graph.add_node(name)
        self.protein_levels[name] = initial_level
        self.removal_rates[name] = removal_rate
        self.beta_naughts[name] = beta_naught
        self.aggregation_types[name] = aggregation_type
    
    def add_interaction(self, source: str, target: str, 
                       n: float,
                       K: float,
                       interaction_type: str = 'activation'):
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
    
    def calculate_hill_regulation(self, source_level: float, interaction: HillInteraction) -> float:
        """Calculate regulatory effect using Hill function"""
        hill_term = (source_level ** interaction.n) / \
                   (interaction.K ** interaction.n + source_level ** interaction.n)
        
        if interaction.type == 'activation':
            return hill_term
        else:  # repression
            return 1 - hill_term
    
    def get_production_rate(self, protein: str, current_levels: Dict[str, float]) -> float:
        """Calculate the total production rate (beta) for a protein"""
        predecessors = list(self.graph.predecessors(protein))
        
        if not predecessors:
            return self.beta_naughts[protein]  # baseline rate
        
        # Calculate individual regulatory effects
        regulations = []
        for source in predecessors:
            interaction = self.interactions[(source, protein)]
            source_level = current_levels[source]
            hill_term = self.calculate_hill_regulation(source_level, interaction)
            regulations.append(hill_term)
            
        # Combine effects based on aggregation type
        if self.aggregation_types[protein] == 'AND':
            # Product of all terms
            combined_effect = np.prod(regulations)
        else:  # 'OR'
            # In high n limit, this approaches logical OR
            combined_effect = max(regulations)
            
        return self.beta_naughts[protein] * combined_effect
    
    def system_equations(self, t: float, y: np.ndarray, protein_order: List[str]) -> np.ndarray:
        """Define the system of differential equations"""
        current_levels = {protein: level for protein, level in zip(protein_order, y)}
        derivatives = np.zeros_like(y)
        
        for i, protein in enumerate(protein_order):
            beta = self.get_production_rate(protein, current_levels)
            alpha = self.removal_rates[protein]
            derivatives[i] = beta - alpha * y[i]
            
        return derivatives
    
    def simulate(self, t_span: Tuple[float, float], 
                method: str = 'RK45',
                rtol: float = 1e-3,
                atol: float = 1e-6,
                max_step: float = np.inf) -> Dict[str, np.ndarray]:
        """Simulate the system using scipy.integrate.solve_ivp"""
        protein_order = list(self.graph.nodes)
        y0 = [self.protein_levels[protein] for protein in protein_order]
        
        solution = solve_ivp(
            fun=lambda t, y: self.system_equations(t, y, protein_order),
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

    def visualize_results_old(self, simulation_results: Dict[str, np.ndarray]):
        """Create network and time series visualizations"""
        # Set up the figure with two subplots
        fig = plt.figure(figsize=(15, 6))
        
        # Network subplot
        ax1 = plt.subplot(121)
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color='lightblue',
                             node_size=2000,
                             alpha=0.7,
                             ax=ax1)
        
        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12)
        
        # Draw edges
        for source, target in self.graph.edges():
            if self.interactions[(source, target)].type == 'activation':
                ax1.add_patch(
                    FancyArrowPatch(
                        pos[source],
                        pos[target],
                        arrowstyle='-|>',
                        color='green',
                        mutation_scale=20,
                        shrinkA=15,
                        shrinkB=15
                    )
                )
            else:
                ax1.add_patch(
                    FancyArrowPatch(
                        pos[source],
                        pos[target],
                        arrowstyle='-[',
                        color='red',
                        mutation_scale=20,
                        shrinkA=15,
                        shrinkB=15
                    )
                )
        
        # Create edge labels
        edge_labels = {}
        for (source, target), interaction in self.interactions.items():
            label = f'n={interaction.n:.1f}\nK={interaction.K:.1f}'
            edge_labels[(source, target)] = label
        
        # Draw edge labels
        nx.draw_networkx_edge_labels(self.graph, pos, 
                                   edge_labels=edge_labels,
                                   font_size=8)
        
        # Add node parameters
        for node, (x, y) in pos.items():
            plt.text(x, y + 0.1, 
                    f'({self.aggregation_types[node]})\nβ₀={self.beta_naughts[node]:.1f}',
                    horizontalalignment='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        ax1.set_title("Protein Regulatory Network")
        ax1.axis('off')
        
        # Time series subplot
        ax2 = plt.subplot(122)
        t = simulation_results['t']
        
        for protein in self.graph.nodes:
            ax2.plot(t, simulation_results[protein], 
                    label=f'{protein} ({self.aggregation_types[protein]})',
                    linewidth=2)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Concentration')
        ax2.set_title('Protein Concentrations Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

    def visualize_results_also_old(self, simulation_results: Dict[str, np.ndarray]):
        """Create network and time series visualizations"""
        # Set up the figure with two subplots
        fig = plt.figure(figsize=(18, 8))
    
        # Network subplot
        ax1 = plt.subplot(121)
    
        # Create layout with more space between nodes
        pos = nx.spring_layout(self.graph, k=1.5, iterations=50)
    
        # Draw nodes with a more appealing color and larger size
        nx.draw_networkx_nodes(self.graph, pos, 
                            node_color='#87CEFA',  # Light sky blue
                            node_size=3000,
                            alpha=0.8,
                            ax=ax1)
    
        # Draw node labels with slightly larger font
        nx.draw_networkx_labels(self.graph, pos, font_size=14, font_weight='bold')
    
        # Draw edges with increased curvature for better visibility
        for source, target in self.graph.edges():
            if self.interactions[(source, target)].type == 'activation':
                ax1.add_patch(
                    FancyArrowPatch(
                        pos[source],
                        pos[target],
                        arrowstyle='-|>',
                        color='#228B22',  # Forest green
                        mutation_scale=25,
                        shrinkA=20,
                        shrinkB=20,
                        connectionstyle="arc3,rad=0.2"
                    )
                )
            else:
                ax1.add_patch(
                    FancyArrowPatch(
                        pos[source],
                        pos[target],
                        arrowstyle='-[',
                        color='#B22222',  # Firebrick red
                        mutation_scale=25,
                        shrinkA=20,
                        shrinkB=20,
                        connectionstyle="arc3,rad=0.2"
                    )
                )
    
        # Create edge labels with slightly reduced font size
        edge_labels = {}
        for (source, target), interaction in self.interactions.items():
            label = f'n={interaction.n:.1f}\nK={interaction.K:.1f}'
            edge_labels[(source, target)] = label
    
        # Draw edge labels
        nx.draw_networkx_edge_labels(self.graph, pos, 
                               edge_labels=edge_labels,
                               font_size=7)
    
        # Add node parameters with improved formatting
        for node, (x, y) in pos.items():
            plt.text(x, y + 0.15, 
                    f'({self.aggregation_types[node]})\nβ₀={self.beta_naughts[node]:.1f}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=8,
                    bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.3'))
    
        ax1.set_title("Protein Regulatory Network", fontsize=16, fontweight='bold')
        ax1.axis('off')
    
        # Time series subplot (unchanged)
        ax2 = plt.subplot(122)
        t = simulation_results['t']
    
        for protein in self.graph.nodes:
            ax2.plot(t, simulation_results[protein], 
                label=f'{protein} ({self.aggregation_types[protein]})',
                linewidth=2)
    
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Concentration', fontsize=12)
        ax2.set_title('Protein Concentrations Over Time', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
        plt.tight_layout()
        plt.show()

    def visualize_results(self, simulation_results: Dict[str, np.ndarray]):
        """Create network and time series visualizations"""
        # Set up the figure with two subplots
        fig = plt.figure(figsize=(18, 8))
    
        # Network subplot
        ax1 = plt.subplot(121)
    
        # Create layout with more space between nodes and from edges
        pos = nx.spring_layout(self.graph, k=1.5, iterations=50)
    
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, 
                         node_color='#87CEFA',
                         node_size=3000,
                         alpha=0.8,
                         ax=ax1)
    
        # Draw node labels
        nx.draw_networkx_labels(self.graph, pos, font_size=14, font_weight='bold')
    
        # Draw edges with increased curvature
        for i, (source, target) in enumerate(self.graph.edges()):
            interaction = self.interactions[(source, target)]
            edge_color = '#228B22' if interaction.type == 'activation' else '#B22222'
        
            # Calculate midpoint for label placement
            mid_x = (pos[source][0] + pos[target][0]) / 2
            mid_y = (pos[source][1] + pos[target][1]) / 2
        
            # Add edge
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
        
            # Add edge label with background
            label = f'n={interaction.n:.1f}, K={interaction.K:.1f}'
            bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
            ax1.text(mid_x, mid_y, label, ha='center', va='center', fontsize=6, bbox=bbox_props)
    
        # Add node parameters
        for node, (x, y) in pos.items():
            plt.text(x, y + 0.15, 
                f'({self.aggregation_types[node]})\nβ₀={self.beta_naughts[node]:.1f}',
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle='round,pad=0.3'))
    
        ax1.set_title("Protein Regulatory Network", fontsize=16, fontweight='bold', pad=10)
        ax1.axis('off')
    
        # Adjust subplot to give more space around the edges
        ax1.set_xlim(min(pos[node][0] for node in pos) - 0.3, max(pos[node][0] for node in pos) + 0.3)
        ax1.set_ylim(min(pos[node][1] for node in pos) - 0.3, max(pos[node][1] for node in pos) + 0.3)
    
        # Time series subplot (unchanged)
        ax2 = plt.subplot(122)
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
    
        plt.tight_layout()
        plt.show()



# Create network
network = ProteinNetwork()

# Add proteins with their production rates
network.add_protein("A", initial_level=1.0, removal_rate=0.1, beta_naught=0.1, aggregation_type='AND')
network.add_protein("B", initial_level=0, removal_rate=0.15, beta_naught=1.5, aggregation_type='AND')
network.add_protein("C", initial_level=0, removal_rate=0.12, beta_naught=1.8, aggregation_type='AND')
network.add_protein("D", initial_level=0, removal_rate=0.12, beta_naught=1.8, aggregation_type='AND')

# Add interactions (no beta_naught needed here anymore)
network.add_interaction(
    source="A", 
    target="B",
    n=10.0,
    K=1.0,
    interaction_type="activation"
)
network.add_interaction(
    source="B", 
    target="C",
    n=10.0,
    K=0.5,
    interaction_type="activation"
)
network.add_interaction(
    source="A", 
    target="C",
    n=10.0,
    K=1.0,
    interaction_type="activation"
)
network.add_interaction(
    source="C", 
    target="D",
    n=10.0,
    K=1.0,
    interaction_type="activation"
)

# Simulate and visualize
results = network.simulate(t_span=(0, 10))
network.visualize_results(results)