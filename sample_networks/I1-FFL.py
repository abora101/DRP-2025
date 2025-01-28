import protein_network as pn

# Example usage:
network = pn.ProteinNetwork("I1-FFL")

# Add proteins
network.add_protein("A", initial_level=1.0, removal_rate=0.0, beta_naught=0.0, aggregation_type='AND')
network.add_protein("B", initial_level=0, removal_rate=0.5, beta_naught=1.5, aggregation_type='AND')
network.add_protein("C", initial_level=0, removal_rate=0.5, beta_naught=1.5, aggregation_type='AND')

# Add interactions
network.add_interaction(source="A", target="B", n=10.0, K=1.0, interaction_type="activation")
network.add_interaction(source="B", target="C", n=10.0, K=0.5, interaction_type="inhibition")
network.add_interaction(source="A", target="C", n=10.0, K=1.0, interaction_type="activation")

# Set external signals
network.set_external_signal("A", lambda t: True) 
network.set_external_signal("B", lambda t: True) 
network.set_external_signal("C", lambda t: False) 

# Simulate and visualize
results = network.simulate(t_span=(0, 20), max_step = 0.01)
network.visualize_results(results, phase_plot_proteins=["B", "C"])





# Example usage:
network = pn.ProteinNetwork("I1-FFL", 1.25)

# Add proteins
network.add_protein("A", initial_level=1.0, removal_rate=0.0, beta_naught=0.0, aggregation_type='AND')
network.add_protein("B", initial_level=0, removal_rate=0.5, beta_naught=1.5, aggregation_type='AND')
network.add_protein("C", initial_level=0, removal_rate=0.5, beta_naught=1.5, aggregation_type='AND')

# Add interactions
network.add_interaction(source="A", target="B", n=10.0, K=1.0, interaction_type="activation")
network.add_interaction(source="B", target="C", n=10.0, K=0.5, interaction_type="inhibition")
network.add_interaction(source="A", target="C", n=10.0, K=1.0, interaction_type="activation")

# Set external signals
network.set_external_signal("A", lambda t: True) 
network.set_external_signal("B", lambda t: True) 
network.set_external_signal("C", lambda t: False) 

# Simulate and visualize
results = network.simulate(t_span=(0, 20), max_step = 0.01)
network.visualize_results(results)
