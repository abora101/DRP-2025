import protein_network as pn

# Example usage:
network = pn.ProteinNetwork("Human Pain Sensation", 0)

# Add proteins
network.add_protein("X1", initial_level=0, removal_rate=0.5, beta_naught=1.5)
network.add_protein("X2", initial_level=0, removal_rate=0.5, beta_naught=1.5)
network.add_protein("Y", initial_level=0, removal_rate=0.66, beta_naught=2)
network.add_protein("Z", initial_level=0, removal_rate=5, beta_naught=5)

# Add interactions
network.add_interaction(source="X1", target="Y", n=8.0, K=0.3, interaction_type="activation")
network.add_interaction(source="X1", target="Z", n=8.0, K=0.3, interaction_type="activation")
network.add_interaction(source="X2", target="Y", n=8.0, K=2.0, interaction_type="inhibition")
network.add_interaction(source="X2", target="Z", n=8.0, K=0.3, interaction_type="activation")

network.add_interaction(source="Y", target="Z", n=8.0, K=1.2, interaction_type="inhibition")

# Set external signals
network.set_external_signal("X1", lambda t: True) #pain present
network.set_external_signal("X2", lambda t: True) #slightly deeper in the skin; takes longer to activate

network.set_external_signal("Y", lambda t: True) #signal always on
network.set_external_signal("Z", lambda t: False) #not signalling protein

# Simulate and visualize
results = network.simulate(t_span=(0, 8), max_step = 0.01)
network.visualize_results(results)
