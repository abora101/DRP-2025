import protein_network as pn

# Example usage:
network = pn.ProteinNetwork("Phage Lambda Cro Repressor")

# Add proteins
network.add_protein("C1", initial_level=3, removal_rate=0.5, beta_naught=1.5, aggregation_type='AND')
network.add_protein("cro", initial_level=0, removal_rate=0.5, beta_naught=1.5, aggregation_type='AND')
network.add_protein("Z", initial_level=1, removal_rate=1, beta_naught=1)

# Add interactions
network.add_interaction(source="C1", target="cro", n=1.0, K=1.0, interaction_type="inhibition")
network.add_interaction(source="cro", target="C1", n=1.0, K=1.0, interaction_type="inhibition")
network.add_interaction(source="Z", target="C1", n=1.0, K=0.5, interaction_type="inhibition")
network.add_interaction(source="Z", target="cro", n=1.0, K=0.5, interaction_type="activation")

network.add_interaction(source="cro", target="cro", n=0.5, K=2.5, interaction_type="activation")
network.add_interaction(source="C1", target="C1", n=0.5, K=2.5, interaction_type="activation")

# Set external signals
network.set_external_signal("Z", lambda t: t > 3 and t < 15) #DNA damaged
network.set_external_signal("C1", lambda t: True) #inhibited by DNA damage
network.set_external_signal("cro", lambda t: True)

# Simulate with phase plot
results = network.simulate(t_span=(0, 35), max_step=0.01)
network.visualize_results(results)