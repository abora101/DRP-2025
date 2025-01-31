import protein_network as pn




# Example usage:
network = pn.ProteinNetwork("Arabinose", 0.25)

# Add proteins
network.add_protein("X", initial_level=1.0, removal_rate=0.0, beta_naught=0.0, aggregation_type='AND')
network.add_protein("Y", initial_level=0, removal_rate=0.75, beta_naught=1.5, aggregation_type='AND')
network.add_protein("Z", initial_level=0, removal_rate=0.75, beta_naught=1.5, aggregation_type='AND')

# Add interactions
network.add_interaction(source="X", target="Y", n=10.0, K=1.0, interaction_type="activation")
network.add_interaction(source="Y", target="Z", n=10.0, K=0.5, interaction_type="activation")
network.add_interaction(source="X", target="Z", n=10.0, K=1.0, interaction_type="activation")

# Set external signals
network.set_external_signal("X", lambda t: t < 5.001) #cAMP always present
network.set_external_signal("Y", lambda t: True) #Arabinose always present
network.set_external_signal("Z", lambda t: False) #not signalling protein

# Simulate and visualize
print(network.get_latex_equations())
results = network.simulate(t_span=(0, 10), max_step = 0.01)
network.visualize_results(results, phase_plot_proteins=["Y", "Z"], time_points=[0.75, 5, 8], display_until=10.0, show_network=False)