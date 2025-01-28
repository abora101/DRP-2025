import protein_network as pn




# Example usage:
network = pn.ProteinNetwork("p53-mdm2 (noised)", 0.225)

# Add proteins
network.add_protein("p53", initial_level=1.05, removal_rate=0.5, beta_naught=1.0)
network.add_protein("mdm2", initial_level=1.0, removal_rate=0.5, beta_naught=1.0)

# Add interactions
network.add_interaction(source="p53", target="mdm2", n=16.0, K=1.0, interaction_type="activation")
network.add_interaction(source="mdm2", target="p53", n=16.0, K=1.0, interaction_type="inhibition")

# Set external signals
network.set_external_signal("p53", lambda t: True) #always active
network.set_external_signal("mdm2", lambda t: True) #always active

# Simulate and visualize
results = network.simulate(t_span=(0, 40), max_step = 0.01)
network.visualize_results(results, phase_plot_proteins=["p53", "mdm2"])






# Example usage:
network = pn.ProteinNetwork("p53-mdm2 (not noised)")

# Add proteins
network.add_protein("p53", initial_level=1.05, removal_rate=0.5, beta_naught=1.0)
network.add_protein("mdm2", initial_level=1.0, removal_rate=0.5, beta_naught=1.0)

# Add interactions
network.add_interaction(source="p53", target="mdm2", n=16.0, K=1.0, interaction_type="activation")
network.add_interaction(source="mdm2", target="p53", n=16.0, K=1.0, interaction_type="inhibition")

# Set external signals
network.set_external_signal("p53", lambda t: True) #always active
network.set_external_signal("mdm2", lambda t: True) #always active

# Simulate and visualize
results = network.simulate(t_span=(0, 40), max_step = 0.01)
network.visualize_results(results, phase_plot_proteins=["p53", "mdm2"])





# Example usage:
network = pn.ProteinNetwork("p53-mdm2 (noise only)", 0.225)

# Add proteins
network.add_protein("p53", initial_level=1.05, removal_rate=0.5, beta_naught=1.0)
network.add_protein("mdm2", initial_level=1.0, removal_rate=0.5, beta_naught=1.0)

# Set external signals
network.set_external_signal("p53", lambda t: True) #always active
network.set_external_signal("mdm2", lambda t: True) #always active

# Simulate and visualize
results = network.simulate(t_span=(0, 40), max_step = 0.01)
network.visualize_results(results, phase_plot_proteins=["p53", "mdm2"])
