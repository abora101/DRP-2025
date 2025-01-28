import protein_network as pn




# Example usage:
network = pn.ProteinNetwork("Arabinose")

# Add proteins
network.add_protein("CRP", initial_level=1.0, removal_rate=0.0, beta_naught=0.0, aggregation_type='AND')
network.add_protein("AraC", initial_level=0, removal_rate=0.15, beta_naught=1.5, aggregation_type='AND')
network.add_protein("araBAD", initial_level=0, removal_rate=0.15, beta_naught=1.5, aggregation_type='AND')

# Add interactions
network.add_interaction(source="CRP", target="AraC", n=10.0, K=1.0, interaction_type="activation")
network.add_interaction(source="AraC", target="araBAD", n=10.0, K=0.5, interaction_type="activation")
network.add_interaction(source="CRP", target="araBAD", n=10.0, K=1.0, interaction_type="activation")

# Set external signals
network.set_external_signal("CRP", lambda t: True) #cAMP always present
network.set_external_signal("AraC", lambda t: True) #Arabinose always present
network.set_external_signal("araBAD", lambda t: False) #not signalling protein

# Simulate and visualize
results = network.simulate(t_span=(0, 4), max_step = 0.01)
network.visualize_results(results, phase_plot_proteins=["AraC", "araBAD"])






# Example usage:
network = pn.ProteinNetwork("Arabinose", 2.5)

# Add proteins
network.add_protein("CRP", initial_level=1.0, removal_rate=0.0, beta_naught=0.0, aggregation_type='AND')
network.add_protein("AraC", initial_level=0, removal_rate=0.15, beta_naught=1.5, aggregation_type='AND')
network.add_protein("araBAD", initial_level=0, removal_rate=0.15, beta_naught=1.5, aggregation_type='AND')

# Add interactions
network.add_interaction(source="CRP", target="AraC", n=10.0, K=1.0, interaction_type="activation")
network.add_interaction(source="AraC", target="araBAD", n=10.0, K=0.5, interaction_type="activation")
network.add_interaction(source="CRP", target="araBAD", n=10.0, K=1.0, interaction_type="activation")

# Set external signals
network.set_external_signal("CRP", lambda t: True) #cAMP always present
network.set_external_signal("AraC", lambda t: True) #Arabinose always present
network.set_external_signal("araBAD", lambda t: False) #not signalling protein

# Simulate and visualize
results = network.simulate(t_span=(0, 4), max_step = 0.01)
network.visualize_results(results, phase_plot_proteins=["AraC", "araBAD"])
