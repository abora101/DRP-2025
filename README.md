Read the book "An Introduction to Systems Biology", second edition, by Uri Alon.

Gave a talk at the DRP Symposium; slides are in this repo. Slides made extensive use of diagrams generated using this code.

Usage:
-

Import "protein_network" from the sample_networks folder in this repo

Network initialization:
network = ProteinNetwork("Name", noise_levels=0)

Add protein:
network.add_protein("Name", initial_level=1.0, removal_rate=1.0, beta_naught=1.0, aggregation_type="AND")
- beta naught is base production level (in absence of regulation)
- aggregation type is "AND" or "OR"

Add interaction:
network.add_interaction(source="Protein 1", target="Protein 2", n=10.0, K=1.0, interaction_type="activation")
- interaction type is "activation" or "inhibition"
- n and K are as in the hill equation

Set external signal:
network.set_external_signal("Protein", lambda t: True) 
- lambda function (second parameter) returns true when the protein is in activated form

Visualize:
results = network.simulate(t_span=(0, 4), max_step = 0.01)
- uses scipy's solve_ivp to simulate the coupled SDE.
- results, t_span, and max_step are akin to outputs/inputs from solve_ivp

network.visualize_results(results, phase_plot_proteins=["Protein 1", "Protein 2"], time_points=[], display_until: float, show_network = True)
- phase_plot_proteins must be exactly two proteins long if used. if set to None, no phase plot is displayed.
- A point on the phase plot is plotted and a black dotted line is drawn for every timestamp in time_points < display_until
- The time series and phase plot are greyed out after display_until, for use in presentations
- show_network controls whether or not you want to display the actual network itself
- displays the protein network as a directed graph and the time series of the proteins


Dependencies:
- 
networkx 3.4.2

numpy 2.2.2

scipy.integrate 1.15.1

matplotlib 3.10.0

(Older versions probably still work)
