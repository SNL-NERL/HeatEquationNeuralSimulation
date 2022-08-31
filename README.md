Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Introduction 
`<heat_eq_neural_sim.py>` produces a spiking neural network configuration that
solves a simple steady state heat equation utilizing. This uses a random walk
method on a wire of length L, where walkers are allowed to move left, right, or
stay in their same location in a single time step.

The resulting network can be used to benchmark neuromorphic hardware and was detailed in Smith, et al., "Solving a steady-state PDE using spiking networks
and neuromorphic hardware", ICONS 2020.

The problem of this wire of length L is as follows: we attach the left end to
a cooler external heat was and enforce that no heat can transfer across this
left boundary (u(0) = 0, u'(0) = 0). There is a heat source nearby the wire
with a given heat flux density profile.

# Usage 

### As a script
The code can be used as a standalone script as follows:

```
usage: heat_eq_neural_sim.py [-h] [-L LENGTH] [-N NODES] [-w WALKERS] [-d DT] [-i [INITIAL_POSITIONS ...]] [-g GRAPH_FILE] [-m MATRIX_FILE] [-p PICKLE_FILE]

options:
  -h, --help            show this help message and exit
  -L LENGTH, --Length LENGTH
                        Length of wire
  -N NODES, --Nodes NODES
                        Number of nodes
  -w WALKERS, --walkers WALKERS
                        Number of walkers to start at each initial location
  -d DT, --dt DT        dt used by simulation
  -i [INITIAL_POSITIONS ...], --initial_positions [INITIAL_POSITIONS ...]
                        List of initial walker positions
  -g GRAPH_FILE, --graph_file GRAPH_FILE
                        Filename if you want to save the graph as yaml
  -m MATRIX_FILE, --matrix_file MATRIX_FILE
                        Filename if you want to save the matrix
  -p PICKLE_FILE, --pickle_file PICKLE_FILE
                        Filename if you want to save the graph, injection and MarkovNet as pickle
```

The default values for L, N, dt are those used in the paper: wire length L = 2,
mesh N = 40, time step dt = 0.0001.

The matrix (Numpy ndarray) saved using MATRIX_FILE is the transition matrix if
you want to implement your own network.

The graph (NetworkX DiGraph) saved using GRAPH_FILE is a yaml version of the DiGraph.
See this [NetworkX documentation](https://networkx.github.io/documentation/stable/reference/readwrite/yaml.html).

### As a module
Alternatively, you can get the transition matrix and graph objects directly using `generate_prob_mtx` and `build_graph` directly.  Please see code and code comments for parameters and set up. 

# Dependencies 
The code is not fully tested on other versions, but should generally be compatible.

* python >= 3.6
* numpy == 1.14.5
* networkx == 2.4
* scipy == 1.4.1

# Graph and Network Details
### NetworkX Graph Requirements
The network defined is a NetworkX DiGraph object.  There are 
three main components.
1. The graph requires a property 'has_delay' with values either `True` 
(if variable delays are included) or `False` (if all the delays are 1).
2. Each node represents a neuron.  Neuron properties should be set using node 
properties: 'threshold', 'potential', 'decay', 'p', 'record' (See neuron model 
below for full details).  
3. Each edge represents a synapse. Synapse properties should be set using 
edge properties: 'weight', 'delay' (See synapses below).

### Neuron Model
The neuron model used is generalization of a stochastic leaky-integrate-and-fire neuron.  We include the following reference neuron definition, but many other compatible neuron models exist.  Additionally, some other neuron models (e.g. that of TrueNorth and Loihi) can easily be adapted by simple updates to the graph.

The state is computed as follows for neuron *i*, where *I* represents an input injection, *Weights* are the synaptic weights and *Spikes* are the incoming spikes from other neurons:

1. *x[i] = x[i]* + *I[i]* + *Weights[i]* * *Spikes*
2. If *x[i] > T[i]*, then *Spikes[x_i] = (a < p[i])*.
3. If *x[i] <= T[i]*, then *x[i]* = *x[i]* * *(1-m[i])*

|Variable Key | Definition | Variable Name | Type | 
| ----- | ----- | ----- | ---- |
| potential | Internal potential value of the neuron | x[i] | float |
| threshold | Neuron threshold | T[i] | float |
| decay | Decay constant | m[i] | float |
| p | Probability of fire | p[i] | float [0,1] |
| record | List of values to record | N/A | list of strings |
| N/A | Random Draw | a | float [0,1] |

### Recording from Neurons
As referenced above, the 'record' neuron property defines which quantities should be recorded during simulation at each timestep.  In this context, this value is just a note to the users and can be ignored if desired.

### Synapse Model
Synapses are point synapses, where the synaptic weight is integrated into the 
post-synaptic neuron in a single timestep. 

Delays are assumed to be integers of at least 1.

| Record Key | Definition | Type |
| ----- | ----- | ----- |
| 'weight' | Synaptic Weight | float | 
| 'delay' | Synaptic Delay | int [1, inf) |

### Injection 
We represent external input into the system by an 'injection' dictionary (keys: timestep, value: array of length number of neuron).  The idea is that these are integrated into the neurons directly at any timestep where it is defined.  For the random walk networks, we only use input at the initial timestep, so these can be replicated with an initial potential value or incoming spikes with appropriate weighting.  When exporting to yaml, the injection is stored as a graph property, i.e. graph.graph['injection'].  Alternatively, it is returned as the second entry from `build_graph`.

