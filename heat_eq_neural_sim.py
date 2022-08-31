""" 
Heat Equation Neural Simulation
Produces a spiking neural network configuration that
solves a simple steady state heat equation utilizing. This uses a random walk
method on a wire of length L, where walkers are allowed to move left, right, or
stay in their same location in a single time step.

The resulting network can be used to benchmark neuromorphic hardware and was 
detailed in Smith, et al., "Solving a steady-state PDE using spiking networks
and neuromorphic hardware", ICONS 2020.

The problem of this wire of length L is as follows: we attach the left end to
a cooler external heat was and enforce that no heat can transfer across this
left boundary (u(0) = 0, u'(0) = 0). There is a heat source nearby the wire
with a given heat flux density profile.

Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC (NTESS). 
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains 
certain rights in this software.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this 
list of conditions and the following disclaimer in the documentation and/or other 
materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may 
be used to endorse or promote products derived from this software without specific 
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
OF SUCH DAMAGE.
"""
# needed imports 
import numpy as np
import networkx as nx
from collections import deque
from scipy.stats import norm
import argparse
###############################################
# 
# Spiking Network Objects 
#
###############################################

# Global Variables
num_neurons = 0
neuron_list = []

# Resets an internal list of neuron_list
# This is required if you want start a new network once a network has been created
def reset_neuron_list():
    global num_neurons
    num_neurons = 0
    global neuron_list
    neuron_list = []
    
# A class for Neuron Objects
# Parameters are set using keywords
# Paramters are name, group, threshold, decay, p, reset_mode
# At initialization group should be a single hashable Objects
# However, the group is assigned as the first element in a list groups
# This allows a neuron to belong to more than one group
# p is the probability of spiking when the threshold is reached
# reset_mode has two options:  'spike' = reset after a spike
# 'threshold' - reset after a threshold is reached
class Neuron:
    def __init__(self,**kwargs):
        global num_neurons
        self.name = num_neurons
        if 'name' in kwargs:
            self.name = kwargs['name']
        self.groups = []
        if 'group' in kwargs:
            self.groups = [kwargs['group']]
        self.threshold = 1
        if 'threshold' in kwargs:
            self.threshold = kwargs['threshold']
        self.decay = 1
        if 'decay' in kwargs:
            self.decay = kwargs['decay']
        self.p = 1
        if 'p' in kwargs:
            self.p = kwargs['p']
        self.reset_mode = 'spike'
        ##reset_mode options are
        ## 'spike' - reset after a spike
        ## 'threshold' - reset after a threshold is reached
        if 'reset_mode' in kwargs:
            self.reset_mode = kwargs['reset_mode']
        self.potential = 0
        num_neurons = num_neurons+1
        global neuron_list
        neuron_list.append(self)

#####################
#
# Markov Objects    
#
#####################


# Represents the transition matrix from one unit to its neighbors.
# 'location' is a user-defined key for the origin unit's location
# A standard choice is a tuple of coordinates e.g. (1,2)
# 'neighbors' is a list of tuples [(destination.location, probability),... ]
class Transition:
    def __init__(self, **kwargs):
        if 'location' in kwargs:
            self.location = kwargs['location']
        self.neighbors = []
        if 'neighbors' in kwargs:
            self.neighbors = kwargs['neighbors']


# A spiking network that simulates random markov walks
# 'transitions' is a dictionary {location : trasition }
# 'initial_walkers' is a dictionary {location : number of walkers}
# 'log_potential' is boolean, if True, the potentials can be logged 
# self.potential_log is otherwise None
# 'log_spikes' is boolean, if True, the potentials can be logged 
# self.spike_log is otherwise None
# 'syncrhonized' is a boolean.  If true, walkers and stored in a buffer before being sent to
# a new unit.  This ensures that each walker takes the same number of steps.
class MarkovNetwork:
    def __init__(self,**kwargs):
        self.transitions = {}
        if 'transitions' in kwargs:
            self.transitions = kwargs['transitions']
        self.initial_walkers = {}
        if 'initial_walkers' in kwargs:
            self.initial_walkers = kwargs['initial_walkers']
        self.log_potential = False
        self.potential_log = None
        if 'log_potential' in kwargs:
            self.log_potential = kwargs['log_potential']
        self.log_spikes = False
        self.spike_log = None
        if 'log_spikes' in kwargs:
            self.log_spikes = kwargs['log_spikes']
        self.syncrhonized = True
        if 'synchronized' in kwargs:
            self.synchronized = kwargs['synchronized']
        self.built = False

    def build(self):
        self.num_units = len(self.transitions)
        self.graph = nx.DiGraph()
        reset_neuron_list()
        self.all_units = []
        for location in self.transitions:
            neighbors = self.transitions[location].neighbors
            ##EDIT  len(neighbors)-1 -> max(len(neighbors)-1, 1)
            self.all_units.append(Unit(location, max(len(neighbors)-1,1), synchronized=self.synchronized))
        for unit in self.all_units:
            for (neighbor, p) in self.transitions[unit.coordinates].neighbors:
                matching_units = [unit for unit in self.all_units if unit.coordinates==neighbor]
                if len(matching_units)<1:
                    print("Something went wrong in finding neighbors.... At least one neighbor is missing")
                    print("Availible Coordinates")
                    print([unit.coordinates for unit in self.all_units])
                    print("Needed Coordinates")
                    print(neighbor)
                if len(matching_units)>1:
                    print("Something went wrong in finding neighbors.... Too many neighbors found")
                    print("Availible Coordinates")
                    print([unit.coordinates for unit in self.all_units])
                    print("Needed Coordinates")
                    print(neighbor)
                unit.connect(matching_units[0], p)
        #Control neruons
        self.walker_supervisor = Neuron(threshold=1.0, decay=1.0, group='controller')
        self.simulation_supervisor = Neuron(threshold=100.0, decay=1.0, group='controller')
        self.walks_complete = Neuron(threshold=self.num_units-0.5,decay=0.0, group='controller')
        self.buffer_supervisor = None
        self.buffer_clear = None
        if self.syncrhonized:
            self.buffer_supervisor = Neuron(threshold=1.0, decay=1.0, group='controller')
            self.buffer_supervisor.groups.append('buffer-control')
            self.buffer_clear = Neuron(threshold=self.num_units-0.5, decay=0.0, group = 'controller')
            self.buffer_clear.groups.append('buffer-control')

        self.injection = {}
        self.injection[0] = [(self.walker_supervisor,10)]

        for unit in self.all_units:
            if unit.coordinates in self.initial_walkers:
                self.injection[0].append((unit.walker_counter_neuron,-self.initial_walkers[unit.coordinates]))

        #Add units to graph
        for unit in self.all_units:
            unit.add_to_graph(self.graph, walker_supervisor=self.walker_supervisor, walks_complete=self.walks_complete, simulation_supervisor=self.simulation_supervisor, buffer_supervisor=self.buffer_supervisor, buffer_clear=self.buffer_clear)

        if self.syncrhonized:
            self.graph.add_edge(self.buffer_clear.name, self.walker_supervisor.name, weight=2.0, delay=2)
            self.graph.add_edge(self.walks_complete.name, self.buffer_supervisor.name, weight=2.0, delay=2)
        else:
            self.graph.add_edge(self.walks_complete.name, self.walker_supervisor.name, weight=2.0, delay=2)


        if self.log_potential:
            self.potential_log = np.zeros((0, len(neuron_list)))
            self.potential_log[:] = np.NaN
        if self.log_spikes:
            self.spike_log = deque()

        self.built = True
        return self.graph

# A Unit represents a discrete area in space through which walkers pass.
# Walkers may occupy only one unit
# Unless in transit, walkers must occupy exactly one unit
# Units are initialized with coordinates (a tuple of user-defined values), and probability_bits (which is the number of probability gates)
# Methods:
# connect - Connects one unit to the passed unit with specified probability
# add_to_graph - Adds neurons and edges to the specified graph.  Units should be connected before being added to a add_to_graph
class Unit:
    def __init__(self, coordinates, probability_bits, readout=None, synchronized=True):
        self.coordinates = coordinates
        self.synchronized = synchronized
        if readout is None:
            self.readout_neuron = Neuron(group='readout', decay=0, threshold=0.5)
        else:
            self.readout_neuron = readout
        self.walker_counter_neuron = Neuron(group='counter', decay=0, threshold=0.0)
        self.walker_generator_neuron = Neuron(group='generator', threshold=0.5)
        self.random_gates = []
        for i in range(0, probability_bits):
            self.random_gates.append(Neuron(group='random_gate', threshold=0.5, reset_mode='threshold'))
        self.probabilities = np.zeros((probability_bits+1,))
        self.neighbors = []
        self.output_gate_neurons = []
        for i in range(0, probability_bits+1):
            self.output_gate_neurons.append(Neuron(group='output_gate', reset_mode='threshold', threshold=0.5))
        self.neurons = [self.readout_neuron, self.walker_counter_neuron, self.walker_generator_neuron]
        self.neurons.extend(self.random_gates)
        self.neurons.extend(self.output_gate_neurons)
        if self.synchronized:
            self.buffer = Neuron(group='buffer', decay = 0, threshold=0.0)
            self.buffer_control = Neuron(group='generator', threshold=0.5)
            self.neurons.extend([self.buffer, self.buffer_control])
        for neuron in self.neurons:
            neuron.groups.append(str(coordinates))


    def connect(self, target_unit, probability):
        if(len(self.neighbors) < len(self.probabilities)):
            self.probabilities[len(self.neighbors)] = probability
            self.neighbors.append(target_unit)
        else:
            print("Out of space for neighbors!  Allocate more using probability_bits!")

    def add_to_graph(self, graph, walker_supervisor=None, walks_complete=None, simulation_supervisor=None, buffer_supervisor=None, buffer_clear=None):
        for neuron in self.neurons:
            graph.add_node(neuron.name,label=neuron.name)
        if self.synchronized:
            graph.add_edge(self.buffer_control.name, self.walker_counter_neuron.name, weight = -1.0, delay=1)
            graph.add_edge(self.buffer_control.name, self.buffer_control.name, weight =1.0, delay=1)
            graph.add_edge(self.buffer_control.name, self.buffer.name, weight=1.0, delay=1)
            graph.add_edge(self.buffer.name, self.walker_counter_neuron.name, weight = 1.0, delay=1)
            graph.add_edge(self.buffer.name, self.buffer_control.name, weight=-1.0, delay=1)
            graph.add_edge(self.buffer.name, self.buffer.name, weight=-1.0, delay=1)
            graph.add_edge(self.buffer_control.name, self.walker_counter_neuron.name, weight=-1.0, delay=1)
            if buffer_supervisor is not None:
                graph.add_edge(buffer_supervisor.name, self.buffer.name, weight=1.0, delay=1)
                graph.add_edge(buffer_supervisor.name, self.buffer_control.name, weight=1.0, delay = 1)
            if buffer_clear is not None:
                graph.add_edge(self.buffer.name, buffer_clear.name, weight=1.0, delay=1)
        graph.add_edge(self.walker_generator_neuron.name,  self.readout_neuron.name, weight=1.0, delay=1)
        graph.add_edge(self.walker_generator_neuron.name, self.walker_generator_neuron.name, weight=1.0, delay = 1)
        graph.add_edge(self.walker_generator_neuron.name, self.walker_counter_neuron.name, weight=1.0, delay=1)
        graph.add_edge(self.walker_counter_neuron.name, self.readout_neuron.name, weight=-1.0, delay=1)
        graph.add_edge(self.walker_counter_neuron.name, self.walker_counter_neuron.name, weight=-1.0, delay=1)
        graph.add_edge(self.walker_counter_neuron.name, self.walker_generator_neuron.name, weight=-1.0, delay=1)
        self.moderator_values = np.zeros(len(self.output_gate_neurons))

        output_length = len(self.output_gate_neurons)
        self.random_gates[0].p = self.probabilities[0]
        for i in range(0,output_length-1):
            self.random_gates[i].p = self.probabilities[i]/(1 - sum(self.probabilities[0:i]))
        
        for i in range(0, output_length-1):
            graph.add_edge(self.walker_generator_neuron.name, self.random_gates[i].name, weight=1.0, delay=i+1)
            graph.add_edge(self.walker_counter_neuron.name, self.random_gates[i].name, weight=-1.0, delay=i+1)
            graph.add_edge(self.random_gates[i].name, self.output_gate_neurons[i].name, weight=1.0, delay=1)
            k = 1
            for j in range(i+1, output_length-1):
                graph.add_edge(self.random_gates[i].name, self.random_gates[j].name, weight=-1.0, delay=k)
                k += 1
            graph.add_edge(self.random_gates[i].name, self.output_gate_neurons[-1].name, weight=-1.0, delay=k)
        graph.add_edge(self.walker_generator_neuron.name, self.output_gate_neurons[-1].name, weight=1.0, delay=output_length)
        graph.add_edge(self.walker_counter_neuron.name, self.output_gate_neurons[-1].name, weight=-1.0, delay=output_length)
        
        for i in range(0, len(self.neighbors)):
            neighbor = self.neighbors[i]
            gate = self.output_gate_neurons[i]
            input_neuron = neighbor.walker_counter_neuron
            if neighbor.synchronized:
                input_neuron = neighbor.buffer
            
            graph.add_edge(gate.name, input_neuron.name, weight=-1.0, delay=1)
        if walker_supervisor is not None:
            graph.add_edge(walker_supervisor.name, self.walker_counter_neuron.name, weight=1.0, delay=1)
            graph.add_edge(walker_supervisor.name, self.walker_generator_neuron.name, weight=1.0, delay=1)
        if simulation_supervisor is not None:
            graph.add_edge(simulation_supervisor.name, self.readout_neuron.name, weight=1.0, delay=1)
        if walks_complete is not None:
            graph.add_edge(self.walker_counter_neuron.name, walks_complete.name, weight=1.0, delay=len(self.probabilities))

# Converts compatibility 
def flatten_neuron_properties(graph, neuron_list):
    graph_to_return = graph.copy()
    for neuron in neuron_list:
        node = graph_to_return.nodes[neuron.name]
        for var in vars(neuron):
            node[var] = vars(neuron)[var]
    graph_to_return.graph['has_delay'] = True
    return graph_to_return

# Create injection from initial walker
def get_injection_from_initial_walkers(initial_dictionary,graph, num_neurons):
    target_tensor = np.zeros((num_neurons,))
    for target in initial_dictionary:
        value = initial_dictionary[target]
        for i, node in enumerate(graph.nodes):
            if 'groups' in graph.nodes[node]:
                groups = graph.nodes[node]['groups']
            else:
                groups = []
            if str(target) in groups and 'counter' in groups:
                target_idx = graph.nodes[node]['name']
                target_tensor[target_idx] = -value
    return target_tensor


##############################################################################
# 
# Build wire and neural circuit with transition probabilities
# 
##############################################################################

# determine probabilities 
def get_probabilities(cp, p1, p2, dt):
    bigp = norm.cdf(p2, cp, np.sqrt(2*dt))
    lilp = norm.cdf(p1, cp, np.sqrt(2*dt))
    p = bigp - lilp
    return p

# generate probability matrix
def generate_prob_mtx(L=2.0, N=40,dt=0.0001,verbose=1):
    dx = L/N
    space = [dx/2 + i*dx for i in range(N)]
    prob_mtx = np.zeros((N+1,N+1))
    for i in range(1,N):
        prob_mtx[i,i] = get_probabilities(space[i], space[i] - (dx/2), space[i] + (dx/2), dt)
        prob_mtx[i,i-1] = (1 - prob_mtx[i,i])/2
        prob_mtx[i,i+1] = prob_mtx[i,i-1]
    prob_mtx[0,0] = get_probabilities(dx/2, 0, dx, dt)
    prob_mtx[0,1] = 1 - prob_mtx[0,0]
    prob_mtx[N, N] = 1
    if all([sum(prob_mtx[i,:]) for row in range(N+1)]) == 1:
        if verbose>0:
            print('Transition matrix completed')
    else:
        raise ValueError("Something went wrong when calculating transition matrix.")
    return prob_mtx

# build neuron graph using spiking objects
def build_graph(prob_mtx, num_walkers = 10, initial_positions = [0], recorded_neuron_groups=['readout'], record_all=False, remove_sink_connections=True, verbose=1):
    reset_neuron_list()
    if record_all and recorded_neuron_groups is not None and len(recorded_neuron_groups)>0:
        raise ValueError('Only one of record_all and recorded_neuron_groups should be set. Set either recorded_neuron_groups to None or [] or set record_all to False.')
    N = np.shape(prob_mtx)
    transitions = {}
    initial = {}
    for i in range(N[0]):
        if i in initial_positions:
            initial[(i,)] = num_walkers
        neighbors = []
        p = 0
        for j in range(N[1]):
            prob_i_j = prob_mtx[i,j]
            p += prob_i_j
            if prob_i_j > 0:
                neighbors.append(((j,), prob_mtx[i,j]))
        if verbose>0:
            print("node " + str(i) + " has total probability of " + str(p) )
        if remove_sink_connections and len(neighbors)==1 and neighbors[0]==((i,),1.0):
            if verbose>0:
                print("Removing connections from a sink at " + str((i,)))
            neighbors=[]
        transitions[(i,)] = Transition(location=(i,), neighbors=neighbors)
    net = MarkovNetwork(initial_walkers=initial,
                        transitions=transitions,
                        synchronized=True,
                        log_potential=False,
                        log_spikes=True)
    graph = net.build()
    graph = flatten_neuron_properties(graph, neuron_list)
    for node in graph.nodes():
        for group in graph.nodes[node]['groups']:
            if group in recorded_neuron_groups or record_all:
                graph.nodes[node]['record'] = ['spikes']
    injection = {0:get_injection_from_initial_walkers(initial, graph, graph.number_of_nodes())}
    injection[0][net.walker_supervisor.name] = 10  #This starts a simulation
    graph.graph['injection'] = injection #Added for convenience of stored graph objects
    return graph, injection, net

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-L','--Length',default=2.0,type=float, help="Length of wire")
    parser.add_argument('-N',"--Nodes", default=40, type=int, help="Number of nodes")
    parser.add_argument('-w','--walkers', default=100, type=int, help="Number of walkers to start at each initial location")
    parser.add_argument('-d', '--dt', default=0.0001, type=float, help="dt used by simulation")
    parser.add_argument('-i','--initial_positions',default=[20], nargs='*', type=int, help="List of initial walker positions")
    parser.add_argument('-g', '--graph_file', default=None, help="Filename if you want to save the graph as yaml")
    parser.add_argument('-m', '--matrix_file',default=None, help="Filename if you want to save the matrix")
    parser.add_argument('-p','--pickle_file', default=None, help="Filename if you want to save the graph, injection and MarkovNet as pickle")
    # wire length L = 2, mesh N = 40, time step dt = 0.0001
    args = parser.parse_args()
    print("Building graph for 1-D heat equation along wire")
    prob_mtx = generate_prob_mtx(L=args.Length, N=args.Nodes, dt=args.dt, verbose=1)
    graph, injection, net = build_graph(prob_mtx, 
                                        num_walkers=args.walkers, 
                                        initial_positions=args.initial_positions, 
                                        recorded_neuron_groups=['readout'], 
                                        record_all=False, 
                                        remove_sink_connections=True, 
                                        verbose=1)
    if args.graph_file:
        import yaml
        with open(args.graph_file, 'w') as f:
            yaml.dump(graph, f)
    if args.matrix_file:
        np.save(args.matrix_file, prob_mtx)
    if args.pickle_file:
        import pickle
        with open(args.pickle_file, 'wb') as f:
            pickle.dump({'graph':graph,
                        'injection':injection,
                        'net': net}, f)
        