from protein_sequence import ProteinSequence
from mutation_strategy import MutationStrategy, MinLogitPosSub
from evaluation_strategy import EvaluationStrategy
from model_singleton import ModelSingleton
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class Evolution:
    def __init__(self,root_sequence,mutation_strategy,evaluation_strategy,max_generations): #evaluation_strategy
        self.root_sequence = root_sequence
        self.mutation_strategy = mutation_strategy
        self.evaluation_strategy = evaluation_strategy
        self.max_generations = max_generations
        self.G = nx.DiGraph()
        self.G.add_node(root_sequence.id,object=root_sequence)
        # esm data
        self.model = ModelSingleton().get_model()
        self.alphabet = ModelSingleton().get_alphabet()
        self.batch_converter = ModelSingleton().get_batch_converter()

    def evolve_sequence(self,current_seq=None,generation=0):
        if current_seq is None:
            current_seq = self.root_sequence
            current_seq.mutation_score = 1 # root seq is unmutated therefore has max worse score

        if generation<self.max_generations: 
            # process potential mutations
            potential_mutations = self.mutation_strategy.get_next_mutations(current_seq) 
            if len(potential_mutations)==0: 
                print("No potential mutations found.")
                return
            print(potential_mutations)

            positions = potential_mutations[:,0].astype(int)
            aa_chars = potential_mutations[:,1]
            mutated_seqs = np.vectorize(current_seq.generate_mutated_sequence)(positions,aa_chars) 
            original_aa_chars = np.array(list(current_seq.sequence))[positions]
            mutations = np.char.add(np.char.add(original_aa_chars,positions.astype(str)),aa_chars) # e.g. A138G
            mutated_seqs_nodes = np.vectorize(self.get_or_create_seq_node)(id=mutations,mutated_sequence=mutated_seqs,parent_seq=current_seq,mutation=mutations)

            reverse_mutations_mask = np.vectorize(lambda mutation_node: mutation_node.id not in current_seq.parent_seqs and not self.is_reverse_mutation(mutation_node.id,current_seq.id))(mutated_seqs_nodes) # disallow  reverse mutations to immediate ancestors, allow traversal to filter out distant reverse mutations
            mutated_seqs_nodes = mutated_seqs_nodes[reverse_mutations_mask]

            accepted_mutations_mask = np.vectorize(self.evaluation_strategy.should_accept_mutated_sequence)(mutated_seqs_nodes,current_seq)
            accepted_mutated_seqs_nodes = mutated_seqs_nodes[accepted_mutations_mask]

            for mutated_seq in accepted_mutated_seqs_nodes:
                mutated_seq.add_parent_seq(current_seq.id)
                current_seq.add_child_seq(mutated_seq.id)
                self.evaluation_strategy.set_mutation_score(mutated_seq,current_seq) 
                self.G.add_node(mutated_seq.id,object=mutated_seq)
                self.G.add_edge(current_seq.id,mutated_seq.id,weight=mutated_seq.mutation_score) 
                if not nx.is_directed_acyclic_graph(self.G): # check if the addition of the edge created a cycle
                    self.G.remove_edge(current_seq.id,mutated_seq.id)

            continue_mutating_mask = np.vectorize(self.evaluation_strategy.should_continue_mutating)(accepted_mutated_seqs_nodes,current_seq)
            continue_mutating_seqs_nodes = accepted_mutated_seqs_nodes[continue_mutating_mask]

            for mutated_seq in continue_mutating_seqs_nodes:
                self.evolve_sequence(current_seq=mutated_seq,generation=generation+1)

            # for pos,aa_char in potential_mutations:
            #     mutated_sequence = current_seq.generate_mutated_sequence(pos,aa_char) 
            #     mutation = f"{pos}{aa_char}"
            #     mutated_seq = self.get_or_create_seq_node(id=mutation,mutated_sequence=mutated_sequence,parent_seq=current_seq,mutation=mutation)
                
            #     if current_seq in mutated_seq.child_seqs: 
            #         continue # disallow  reverse mutations

            #     should_accept_mutation = self.evaluation_strategy.should_accept_mutated_sequence(mutated_seq,current_seq)
            #     if should_accept_mutation: # based on seq probability and similarity
            #         mutated_seq.add_parent_seq(current_seq)
            #         current_seq.add_child_seq(mutated_seq)
            #         self.evaluation_strategy.set_mutation_score(mutated_seq,current_seq) 
            #         self.G.add_node(mutated_seq.id,object=mutated_seq)
            #         self.G.add_edge(current_seq.id,mutated_seq.id,weight=mutated_seq.mutation_score) 
            #         if not nx.is_directed_acyclic_graph(self.G): # check if the addition of the edge created a cycle
            #             self.G.remove_edge(current_seq.id,mutated_seq.id)
                    
            #         should_continue_mutating = self.evaluation_strategy.should_continue_mutating(mutated_seq,current_seq) # based on improvement of mutation score
            #         if should_continue_mutating:
            #             self.evolve_sequence(current_seq=mutated_seq,generation=generation+1)

        else:
            print("Max generations reached for this path.")     
        return # stop evolving since max generations reached

    def get_or_create_seq_node(self,id,mutated_sequence,parent_seq,mutation):
        # mutation resulted in a new sequence 
        if id not in self.G.nodes: 
            mutated_seq = ProteinSequence(id=id,sequence=mutated_sequence,parent_seqs={parent_seq},mutation=mutation) # create new node
        # mutation resulted in an existing sequence
        else:
            mutated_seq = self.G.nodes[id]["object"] # get existing node
        return mutated_seq

    def is_reverse_mutation(self,mutation_seq_id,parent_seq_id):
        orig_aa_char,pos,new_aa_char = mutation_seq_id[0],mutation_seq_id[1:-1],mutation_seq_id[-1]
        reverse_mutation = f"{new_aa_char}{pos}{orig_aa_char}"
        return reverse_mutation==parent_seq_id
        
    def get_path_with_highest_mutation_score(self):
        # get the path with the highest mutation score
        if len(self.G.nodes)==1:
            print("The root node was not mutated, therefore there is are no paths.")
            return None
        path = nx.dag_longest_path(self.G, weight="weight") #works for positive weight values
        # path = nx.shortest_path(self.G, source=self.root_sequence.id, weight="weight", method="dijkstra")
        
        # visualise longest path
        if path:
            self.visualise_graph(self.G.subgraph(path))
        else:
            print("No path found.")
            return None
        return path
    
    def visualise_graph(self,graph,seed=0):
        graph_is_a_dag = nx.is_directed_acyclic_graph(graph)

        if not graph_is_a_dag:
            return "Evolution graph is not a directed acyclic graph therefore topological sorting cannot be applied"

        # compute node size
        in_degrees = dict(graph.in_degree())
        max_in_degree = max(in_degrees.values()) if in_degrees else 1
        min_in_degree = min(in_degrees.values()) if in_degrees else 0
        
        # Normalize node sizes (scaling between 500 and 3000)
        node_sizes = {
            node: 500 + 2500 * (deg - min_in_degree) / (max_in_degree - min_in_degree) if max_in_degree != min_in_degree else 1000
            for node, deg in in_degrees.items()
        }

        # weight the edges
        edge_weights = nx.get_edge_attributes(graph,"weight")
        max_weight=max(edge_weights.values())
        min_weight=min(edge_weights.values())
        norm_weights = {
            k: 0.1 + 0.9 * (v - min_weight) / (max_weight - min_weight) if max_weight != min_weight else 1
            for k, v in edge_weights.items()
        }

        plt.figure(figsize=(12,7))
        # pos =  nx.drawing.nx_agraph.graphviz_layout(graph, prog="dot",seed=seed) 
        pos = nx.spring_layout(graph,k=3,seed=seed) 
        # pos = nx.kamada_kawai_layout(graph,seed=seed)  
        nx.draw(graph, pos, with_labels=True,node_size=[node_sizes[n] for n in graph.nodes()])

        labels = nx.get_node_attributes(graph,"label")
        nx.draw_networkx_labels(graph,pos,labels=labels,font_size=10,)

        edge_widths = [norm_weights.get(edge, 1) * 2 for edge in graph.edges()]
        nx.draw_networkx_edges(graph, pos, arrowsize=20, width=edge_widths)

        plt.title("Evolutionary DAG of FMDVP1")
        plt.show()

    def visualise_evolution_G(self):
        self.visualise_graph(self.G)

        
# compare logits with next step and see the change