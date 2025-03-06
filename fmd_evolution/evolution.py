import copy
from .protein_sequence import ProteinSequence
from .mutation_strategy import MutationStrategy, MinLogitPosSub
from .evaluation_strategy import EvaluationStrategy
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from . import SEED

class Evolution:
    def __init__(self,root_sequence,mutation_strategy,evaluation_strategy,max_generations): #evaluation_strategy
        self.root_sequence = root_sequence
        self.mutation_strategy = mutation_strategy
        self.evaluation_strategy = evaluation_strategy
        self.max_generations = max_generations
        self.G = nx.DiGraph()
        self.G.add_node(root_sequence.id,object=root_sequence)
        # evaluation data
        self.np_alignments_seq_records = None

    def evolve_sequence(self,current_seq=None,generation=0):
        if current_seq is None:
            current_seq = self.root_sequence
            current_seq.mutation_score = 0 # root seq is unmutated therefore has min worse score

        if generation<self.max_generations: 
            # process potential mutations
            potential_mutations = self.mutation_strategy.get_next_mutations(current_seq) 
            if len(potential_mutations)==0: 
                print("No potential mutations found.")
                return
            print(f"Pool of potential mutations: {potential_mutations}")

            for current_aa_char,pos,new_aa_char in potential_mutations:
                mutated_sequence = current_seq.generate_mutated_sequence(pos,new_aa_char) 
                mutation = f"{current_aa_char}{str(int(pos)+1)}{new_aa_char}" # adjust pos displayed for 1-indexing
                mutated_seq = self.get_or_create_seq_node(id=mutation,mutated_sequence=mutated_sequence,parent_seq=current_seq,mutation=mutation)
                
                if mutated_seq.id in current_seq.parent_seqs or self.is_reverse_mutation(mutated_seq.id,current_seq.id): 
                    continue # disallow  reverse mutations
                
                mutation_score = self.evaluation_strategy.get_mutation_score(mutated_seq,current_seq) # do not do this in isolation, keep track of all scores for the pool and rank them, then comapre to parents rank
                is_mutation_viable = self.evaluation_strategy.is_mutation_viable(current_seq,mutation_score)
                if is_mutation_viable: # based on seq probability and similarity
                    mutated_seq.set_mutation_score(mutation_score)
                    mutated_seq.add_parent_seq(current_seq.id)
                    current_seq.add_child_seq(mutated_seq.id)
                    self.G.add_node(mutated_seq.id,object=mutated_seq)
                    self.G.add_edge(current_seq.id,mutated_seq.id,weight=mutation_score) 
                    if not nx.is_directed_acyclic_graph(self.G): # check if the addition of the edge created a cycle
                        self.G.remove_edge(current_seq.id,mutated_seq.id)
                            
                    self.evolve_sequence(current_seq=mutated_seq,generation=generation+1)

                # else: # prune branch with non-viable mutation
                #     if current_seq.id in self.G.nodes: 
                #         self.prune_branch(current_seq.id)
                #         print(f"Pruned branch leading to mutation {mutation}")
                #     continue

        else:
            print("Max generations reached for this path.")     
        return # stop evolving since max generations reached
    
    # def evolve_sequence_with_ranking(self,current_seq=None,generation=0):
    #     if current_seq is None:
    #         current_seq = self.root_sequence
    #         current_seq.mutation_score = 0 # root seq is unmutated therefore has min worse score

    #     if generation<self.max_generations: 
    #         # process potential mutations
    #         potential_mutations = self.mutation_strategy.get_next_mutations(current_seq) 
    #         if len(potential_mutations)==0: 
    #             print("No potential mutations found.")
    #             return

    #         mutated_seq_nodes = []
            
    #         for current_aa_char,pos,new_aa_char in potential_mutations:
    #             mutated_sequence = current_seq.generate_mutated_sequence(pos,new_aa_char) 
    #             mutation = f"{current_aa_char}{str(int(pos)+1)}{new_aa_char}" # adjust pos displayed for 1-indexing
    #             mutated_seq = self.get_or_create_seq_node(id=mutation,mutated_sequence=mutated_sequence,parent_seq=current_seq,mutation=mutation)
                
    #             if mutated_seq.id in current_seq.parent_seqs or self.is_reverse_mutation(mutated_seq.id,current_seq.id): 
    #                 continue # disallow  reverse mutations
    #             else:
    #                 mutated_seq_nodes.append(mutated_seq)
            
    #         # remove potential mutations that will creates a cycle
    #         mutated_seq_nodes_copy = copy.deepcopy(mutated_seq_nodes)
    #         for mutated_seq in mutated_seq_nodes_copy: # make a copy insterad   
    #             G_copy = copy.deepcopy(self.G)
    #             G_copy.add_node(mutated_seq.id,object=mutated_seq)
    #             G_copy.add_edge(current_seq.id,mutated_seq.id) 
    #             if nx.is_directed_acyclic_graph(self.G_copy): # check if the addition of the edge created a cycle
    #                 mutated_seq_nodes.remove(mutated_seq) 
                
    #         self.evaluation_strategy.set_ranked_mutation_scores(mutated_seq_nodes,parent_sequence=current_seq) 

    #         for mutated_seq in mutated_seq_nodes:
    #             mutation_score = mutated_seq.mutation_score
    #             is_mutation_viable = self.evaluation_strategy.is_mutation_viable(current_seq,mutation_score)
    #             if is_mutation_viable: # based on seq probability and similarity
    #                 mutated_seq.set_mutation_score(mutation_score)
    #                 mutated_seq.add_parent_seq(current_seq.id)
    #                 current_seq.add_child_seq(mutated_seq.id)
    #                 self.G.add_node(mutated_seq.id,object=mutated_seq)
    #                 self.G.add_edge(current_seq.id,mutated_seq.id,weight=mutation_score) 
                            
    #                 self.evolve_sequence(current_seq=mutated_seq,generation=generation+1)

    #             # else: # prune branch with non-viable mutation
    #             #     if current_seq.id in self.G.nodes: 
    #             #         self.prune_branch(current_seq.id)
    #             #         print(f"Pruned branch leading to mutation {mutation}")
    #             #     continue

    #     else:
    #         print("Max generations reached for this path.")     
    #     return # stop evolving since max generations reached

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
    
    def prune_branch(self,seq_id):
        if seq_id not in self.G.nodes:
            return 
        seq = self.G.nodes[seq_id]
        if "object" not in seq: 
            return  
        if self.G.out_degree(seq_id)==0: # only prune if it is a leaf node
            seq = self.G.nodes[seq_id]["object"]
            parent_seqs = seq.parent_seqs
            self.G.remove_node(seq_id)

            for parent_id in parent_seqs:
                if self.G.out_degree(parent_id)==0: 
                    self.prune_branch(parent_id) # recursively prune ancestor nodes with no other children

        return
    
    def get_best_paths_in_order(self):
        if not nx.is_directed_acyclic_graph(self.G):
            print("Topological sorting cannot be applied as graph is not a directed acyclic graph.")
            return None
        
        path_mean_mutation_scores = []
        leaf_nodes = [node for node in self.G.nodes if self.G.out_degree(node)==0] # use to filter out paths to intermediate nodes
        for leaf in leaf_nodes:
            path = nx.shortest_path(self.G,source=self.root_sequence.id,target=leaf,weight="weight")
            path_mutation_scores = [self.G.nodes[seq_id]["object"].mutation_score for seq_id in path]
            mean_mutation_score = sum(path_mutation_scores)/len(path_mutation_scores)
            path_mean_mutation_scores.append((mean_mutation_score,path))

        best_paths_in_order = sorted(path_mean_mutation_scores, key=lambda x:x[0], reverse=True) # maximise score
        return best_paths_in_order




        path_distances,shortest_paths_to_every_node = nx.single_source_dijkstra(self.G, source=self.root_sequence.id,weight="weight")
        shortest_path_to_leaf_nodes = {node:shortest_paths_to_every_node[node] for node in leaf_nodes}
        path_distances_to_leaf_nodes = {node:path_distances[node] for node in leaf_nodes}

        return path_distances_to_leaf_nodes,shortest_path_to_leaf_nodes
    
    def sort_paths_by_mutation_score(self,distances,paths):
        sorted_paths = sorted(paths.items(), key=lambda item: distances[item[0]])
        return sorted_paths
    
    def visualise_graph(self,path=None,seed=0):
        if path is None:
            graph = self.G # visualise the entire graph
        graph = self.G.subgraph(path) 
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

    def plot_path_mutation_matches(self,percentage_of_mutation_matches):
        # a bar plot with error bars 

        return

    def evaluate_path_using_alignments(self,evolutionary_path):
        data_length = len(self.np_alignments_seq_records)
        mutation_matches_list = []
        for seq_id in evolutionary_path:
            if seq_id==self.root_sequence.id:
                continue # root seq is unmutated so no mutation to evaluate
            percentage_of_mutation_matches = self.evaluate_mutation_only_using_alignments(seq_id,data_length)
            mutation_matches_list.append(percentage_of_mutation_matches)
        return mutation_matches_list
    
    # position-specific identity calculator
    def evaluate_mutation_only_using_alignments(self,seq_id,data_length):
        mutated_pos = int(seq_id[1:-1])-1 # adjust pos to 0-indexing
        new_aa = seq_id[-1] 
        # check percentage of matches for the mutated position containing the new amino acid
        relative_mutated_pos = mutated_pos - self.mutation_strategy.start_pos # start pos is already 0-indexed so add 1
        alignment_amino_acids = np.array([seq_record[relative_mutated_pos] for seq_record in self.np_alignments_seq_records])
        num_of_matches = np.sum(alignment_amino_acids==new_aa)
        percentage_of_matches = round((num_of_matches/data_length)*100,2)
        print(f"% of mutation matches for {seq_id}: {percentage_of_matches}%")
        return percentage_of_matches

    def process_alignment_data(self,file_path):
        seq_records = list(SeqIO.parse(file_path, "fasta"))
        self.np_alignments_seq_records = np.array([str(record.seq) for record in seq_records])
        return self.np_alignments_seq_records
