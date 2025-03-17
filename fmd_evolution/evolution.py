import copy
from .protein_language_model import ProteinLanguageModel
from .protein_sequence import ProteinSequence
from .mutation_strategy import MutationStrategy, MinLogitPosSub
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from . import SEED

class Evolution:
    def __init__(self,root_sequence,mutation_strategy,ranked_evaluation_strategy,max_generations): 
        self.plm = ProteinLanguageModel() # factory to create ProteinSequence objects
        self.root_sequence = root_sequence
        self.mutation_strategy = mutation_strategy
        self.ranked_evaluation_strategy = ranked_evaluation_strategy
        self.max_generations = max_generations
        self.G = nx.DiGraph()
        root_full_sequence = root_sequence.sequence
        self.root_node_id = root_full_sequence[self.mutation_strategy.start_pos:self.mutation_strategy.end_pos+1]
        self.G.add_node(self.root_node_id,object=root_sequence)
        # evaluation data

    # def evolve_sequence(self,current_seq=None,generation=0):
    #     if current_seq is None:
    #         current_seq = self.root_sequence
    #         current_seq.mutation_score = 0 # root seq is unmutated therefore has min worse score

    #     if generation<self.max_generations: 
    #         # process potential mutations
    #         potential_mutations = self.mutation_strategy.get_next_mutations(current_seq) 
    #         if len(potential_mutations)==0: 
    #             print("No potential mutations found.")
    #             return
    #         print(f"Pool of potential mutations: {potential_mutations}")

    #         for current_aa_char,pos,new_aa_char in potential_mutations:
    #             mutated_sequence = current_seq.generate_mutated_sequence(pos,new_aa_char) 
    #             mutation = f"{current_aa_char}{str(int(pos)+1)}{new_aa_char}" # adjust pos displayed for 1-indexing
    #             mutated_seq = self.create_seq_node(mutated_sequence=mutated_sequence,parent_seq=current_seq,mutation=mutation)
                
    #             if mutated_seq.id == current_seq.parent_seq or self.is_reverse_mutation(mutated_seq.id,current_seq.id): 
    #                 continue # disallow  reverse mutations
                
    #             mutation_score = self.evaluation_strategy.get_mutation_score(mutated_seq,current_seq) # do not do this in isolation, keep track of all scores for the pool and rank them, then comapre to parents rank
    #             is_mutation_viable = self.evaluation_strategy.is_mutation_viable(current_seq,mutation_score)
    #             if is_mutation_viable: # based on seq probability and similarity
    #                 mutated_seq.set_mutation_score(mutation_score)
    #                 mutated_seq.set_parent_seq(current_seq.id)
    #                 current_seq.add_child_seq(mutated_seq.id)
                    

    #                 current_full_sequence = current_seq.sequence
    #                 current_node_id = current_full_sequence[self.mutation_strategy.start_pos:self.mutation_strategy.end_pos+1]
    #                 mutated_full_sequence = mutated_seq.sequence 
    #                 mutated_node_id = mutated_full_sequence[self.mutation_strategy.start_pos:self.mutation_strategy.end_pos+1]
    #                 self.G.add_node(mutated_node_id,object=mutated_seq)
    #                 self.G.add_edge(current_node_id,mutated_node_id,weight=mutation_score) 
    #                 if not nx.is_directed_acyclic_graph(self.G): # check if the addition of the edge created a cycle
    #                     self.G.remove_edge(current_seq.id,mutated_node_id)
                            
    #                 self.evolve_sequence(current_seq=mutated_seq,generation=generation+1)

    #             # else: # prune branch with non-viable mutation
    #             #     if current_seq.id in self.G.nodes: 
    #             #         self.prune_branch(current_seq.id)
    #             #         print(f"Pruned branch leading to mutation {mutation}")
    #             #     continue

    #     else:
    #         print("Max generations reached for this path.")     
    #     return # stop evolving since max generations reached
    
    def evolve_sequence_with_ranking(self,current_seq=None,generation=0):
        if current_seq is None:
            current_seq = self.root_sequence
            current_seq.mutation_score = 1000 # root seq is unmutated therefore has min worse score
            current_seq.probability = round(self.ranked_evaluation_strategy.get_sequence_probability(current_seq),5)
            current_seq.embedding_distance = 1 # cosine similarity of 1 with itself 

        if generation<self.max_generations: 
            # process potential mutations
            potential_mutations = self.mutation_strategy.get_next_mutations(current_seq) 
            if len(potential_mutations)==0: 
                print("No potential mutations found.")
                return
            print(f"Pool of potential mutations: {potential_mutations}")

            valid_potential_mutations = []
            for current_aa_char,pos,new_aa_char in potential_mutations:
                mutated_sequence = current_seq.generate_mutated_sequence(pos,new_aa_char) 
                mutation = f"{current_aa_char}{str(int(pos))}{new_aa_char}"
                mutated_seq = self.create_seq_node(reference_seq=self.root_sequence,mutated_sequence=mutated_sequence,parent_seq=current_seq,mutation=mutation)
                
                if self.is_reverse_mutation(mutated_seq.id,current_seq.id): 
                    print(f"Reverse mutation detected: {mutated_seq.id} and {current_seq.id}")
                    continue # disallow  reverse mutations >> INDICATE THIS PATH HAS TERMINATED. flipflop 

                if self.is_flip_flop_mutation(mutated_seq.id,current_seq):
                    print(f"Flip-flop mutation detected: {mutated_seq.id} and {current_seq.id}")
                    continue

                valid_potential_mutations.append(mutated_seq)
            
            if len(valid_potential_mutations)==0: 
                print("No valid potential mutations found.")
                
            mutation_scores = self.ranked_evaluation_strategy.get_ranked_mutation_scores(valid_potential_mutations,parent_sequence=current_seq) 
            self.ranked_evaluation_strategy.set_ranked_mutation_scores(valid_potential_mutations, mutation_scores) 
            self.ranked_evaluation_strategy.set_probability_and_embedding_distance(valid_potential_mutations,parent_sequence=current_seq)

            #viable_mutations = self.ranked_evaluation_strategy.get_viable_mutations(valid_potential_mutations,parent_sequence=current_seq)
            viable_mutations = self.ranked_evaluation_strategy.get_viable_mutations_via_pareto(valid_potential_mutations,parent_sequence=current_seq)
            print(f"Viable mutations: {[mutation.mutation for mutation in viable_mutations]}")

            if len(viable_mutations)==0: 
                print("No valid potential mutations found.")
                
            for mutated_seq in viable_mutations:
                mutation_score = mutated_seq.mutation_score
                mutated_seq.set_mutation_score(mutation_score)
                mutated_seq.set_parent_seq(current_seq.id)
                mutated_seq.set_parent_obj(current_seq)
                current_seq.add_child_seq(mutated_seq.id)
                
                current_full_sequence = current_seq.sequence
                current_node_id = current_full_sequence[self.mutation_strategy.start_pos:self.mutation_strategy.end_pos+1]
                mutated_full_sequence = mutated_seq.sequence 
                mutated_node_id = mutated_full_sequence[self.mutation_strategy.start_pos:self.mutation_strategy.end_pos+1]
                self.G.add_node(mutated_node_id,object=mutated_seq)
                self.G.add_edge(current_node_id,mutated_node_id,weight=mutation_score) 
                
            for mutated_seq in viable_mutations:
                self.evolve_sequence_with_ranking(current_seq=mutated_seq,generation=generation+1)

        else:
            print("Max generations reached for this path.")     
        return # stop evolving since max generations reached

    def create_seq_node(self,reference_seq,mutated_sequence,parent_seq,mutation):
        mutated_seq = self.plm.create_protein_sequence(id=mutation,reference_seq=reference_seq,sequence=mutated_sequence,parent_seq=parent_seq,mutation=mutation) # create new node
        return mutated_seq

    def is_reverse_mutation(self,mutation_seq_id,parent_seq_id):
        orig_aa_char,pos,new_aa_char = mutation_seq_id[0],mutation_seq_id[1:-1],mutation_seq_id[-1]
        reverse_mutation = f"{new_aa_char}{pos}{orig_aa_char}"
        return reverse_mutation==parent_seq_id
    
    def is_flip_flop_mutation(self,mutated_seq_id,parent_seq_obj):
        parent_parent_seq = parent_seq_obj.parent_obj
        if parent_parent_seq is None or parent_parent_seq.id=="base": # current seq is root node
            return False
        parent_parent_seq_id = parent_parent_seq.id
        if mutated_seq_id == parent_parent_seq_id:
            return True
        return False

    def get_best_paths_in_order(self):
        path_mean_mutation_scores = []
        leaf_nodes = [node for node in self.G.nodes if self.G.out_degree(node)==0] # use to filter out paths to intermediate nodes
        for leaf in leaf_nodes:
            path = nx.shortest_path(self.G,source=self.root_node_id,target=leaf,weight="weight")
            path_mutation_scores = [self.G.nodes[seq_id]["object"].mutation_score for seq_id in path]
            print(f"Mutation scores for path {path}: {path_mutation_scores}")
            mean_mutation_score = (sum(path_mutation_scores)-1000)/len(path_mutation_scores)
            path_mean_mutation_scores.append((mean_mutation_score,path))

        #best_paths_in_order = sorted(path_mean_mutation_scores, key=lambda x:x[0]) 
        return path_mean_mutation_scores #best_paths_in_order




        path_distances,shortest_paths_to_every_node = nx.single_source_dijkstra(self.G, source=self.root_sequence.id,weight="weight")
        shortest_path_to_leaf_nodes = {node:shortest_paths_to_every_node[node] for node in leaf_nodes}
        path_distances_to_leaf_nodes = {node:path_distances[node] for node in leaf_nodes}

        return path_distances_to_leaf_nodes,shortest_path_to_leaf_nodes
    
    def sort_paths_by_mutation_score(self,distances,paths):
        sorted_paths = sorted(paths.items(), key=lambda item: distances[item[0]])
        return sorted_paths
    
    def visualise_graph(self,path=None,title="Evolutionary Tree of FMDV-VP1 Protein"):
        if path is None:
            graph = self.G # visualise the entire graph
        else:
            graph = self.G.subgraph(path) 

        # # compute node size
        # in_degrees = dict(graph.in_degree())
        # max_in_degree = max(in_degrees.values()) if in_degrees else 1
        # min_in_degree = min(in_degrees.values()) if in_degrees else 0
        
        # # Normalize node sizes (scaling between 500 and 3000)
        # node_sizes = {
        #     node: 500 + 2500 * (deg - min_in_degree) / (max_in_degree - min_in_degree) if max_in_degree != min_in_degree else 1000
        #     for node, deg in in_degrees.items()
        # }

        # weight the edges
        edge_weights = nx.get_edge_attributes(graph,"weight")
        # max_weight=max(edge_weights.values())
        # min_weight=min(edge_weights.values())
        # norm_weights = {
        #     k: 0.1 + 0.9 * (v - min_weight) / (max_weight - min_weight) if max_weight != min_weight else 1
        #     for k, v in edge_weights.items()
        # }

        node_colors = [
            'red' if node == self.root_node_id  else # root node
            'green' if self.G.out_degree(node)==0. else # leaf node
            'lightblue' for node in graph.nodes()] # intermediate node

        plt.figure(figsize=(12,7))
        # pos = nx.spring_layout(graph,k=3,seed=SEED) 
        # pos = nx.multipartite_layout(graph)
        pos = nx.kamada_kawai_layout(graph,weight="weight",scale=3)  
        nx.draw(graph, pos, node_color=node_colors, with_labels=True) #node_size=[node_sizes[n] for n in graph.nodes()]

        labels = nx.get_node_attributes(graph,"label")
        nx.draw_networkx_labels(graph,pos,labels=labels,font_size=10,)

        # edge_widths = [norm_weights.get(edge, 1) * 2 for edge in graph.edges()]
        nx.draw_networkx_edges(graph, pos, arrowsize=20) #width=edge_widths

        plt.title(title)
        plt.show()

