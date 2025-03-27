import numpy as np
import torch
import torch.nn.functional as F
from . import SEED, rng

class EvaluationStrategy:
    def __init__(self,root_sequence,num_of_mutations_desired=3):
        self.root_sequence = root_sequence
        self.root_p = None
        self.root_p = self.get_sequence_probability(self.root_sequence)
        self.num_of_mutations_desired = num_of_mutations_desired
        
    def get_sequence_probability(self,sequence):
        sequence_aa_probabilities = sequence.sequence_aa_probabilities
        log_p = torch.log(sequence_aa_probabilities)
        mean_log_p = torch.mean(log_p)
        sequence_p = torch.exp(mean_log_p)
        return sequence_p.item()
    
    def get_embedding_distance(self,sequence,parent_sequence):
        sequence_embedding = sequence.embeddings
        parent_sequence_embedding = parent_sequence.embeddings
        cosine_similarity = F.cosine_similarity(sequence_embedding,parent_sequence_embedding,dim=1)
        mean_cosine_sim = cosine_similarity.mean().item()
        embedding_distance = 1-mean_cosine_sim
        return embedding_distance

    def get_mutation_scores(self,mutated_nodes,parent_sequence):
        seq_p_list = []
        embedding_distance_list = []

        for sequence in mutated_nodes:
            sequence_p = round(self.get_sequence_probability(sequence),5)
            seq_p_list.append(sequence_p)
            distance = round(self.get_embedding_distance(sequence,parent_sequence),5)
            embedding_distance_list.append(distance)   
        
        return seq_p_list,embedding_distance_list
    
    def set_probability_and_embedding_distance(self,mutated_nodes,parent_sequence):
        seq_p_list,embedding_distance_list = self.get_mutation_scores(mutated_nodes,parent_sequence) # unranked scores
        for sequence,seq_p,distance in zip(mutated_nodes,seq_p_list,embedding_distance_list):
            sequence.probability = seq_p
            sequence.embedding_distance = distance
        return
    
    def check_dominate(self,sequence,mutation):
        sequence_p = float(sequence.probability)
        sequence_d = float(sequence.embedding_distance)
        mutation_p = float(mutation.probability)
        mutation_d = float(mutation.embedding_distance)
        # mutation is dominated if sequence is at least as good as mutation in both metrics, and at least one of the metrics is higher
        met_mutation_p = (sequence_p >= mutation_p) 
        met_mutation_d = (sequence_d >= mutation_d)
        surpassed_one = (sequence_p > mutation_p) or (sequence_d > mutation_d)
        is_dominated = met_mutation_p and met_mutation_d and surpassed_one

        return is_dominated
        

    
    def get_viable_mutations(self,potential_mutations,parent_sequence):
        #all_sequences = potential_mutations + [parent_sequence]
        parent_probability = parent_sequence.probability
        parent_embedding_distance = parent_sequence.embedding_distance
        print(f"Parent: mutation:{parent_sequence.id}, probability:{parent_probability}, distance {parent_embedding_distance}")

        viable_mutations = []

        # check for thresholds
        for mutation in potential_mutations:
            mutation_probability = mutation.probability
            mutation_embedding_distance = mutation.embedding_distance

            if mutation_probability>parent_probability or mutation_embedding_distance>parent_embedding_distance:
                print(f"Mutation accepted: {mutation.id}, probability: {mutation_probability}, distance: {mutation_embedding_distance}")
                viable_mutations.append(mutation)
        return viable_mutations

        # # check for pareto dominance
        # for mutation in potential_mutations:
        #     is_dominated = any(
        #         self.check_dominate(sequence,mutation) for sequence in all_sequences if sequence!=mutation) # do not check sequence against itself
        #     if not is_dominated:
        #         print(f"Accepted: mutation:{mutation.id}, probability:{mutation.probability}, distance {mutation.embedding_distance}")
        #         viable_mutations.append(mutation)
        #     else:
        #         print(f"Dominated: mutation:{mutation.id}, probability:{mutation.probability}, distance {mutation.embedding_distance}")
            
        # return viable_mutations
  