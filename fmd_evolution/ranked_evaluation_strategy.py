import numpy as np
import torch
import torch.nn.functional as F
from . import SEED, rng

class RankedEvaluationStrategy:
    def __init__(self,root_sequence):
        self.root_sequence = root_sequence
        self.root_p = None
        self.root_p = self.get_sequence_probability(self.root_sequence)

    def get_sequence_probability(self,sequence):
        sequence_aa_logits = sequence.sequence_aa_logits
        log_p = torch.log(sequence_aa_logits)
        mean_log_p = torch.mean(log_p)
        sequence_p = torch.exp(mean_log_p)
        return sequence_p.item()
    
    def get_embedding_distance(self,sequence,parent_sequence):
        sequence_embedding = sequence.embeddings
        parent_sequence_embedding = parent_sequence.embeddings
        cosine_similarity = F.cosine_similarity(sequence_embedding,parent_sequence_embedding,dim=1)
        embedding_distance = cosine_similarity.mean().item()
        return embedding_distance

    def get_mutation_scores(self,mutated_nodes,parent_sequence):
        seq_p_list = []
        cosine_sim_list = []

        for sequence in mutated_nodes:
            sequence_p = round(self.get_sequence_probability(sequence),5)
            seq_p_list.append(sequence_p)
            cosine_sim = round(self.get_embedding_distance(sequence,parent_sequence),5)
            cosine_sim_list.append(cosine_sim)   
        
        return seq_p_list,cosine_sim_list
    
    def get_ranked_mutation_scores(self,mutated_nodes,parent_sequence):
        seq_p_list,cosine_sim_list = self.get_mutation_scores(mutated_nodes,parent_sequence)
        # rank the mutations based on the two metrics
        mutated_seqs_sorted_by_p = sorted(mutated_nodes,  key=lambda item: seq_p_list, reverse=True) # higher probability is better 
        mutated_seqs_sorted_by_cs = sorted(mutated_nodes,  key=lambda item: cosine_sim_list, reverse=True) # higher cosine similarity is better 

        # calculate the mutation scores based on rank averages; rank 1 is the best mutation
        mutation_scores = [mutated_seqs_sorted_by_p.index(seq)+mutated_seqs_sorted_by_cs.index(seq) for seq in mutated_nodes]
        return mutation_scores
    
    def set_ranked_mutation_scores(self,mutated_nodes,mutation_scores):
        for sequence,score in zip(mutated_nodes,mutation_scores):
            sequence.set_mutation_score(score)
            
        return 
    
    
    def get_viable_mutations(self,potential_mutations,parent_sequence):
        parent_sequence_mutation_score = parent_sequence.mutation_score
        viable_mutations = []
        for sequence in potential_mutations:
            sequence_mutation_score = sequence.mutation_score
            if sequence_mutation_score>=parent_sequence_mutation_score: 
                viable_mutations.append(sequence) # guaranteed mutation if beneficial change
            else:
                # convert to metropolis hastings later
                continue

        return viable_mutations