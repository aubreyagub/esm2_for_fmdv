import numpy as np
import torch
import torch.nn.functional as F
from . import SEED, rng

class RankedEvaluationStrategy:
    def __init__(self,root_sequence,num_of_mutations_desired=3):
        self.root_sequence = root_sequence
        self.root_p = None
        self.root_p = self.get_sequence_probability(self.root_sequence)
        self.num_of_mutations_desired = num_of_mutations_desired
        
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
        mutation_data = list(zip(mutated_nodes,seq_p_list,cosine_sim_list))
        
        # rank the mutations based on the two metrics
        mutated_seqs_sorted_by_p = sorted(mutation_data,  key=lambda x:x[1], reverse=True) # higher probability is better 
        p_ranks ={mutation[0]:rank for rank,mutation in enumerate(mutated_seqs_sorted_by_p)}
        mutated_seqs_sorted_by_cs = sorted(mutation_data,  key=lambda x:x[2]) # lower cosine similarity is better 
        cs_ranks ={mutation[0]:rank for rank,mutation in enumerate(mutated_seqs_sorted_by_cs)}

        # calculate the mutation scores based on rank averages; rank 1 is the best mutation
        mutation_scores = [p_ranks[seq]+cs_ranks[seq] for seq in mutated_nodes]
        return mutation_scores
    
    def set_ranked_mutation_scores(self,mutated_nodes,mutation_scores):
        for sequence,score in zip(mutated_nodes,mutation_scores):
            sequence.set_mutation_score(score)
            
        return 
    
    def apply_softmax_to_ranked_scores(self, mutation_scores):
        mutation_scores_torch = torch.tensor(mutation_scores,dtype=torch.float32)
        softmax = torch.nn.Softmax(dim=0)
        softmax_scores  = softmax(-mutation_scores_torch) # negative to give ranked scores closer to 0 a probability closer to 1, and T =1 
        return softmax_scores.numpy()
    
    def get_viable_mutations(self,potential_mutations,parent_sequence):
        mutation_scores = self.get_ranked_mutation_scores(potential_mutations,parent_sequence)
        print(f"Mutation scores: {mutation_scores}")
        softmax_scores = self.apply_softmax_to_ranked_scores(mutation_scores)
        print(f"Softmax scores: {softmax_scores}")
        num_of_mutations_desired = min(self.num_of_mutations_desired,len(potential_mutations))
        viable_mutations = rng.choice(potential_mutations,num_of_mutations_desired,replace=False,p=softmax_scores)
        return viable_mutations