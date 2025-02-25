import numpy as np
import torch
import torch.nn.functional as F
np.random.seed = 42 # for reproducibility

class EvaluationStrategy:
    def __init__(self,root_sequence,tolerance=0.1,p_tolerance=0.1,f_tolerance=0.5):
        self.root_sequence = root_sequence
        self.tolerance = tolerance
        self.p_tolerance = p_tolerance
        self.f_tolerance = f_tolerance
        self.root_p = None
        self.root_p = self.get_sequence_likelihood(self.root_sequence)
        self.max_p = self.root_p + self.p_tolerance
        self.max_s_score = self.f_tolerance

    def get_sequence_likelihood(self,sequence):
        sequence_aa_logits = sequence.sequence_aa_logits
        log_p = torch.log(sequence_aa_logits) # epsilon to avoid nan
        mean_log_p = torch.mean(log_p)
        sequence_p = torch.exp(mean_log_p)
        return sequence_p.item() # so both metrics are in the same direction, min value is better
    
    # def is_sequence_functional(self,sequence_p): # sequence probability as approximation of fitness
    #     return sequence_p<=self.max_p
    
    # def is_sequence_likelihood_increasing(self, seq_p,parent_seq_p):
    #     return seq_p<=parent_seq_p # check if decreasing as score is inverted

    def get_embedding_distance(self,sequence,parent_sequence):
        sequence_embedding = sequence.embeddings
        parent_sequence_embedding = parent_sequence.embeddings
        cosine_similarity = F.cosine_similarity(sequence_embedding,parent_sequence_embedding,dim=1)
        embedding_distance = cosine_similarity.mean().item() # so both metrics are in the same direction, min value is better
        # l2_distance = torch.norm(sequence_embedding - parent_sequence_embedding, p=2, dim=1)
        # mean_l2_distance = torch.mean(l2_distance).item()
        # functional_score = mean_l2_distance 
        return embedding_distance

    # def is_sequence_structurally_similar(self,sequence_functional_score):
    #     return sequence_functional_score <= self.max_s_score
    
    # def is_sequence_structure_similarity_improving(self,seq_f_score,parent_seq_f_score):
    #     return seq_f_score<parent_seq_f_score
    
    def get_sequence_scores(self,sequence,parent_sequence):
        sequence_p = self.get_sequence_likelihood(sequence)
        sequence_f_score = self.get_embedding_distance(sequence,parent_sequence)
        return sequence_p,sequence_f_score

    def normalise_p_change(self,p_change,min_p=-0.03,max_p=0.004):
        # normalise value to be between -1 and 1
        normalised_p_change = 2*(p_change-min_p)/(max_p-min_p)-1
        return normalised_p_change
    
    def normalise_f_score(self,f_score,min_f=0.0003,max_f=0.006):
        # normalise value to be between -1 and 1
        normalised_f_score = 2*(f_score-min_f)/(max_f-min_f)-1
        return normalised_f_score

    def get_mutation_score(self,sequence,parent_sequence,p_scaling_factor=10,c_scaling_factor=50000,p_cluster=0.8,c_cluster=0.999):
        sequence_p = round(self.get_sequence_likelihood(sequence),5)
        scaled_sequence_p = torch.sigmoid(torch.tensor((sequence_p - p_cluster) * p_scaling_factor)).item()
        #print(f"sequence_p: {sequence_p}")

        cosine_sim = round(self.get_embedding_distance(sequence,parent_sequence),5)
        #print(f"cosine_sim: {cosine_sim}")
        scaled_cosine_sim = torch.sigmoid(torch.tensor((cosine_sim - c_cluster) * c_scaling_factor)).item() # scale down to range of p_change

        mutation_score = (scaled_sequence_p+scaled_cosine_sim)/2
        #mutation_score = torch.sigmoid(torch.tensor(average_change)).item() # flip so that lower score is a better mutation, needed to minimise graph
        #print(f"mutation_score: {mutation_score}")
        return mutation_score
    
    # def should_accept_mutated_sequence(self,sequence,parent_sequence):
    #     sequence_p,sequence_f_score = self.get_sequence_scores(sequence,parent_sequence)
    #     is_functional = self.is_sequence_functional(sequence_p)
    #     is_sequence_structurally_similar = self.is_sequence_structurally_similar(sequence_f_score)
    #     return is_functional and is_sequence_structurally_similar


    # def should_continue_mutating(self,sequence,parent_sequence):  
    #     sequence_mutation_score = sequence.mutation_score
    #     parent_sequence_mutation_score = parent_sequence.mutation_score
    #     if sequence_mutation_score<=parent_sequence_mutation_score: # use change in mutation score over probability and structure for robustness
    #         return True # mutate
    #     else:
    #         return False # terminate path

    def is_mutation_viable(self,parent_sequence,sequence_mutation_score):
        parent_sequence_mutation_score = parent_sequence.mutation_score
        if sequence_mutation_score>=parent_sequence_mutation_score: 
            return True # guaranteed mutation if beneficial change
        else:
            random_chance_for_worse_mutation = np.random.rand()
            return random_chance_for_worse_mutation < self.tolerance
        
        
    # other ascpects to evaluate: increase protein fitness + antigenic pressure if applicable + env via past data
    
    