import torch
import torch.nn.functional as F


class EvaluationStrategy:
    def __init__(self,root_sequence,p_tolerance=0.1,f_tolerance=0.5):
        self.root_sequence = root_sequence
        self.p_tolerance = p_tolerance
        self.f_tolerance = f_tolerance
        self.root_p = None
        self.root_p = self.get_sequence_probability(self.root_sequence)
        self.max_p = self.root_p + self.p_tolerance
        self.max_s_score = self.f_tolerance

    def get_sequence_probability(self,sequence):
        sequence_aa_logits = sequence.sequence_aa_logits
        log_p = torch.log(sequence_aa_logits) # epsilon to avoid nan
        mean_log_p = torch.mean(log_p)
        sequence_p = torch.exp(mean_log_p)
        return 1-sequence_p.item() # so both metrics are in the same direction, min value is better
    
    def is_sequence_functional(self,sequence_p): # sequence probability as approximation of fitness
        return sequence_p<=self.max_p
    
    def is_sequence_probability_increasing(self, seq_p,parent_seq_p):
        return seq_p<=parent_seq_p # check if decreasing as score is inverted

    def get_sequence_structure_score(self,sequence,parent_sequence):
        sequence_embedding = sequence.embeddings
        parent_sequence_embedding = parent_sequence.embeddings
        l2_distance = torch.norm(sequence_embedding - parent_sequence_embedding, p=2, dim=1)
        mean_l2_distance = torch.mean(l2_distance).item()
        functional_score = mean_l2_distance 
        return functional_score

    def is_sequence_structurally_similar(self,sequence_functional_score):
        return sequence_functional_score <= self.max_s_score
    
    def is_sequence_structure_similarity_improving(self,seq_f_score,parent_seq_f_score):
        return seq_f_score<parent_seq_f_score
    
    def get_sequence_scores(self,sequence,parent_sequence):
        sequence_p = self.get_sequence_probability(sequence)
        sequence_f_score = self.get_sequence_structure_score(sequence,parent_sequence)
        return sequence_p,sequence_f_score

    def get_parent_scores(self,parent_sequence):
        parent_p = self.get_sequence_probability(parent_sequence)
        if parent_sequence.parent_seqs:
            parent_f_score = self.get_sequence_structure_score(parent_sequence,parent_sequence.parent_seqs[-1])
        else:
            parent_f_score = 0 # root node does not have a parent
        return parent_p,parent_f_score

     # create new function to combine score and call in evolution
    def set_mutation_score(self,sequence,parent_sequence):
        sequence_p,sequence_functional_score = self.get_sequence_scores(sequence,parent_sequence)
        average_score = (sequence_p + sequence_functional_score)/2
        sequence.set_mutation_score(average_score) 
        return None
    
    def should_accept_mutated_sequence(self,sequence,parent_sequence):
        sequence_p,sequence_f_score = self.get_sequence_scores(sequence,parent_sequence)
        is_functional = self.is_sequence_functional(sequence_p)
        is_sequence_structurally_similar = self.is_sequence_structurally_similar(sequence_f_score)
        return is_functional and is_sequence_structurally_similar


    def should_continue_mutating(self,sequence,parent_sequence):  
        sequence_mutation_score = sequence.mutation_score
        parent_sequence_mutation_score = parent_sequence.mutation_score
        if sequence_mutation_score<=parent_sequence_mutation_score: # use change in mutation score over probability and structure for robustness
            return True # mutate
        else:
            return False # terminate path

    
        
        
    # other ascpects to evaluate: increase protein fitness + antigenic pressure if applicable + env via past data
    
    