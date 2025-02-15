import torch
import torch.nn.functional as F


class EvaluationStrategy:
    def __init__(self,root_sequence,p_tolerance=0.1,f_tolerance=0.5):
        self.root_sequence = root_sequence
        self.p_tolerance = p_tolerance
        self.f_tolerance = f_tolerance
        self.root_p = None
        self.root_p = self.get_sequence_probability(self.root_sequence)
        self.min_p = self.root_p - self.p_tolerance
        self.min_s_score = 1 - self.f_tolerance

    def get_sequence_probability(self,sequence):
        sequence_aa_logits = sequence.sequence_aa_logits
        log_p = torch.log(sequence_aa_logits) # epsilon to avoid nan
        mean_log_p = torch.mean(log_p)
        sequence_p = torch.exp(mean_log_p)
        return sequence_p
    
    def is_sequence_functional(self,sequence_p): # sequence probability as approximation of fitness
        return self.min_p <= sequence_p
    
    def is_sequence_probability_increasing(self, seq_p,parent_seq_p):
        return seq_p>parent_seq_p 

    def get_sequence_structure_score(self,sequence,parent_sequence):
        sequence_embedding = sequence.embeddings
        parent_sequence_embedding = parent_sequence.embeddings
        # compare with previous embeddings using a distance metric: cosine similarity
        # cosine_sim = F.cosine_similarity(sequence_embedding,previous_sequence_embedding,dim=1)
        # mean_cosine_sim = torch.mean(cosine_sim).item()
        # print(f"Cosine similarity = {mean_cosine_sim}")
        # return mean_cosine_sim> self.functional_threshold,mean_cosine_sim
        l2_distance = torch.norm(sequence_embedding - parent_sequence_embedding, p=2, dim=1)
        mean_l2_distance = torch.mean(l2_distance).item()
        functional_score = 1 - mean_l2_distance # so both metrics are in the same direction
        return functional_score

    def is_sequence_structurally_similar(self,sequence_functional_score):
        return self.min_s_score <= sequence_functional_score 
    
    def is_sequence_structure_increasing(self,seq_f_score,parent_seq_f_score):
        return seq_f_score>parent_seq_f_score
    
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

    def should_continue_mutating(self,sequence,parent_sequence):  ######### refactor to only check threshold
        sequence_p,sequence_f_score = self.get_sequence_scores(sequence,parent_sequence)
        parent_sequence_p,parent_f_score = self.get_parent_scores(parent_sequence)

        is_probability_increasing = self.is_sequence_probability_increasing(sequence_p,parent_sequence_p)
        is_sequence_structure_increasing = self.is_sequence_structure_increasing(sequence_f_score,parent_f_score)

        if is_probability_increasing and is_sequence_structure_increasing:
            return True 
        else:
            return False
    
        
        
    # other ascpects to evaluate: increase protein fitness + antigenic pressure if applicable + env via past data
    
    