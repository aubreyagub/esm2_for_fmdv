from abc import ABC, abstractmethod
from protein_sequence import ProteinSequence
from model_singleton import ModelSingleton
import numpy as np
import torch

class MutationStrategy(ABC):
    @abstractmethod
    def __init__(self,mutations_per_seq=20,start_pos=138,end_pos=143):
        self.mutations_per_seq = mutations_per_seq
        self.alphabet = ModelSingleton().get_alphabet()
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.token_offset=4 # index of aa in alphabet begins at 4
    
    def index_to_char(self,aa_pos):
        return self.alphabet.all_toks[aa_pos+self.token_offset] 

    def get_top_n_mutations(self, scores):
        return np.argsort(scores)[-(self.mutations_per_seq+1):][::-1] # one extra in case current_aa is a potential mutation

    def get_new_amino_acid(self,current_aa,aa_pos):
        aa_char = self.index_to_char(aa_pos)
        if aa_char:
            return aa_char
        return None # no new aa
    
    def get_average_min_logit_pos(self,all_aa_logits): 
        pos_mean_logits = torch.mean(all_aa_logits,dim=1) # use average over direct aa logit for robustness 
        relevant_segment_logits = pos_mean_logits[self.start_pos:self.end_pos+1]
        relative_min_logit_position = torch.argmin(relevant_segment_logits).item()
        absolute_min_logit_position = self.start_pos+relative_min_logit_position
        return relative_min_logit_position,absolute_min_logit_position
    
    # def get_min_logit_pos(self,sequence_aa_logits):
    #     relevant_segment_logits = sequence_aa_logits[self.start_pos:self.end_pos+1]
    #     relative_min_logit_position = np.argmin(relevant_segment_logits)
    #     absolute_min_logit_position = (self.start_pos+relative_min_logit_position).item()
    #     return relative_min_logit_position,absolute_min_logit_position
    
    def get_position_logit_values(self,relative_position,all_aa_logits):
        relevant_segment_amino_acids_logits = all_aa_logits[self.start_pos:self.end_pos+1,:]
        return relevant_segment_amino_acids_logits[relative_position]
    
    def get_min_logit_pos_and_values(self,sequence_aa_logits,all_aa_logits):
        relative_min_logit_position,absolute_min_logit_position = self.get_average_min_logit_pos(all_aa_logits) ##################
        aa_logits_at_min_logit_position = self.get_position_logit_values(relative_min_logit_position,all_aa_logits)

        return absolute_min_logit_position,aa_logits_at_min_logit_position
    
    def validate_potential_mutations(self,current_seq,min_logit_pos,potential_aa_positions):
        current_aa = list(current_seq)[min_logit_pos]
        # print(f"Potential aa mutation positons: {potential_aa_positions}")
        mutations = []
        for aa_pos in potential_aa_positions:
            aa_char = self.get_new_amino_acid(current_aa,aa_pos)  
            if aa_char!=current_aa: 
                mutations.append((min_logit_pos,aa_char))
        mutations = mutations[:self.mutations_per_seq] # ensure only the specified number of mutations are returned
        return mutations

    def acceptance_ratio(self,current_seq):
        # Metropolis-Hastings acceptance ratio
        new_aa_p = None
        current_aa_p = None
        new_aa_given_current = None
        current_aa_given_new = None
        ratio = (new_aa_p/current_aa_p)*(current_aa_given_new/new_aa_given_current)
        return min(1,ratio)
    
    def should_accept_mutation(self,mutated_seq):
        return True

# Mutate through substitution the position with the minimum logit 
class MinLogitPosSub(MutationStrategy):
    def __init__(self,mutations_per_seq=20,start_pos=138,end_pos=143):
        super().__init__(mutations_per_seq,start_pos,end_pos)

    def get_next_mutations(self,current_seq):
        sequence = current_seq.sequence
        all_aa_logits = current_seq.all_aa_logits
        sequence_aa_logits = current_seq.sequence_aa_logits 

        # get position with minimum logit score and its logit scores
        min_logit_pos,aa_logits_at_min_logit_position = self.get_min_logit_pos_and_values(sequence_aa_logits,all_aa_logits)

        # generate mutations
        potential_aa_positions = self.get_top_n_mutations(aa_logits_at_min_logit_position.numpy())
        mutations = self.validate_potential_mutations(sequence,min_logit_pos,potential_aa_positions)
        return mutations

# Use logic of MinLogitPosSub, weighted/penalty using blosum scores
class BlosumWeightedSub(MutationStrategy):
    def __init__(self,blosum_matrix,multiplier=None,mutations_per_seq=20,start_pos=138,end_pos=143):
        super().__init__(mutations_per_seq,start_pos,end_pos)
        self.blosum_matrix = blosum_matrix
        self.multiplier = multiplier

    def get_blosum_scores(self,aa_char):
        row_for_char = self.blosum_matrix.get(aa_char)
        only_valid_amino_acid_cols = row_for_char[0:-4] # final 5 columns are not standard amino acids
        return only_valid_amino_acid_cols

    def weight_logits_with_blosum(self,blosum_scores,logit_scores):
        blosum_alphabet = self.blosum_matrix.alphabet[0:-4] # final 5 columns are not standard amino acids
        weighted_scores = [0 for i in range(20)] 
        for aa_char,b_score in zip(blosum_alphabet,blosum_scores):
            logit_index = self.alphabet.get_idx(aa_char) - self.token_offset
            logit_val = logit_scores[logit_index] 
            if self.multiplier: # weighting enabled
                weighted_scores[logit_index] = logit_val+(self.multiplier*b_score)
            else:
                weighted_scores[logit_index] = logit_val # unweighted scores, same as MinLogitPosSub
        return weighted_scores

    def get_next_mutations(self,current_seq):     
        sequence = current_seq.sequence
        all_aa_logits = current_seq.all_aa_logits
        sequence_aa_logits = current_seq.sequence_aa_logits 
        min_logit_pos,aa_logits_at_min_logit_position = self.get_min_logit_pos_and_values(sequence_aa_logits,all_aa_logits)
        
        # get new weighted scores for amino acids in given pos
        current_aa = list(sequence)[min_logit_pos]
        blosum_scores = self.get_blosum_scores(current_aa)
        weighted_scores = self.weight_logits_with_blosum(blosum_scores,aa_logits_at_min_logit_position)

        # generate mutations
        potential_aa_positions = self.get_top_n_mutations(np.array(weighted_scores))
        mutations = self.validate_potential_mutations(sequence,min_logit_pos,potential_aa_positions)
        return mutations
