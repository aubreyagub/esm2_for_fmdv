from abc import ABC, abstractmethod
from protein_sequence import ProteinSequence
import numpy as np

class MutationStrategy(ABC):
    @abstractmethod
    def __init__(self,alphabet):
        self.alphabet = alphabet
        self.start_pos = 138
        self.end_pos = 143
        self.token_offset=4 # index of aa in alphabet begins at 4
    
    @abstractmethod
    def get_next_mutation(self,current_seq,all_logits,current_seq_logits,start_pos=138,end_pos=143):
        pass

    def index_to_char(self,aa):
        return self.alphabet.all_toks[aa+self.token_offset] 

    def get_aa_char(self,current_aa,new_aa_index):
        aa_char = self.index_to_char(new_aa_index)
        if current_aa!=aa_char:
            return aa_char
        else:
            print(f"Invalid amino acid candidate for mutation as it is the same as the current amino acid: {current_aa}>{aa_char}")
            return None
            
    def get_top_2_candidates(self, scores):
        return np.argsort(scores)[-2:][::-1] # top 2 in case new aa == current aa

    def get_new_amino_acid(self,current_seq,candidates,min_logit_pos):
        current_aa = list(current_seq)[min_logit_pos]
        max_logit_aa = candidates[0]
        
        aa_char = self.get_aa_char(current_aa,max_logit_aa)
        if not aa_char:
            second_max_logit_aa = candidates[1]
            aa_char = self.get_aa_char(current_aa,second_max_logit_aa)
            print(f"Using the second best fit amino acid for this position: {current_aa}>{aa_char}")
        else:
            print(f"The top amino acid candidate for mutation is valid for this position: {current_aa}>{aa_char}")
        return aa_char
        
# Mutate through substitution the position with the minimum logit 
class MinLogitPosSub(MutationStrategy):
    def __init__(self,alphabet):
        super().__init__(alphabet)

    def get_next_mutation(self,current_seq,all_logits,current_seq_logits,start_pos=138,end_pos=143):
        poss_of_interest_current_seq_logits = current_seq_logits[self.start_pos:self.end_pos+1]
        min_logit_pos_relative = np.argmin(poss_of_interest_current_seq_logits)
        min_logit_pos = (self.start_pos+min_logit_pos_relative).item()

        poss_of_interest_logits = all_logits[self.start_pos:self.end_pos+1,:]
        min_logit_pos_logits = poss_of_interest_logits[min_logit_pos_relative]

        # select new aa
        candidates = self.get_top_2_candidates(min_logit_pos_logits.numpy())
        aa_char = self.get_new_amino_acid(current_seq,candidates,min_logit_pos)
        
        return min_logit_pos,aa_char

class BlosumWeightedSub(MutationStrategy):
    def __init__(self,alphabet,blosum_matrix,multiplier=None):
        super().__init__(alphabet)
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

    def get_next_mutation(self,current_seq,all_logits,current_seq_logits,start_pos=138,end_pos=143):
        poss_of_interest_current_seq_logits = current_seq_logits[self.start_pos:self.end_pos+1]
        min_logit_pos_relative = np.argmin(poss_of_interest_current_seq_logits)
        min_logit_pos = (self.start_pos+min_logit_pos_relative).item()

        poss_of_interest_logits = all_logits[self.start_pos:self.end_pos+1,:]
        min_logit_pos_logits = poss_of_interest_logits[min_logit_pos_relative]

        # get weighted scores
        current_aa = list(current_seq)[min_logit_pos]
        blosum_scores = self.get_blosum_scores(current_aa)
        weighted_scores = self.weight_logits_with_blosum(blosum_scores,min_logit_pos_logits)
        
        # select new aa
        candidates = self.get_top_2_candidates(np.array(weighted_scores))
        aa_char = self.get_new_amino_acid(current_seq,candidates,min_logit_pos)
        
        return min_logit_pos,aa_char
    
