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
        
# Mutate through substitution the position with the minimum logit 
class MinLogitPosSub(MutationStrategy):
    def __init__(self,alphabet):
        super().__init__(alphabet)

    def index_to_char(self,aa):
        return self.alphabet.all_toks[aa+self.token_offset] 

    def get_aa_char(self,current_aa,new_aa_index):
        aa_char = self.index_to_char(new_aa_index)
        if current_aa!=aa_char:
            return aa_char
        else:
            print(f"Invalid amino acid candidate for mutation as it is the same as the current amino acid: {current_aa}>{aa_char}")
            return None

    def get_next_mutation(self,current_seq,all_logits,current_seq_logits,start_pos=138,end_pos=143):
        poss_of_interest_current_seq_logits = current_seq_logits[self.start_pos:self.end_pos+1]
        min_logit_pos_relative = np.argmin(poss_of_interest_current_seq_logits)
        min_logit_pos = (self.start_pos+min_logit_pos_relative).item()

        poss_of_interest_logits = all_logits[self.start_pos:self.end_pos+1,:]
        min_logit_pos_logits = poss_of_interest_logits[min_logit_pos_relative]
        top2_max_logit_aa = np.argsort(min_logit_pos_logits.numpy())[-2:][::-1] # top 2 in case new aa == current aa
        
        max_logit_aa = top2_max_logit_aa[0]
        current_aa = list(current_seq)[min_logit_pos]
        aa_char = self.get_aa_char(current_aa,max_logit_aa)
        if not aa_char:
            second_max_logit_aa = top2_max_logit_aa[1]
            aa_char = self.get_aa_char(current_aa,second_max_logit_aa)
            print(f"Using the second best fit amino acid for this position: {current_aa}>{aa_char}")
        else:
            print(f"The top amino acid candidate for mutation is valid for this position: {current_aa}>{aa_char}")
        
        return min_logit_pos,aa_char