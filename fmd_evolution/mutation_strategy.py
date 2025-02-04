from abc import ABC, abstractmethod
from protein_sequence import ProteinSequence
from model_singleton import ModelSingleton
import numpy as np

class MutationStrategy(ABC):
    @abstractmethod
    def __init__(self,mutations_per_seq=2):
        self.mutations_per_seq = mutations_per_seq
        self.alphabet = ModelSingleton().get_alphabet()
        self.start_pos = 138
        self.end_pos = 143
        self.token_offset=4 # index of aa in alphabet begins at 4
    
    @abstractmethod
    def get_next_mutations(self,current_seq,all_logits,current_seq_logits,start_pos=138,end_pos=143):
        pass

    def index_to_char(self,aa):
        return self.alphabet.all_toks[aa+self.token_offset] 

    def get_aa_char(self,current_aa,new_aa_index):
        aa_char = self.index_to_char(new_aa_index)
        if current_aa!=aa_char:
            return aa_char
        else:
            return None # new aa == current aa
            
    def get_top_n_mutations(self, scores):
        return np.argsort(scores)[-(self.mutations_per_seq):][::-1] # top 2 in case new aa == current aa

    def get_new_amino_acid(self,current_seq,pos):
        current_aa = list(current_seq)[pos]
        aa_char = self.get_aa_char(current_aa,pos)
        if aa_char:
            return aa_char
        return None # no new aa
        
# Mutate through substitution the position with the minimum logit 
class MinLogitPosSub(MutationStrategy):
    def __init__(self,mutations_per_seq=2):
        super().__init__(mutations_per_seq)

    def get_next_mutations(self,current_seq,all_logits,current_seq_logits,start_pos=138,end_pos=143):
        poss_of_interest_current_seq_logits = current_seq_logits[self.start_pos:self.end_pos+1]
        min_logit_pos_relative = np.argmin(poss_of_interest_current_seq_logits)
        min_logit_pos = (self.start_pos+min_logit_pos_relative).item()

        poss_of_interest_logits = all_logits[self.start_pos:self.end_pos+1,:]
        min_logit_pos_logits = poss_of_interest_logits[min_logit_pos_relative]

        # generate mutations
        potential_mutations = self.get_top_n_mutations(min_logit_pos_logits.numpy())
        print(f"Potential mutations: {potential_mutations}")
        mutations = []
        for p_mut in potential_mutations:
            aa_char = self.get_new_amino_acid(current_seq,p_mut)  
            if aa_char:
                mutations.append((min_logit_pos,aa_char))
        return mutations

# Use logic of MinLogitPosSub, weighted/penalty using blosum scores
class BlosumWeightedSub(MutationStrategy):
    def __init__(self,blosum_matrix,multiplier=None,mutations_per_seq=2,):
        super().__init__(mutations_per_seq)
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

    def get_next_mutations(self,current_seq,all_logits,current_seq_logits,start_pos=138,end_pos=143):
        poss_of_interest_current_seq_logits = current_seq_logits[self.start_pos:self.end_pos+1]
        min_logit_pos_relative = np.argmin(poss_of_interest_current_seq_logits)
        min_logit_pos = (self.start_pos+min_logit_pos_relative).item()

        poss_of_interest_logits = all_logits[self.start_pos:self.end_pos+1,:]
        min_logit_pos_logits = poss_of_interest_logits[min_logit_pos_relative]

        ########### TO BE UPDATED TO WORK WITH MULTIPLE POTENTIAL MUTATIONS
        # get weighted scores
        current_aa = list(current_seq)[min_logit_pos]
        blosum_scores = self.get_blosum_scores(current_aa)
        weighted_scores = self.weight_logits_with_blosum(blosum_scores,min_logit_pos_logits)
        
        # select new aa
        candidates = self.get_top_n_mutations(np.array(weighted_scores))
        aa_char = self.get_new_amino_acid(current_seq,min_logit_pos)
        
        return min_logit_pos,aa_char
    
