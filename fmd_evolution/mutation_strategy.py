from abc import ABC, abstractmethod
from protein_sequence import ProteinSequence
from model_singleton import ModelSingleton
import numpy as np
import torch

class MutationStrategy(ABC):
    @abstractmethod
    def __init__(self,mutations_per_seq=20,start_pos=138,end_pos=143):
        self.mutations_per_seq = mutations_per_seq # 20 means consider all possible amino acids
        self.alphabet = ModelSingleton().get_alphabet()
        self.start_pos = start_pos-1 # amino acid positions are 1-indexed
        self.end_pos = end_pos-1 # amino acid positions are 1-indexed
        self.token_offset=4 # index of aa in esm alphabet begins at 4
    
    def index_to_char(self,aa_pos):
        return self.alphabet.all_toks[aa_pos+self.token_offset] 

    def get_top_n_mutations(self, scores):
        return np.argsort(scores)[-(self.mutations_per_seq+1):][::-1] # one extra in case current_aa is a potential mutation

    def get_new_amino_acid(self,aa_pos):
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
    
    def get_absolute_aa_positions(self,aa_positions):
        return aa_positions + self.token_offset
        
    def validate_potential_mutations(self,current_seq,min_logit_pos,potential_aa_positions):
        current_aa = list(current_seq)[min_logit_pos]

        potential_aa_positions = np.array(potential_aa_positions)
        adjusted_potential_aa_positions = potential_aa_positions+self.token_offset # match aa positions to esm alphabet aa indices

        alphabet_tokens = np.array(self.alphabet.all_toks)
        aa_chars = alphabet_tokens[adjusted_potential_aa_positions] # convert positions to aa chars

        remove_redundant_mutation_mask = aa_chars!=current_aa 
        cleaned_aa_chars = aa_chars[remove_redundant_mutation_mask]    

        min_logit_as_array = [min_logit_pos]*len(cleaned_aa_chars)
        mutations = list(zip(min_logit_as_array,cleaned_aa_chars))
        return mutations[:self.mutations_per_seq] # ensure only the specified number of mutations are returned

# Mutate through substitution the position with the minimum logit 
class MinLogitPosSub(MutationStrategy):
    def __init__(self,mutations_per_seq=20,start_pos=138,end_pos=143):
        super().__init__(mutations_per_seq,start_pos,end_pos)

    def get_next_mutations(self,current_seq):
        current_seq.constrained_seq = current_seq.sequence[self.start_pos:self.end_pos+1] # set to constrained to segment of interest

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
        current_seq.constrained_seq = current_seq.sequence[self.start_pos:self.end_pos+1] # set to constrained to segment of interest

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

class MetropolisHastings(MutationStrategy):
    def __init__(self,iterations=10,positions_per_seq=5,mutations_per_seq=2,start_pos=138,end_pos=143):
        super().__init__(mutations_per_seq,start_pos,end_pos)
        self.iterations = iterations
        self.positions_per_seq = positions_per_seq

    def get_probability_distro_of_amino_acids(self,aa_logits_at_mh_position):
        aa_logits_at_mh_position = aa_logits_at_mh_position.numpy()
        recomputed_softmax = np.exp(aa_logits_at_mh_position - np.max(aa_logits_at_mh_position))
        relevant_probs = recomputed_softmax/recomputed_softmax.sum() # normalise to get probabilities and ensure sum to 1
        return  relevant_probs 

    def get_probability_distro_of_positions(self,sequence_aa_logits):
        relevant_segment_logits = (sequence_aa_logits[self.start_pos:self.end_pos+1]).numpy()
        recomputed_softmax = np.exp(relevant_segment_logits - np.max(relevant_segment_logits))
        relevant_probs = recomputed_softmax/recomputed_softmax.sum() # normalise to get probabilities and ensure sum to 1
        return  relevant_probs 
        
    def sample_an_amino_acid(self,probability_distro):
        relative_aa_index = np.random.choice(len(probability_distro),p=probability_distro)
        return relative_aa_index # relative since absolute index is calculated at index to char conversion

    def sample_a_position(self,probability_distro):
        relative_pos = np.random.choice(len(probability_distro),p=probability_distro)
        return self.start_pos+relative_pos # absolute position
    
    def calculate_acceptance_ratio(self,current_val,new_val,probability_distro):
        current_p = probability_distro[current_val]
        new_p = probability_distro[new_val]
        acceptance_ratio = min(1,new_p/current_p)
        return acceptance_ratio

    def should_accept_pos(self,acceptance_ratio):
        random_number = np.random.uniform(0,1)
        if random_number<=acceptance_ratio:
            return True 
        else:
            return False

    def get_amino_acid_via_mh(self,aa_logits_at_mh_position):
        probability_distro = self.get_probability_distro_of_amino_acids(aa_logits_at_mh_position)
        current_aa = self.sample_an_amino_acid(probability_distro)
        for _ in range(self.iterations):
            new_aa = self.sample_an_amino_acid(probability_distro)
            acceptance_ratio = self.calculate_acceptance_ratio(current_aa,new_aa,probability_distro)
            if self.should_accept_pos(acceptance_ratio):
                current_aa = new_aa
        return current_aa

    def get_position_via_mh(self,sequence_aa_logits):
        probability_distro = self.get_probability_distro_of_positions(sequence_aa_logits)
        current_pos = self.sample_a_position(probability_distro) # initialise to a random pos
        for _ in range(self.iterations):
            new_pos = self.sample_a_position(probability_distro)
            relative_current_pos = current_pos-self.start_pos
            relative_new_pos = new_pos-self.start_pos
            acceptance_ratio = self.calculate_acceptance_ratio(relative_current_pos,relative_new_pos,probability_distro)
            if self.should_accept_pos(acceptance_ratio):
                current_pos = new_pos
        return current_pos

    def get_next_mutations(self,current_seq):
        current_seq.constrained_seq = current_seq.sequence[self.start_pos:self.end_pos+1] # set to constrained to segment of interest
        
        sequence = current_seq.sequence
        all_aa_logits = current_seq.all_aa_logits
        sequence_aa_logits = current_seq.sequence_aa_logits 

        mutations = []
        for _ in range(self.positions_per_seq):
            mh_absolute_pos = self.get_position_via_mh(sequence_aa_logits)
            mh_relative_pos = mh_absolute_pos-self.start_pos

            aa_logits_at_mh_position = self.get_position_logit_values(mh_relative_pos,all_aa_logits)
            mh_potential_aa_positions = np.array([self.get_amino_acid_via_mh(aa_logits_at_mh_position) for _ in range(self.mutations_per_seq)]) 

            validated_mutations = self.validate_potential_mutations(sequence,mh_absolute_pos+1,mh_potential_aa_positions) # adjust pos for 1-indexing
            mutations.extend(validated_mutations)

        mutations = list(set(mutations)) # remove duplicates
        return mutations
