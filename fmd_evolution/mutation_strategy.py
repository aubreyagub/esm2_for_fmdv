from abc import ABC, abstractmethod
from .protein_sequence import ProteinSequence
from .model_singleton import ModelSingleton
import numpy as np
import torch
from . import SEED,rng

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
    
    def get_average_min_logit_pos(self,all_aa_probabilities): 
        pos_mean_logits = torch.mean(all_aa_probabilities,dim=1) # use average over direct aa logit for robustness 
        relevant_segment_logits = pos_mean_logits[self.start_pos:self.end_pos+1]
        relative_min_logit_position = torch.argmin(relevant_segment_logits).item()
        absolute_min_logit_position = self.start_pos+relative_min_logit_position
        return relative_min_logit_position,absolute_min_logit_position
    
    # def get_min_logit_pos(self,sequence_aa_probabilities):
    #     relevant_segment_logits = sequence_aa_probabilities[self.start_pos:self.end_pos+1]
    #     relative_min_logit_position = np.argmin(relevant_segment_logits)
    #     absolute_min_logit_position = (self.start_pos+relative_min_logit_position).item()
    #     return relative_min_logit_position,absolute_min_logit_position
    
    def get_position_logit_values(self,relative_position,all_aa_probabilities):
        relevant_segment_amino_acids_logits = all_aa_probabilities[self.start_pos:self.end_pos+1,:]
        return relevant_segment_amino_acids_logits[relative_position]
    
    def get_min_logit_pos_and_values(self,sequence_aa_probabilities,all_aa_probabilities):
        relative_min_logit_position,absolute_min_logit_position = self.get_average_min_logit_pos(all_aa_probabilities) ##################
        aa_logits_at_min_logit_position = self.get_position_logit_values(relative_min_logit_position,all_aa_probabilities)

        return absolute_min_logit_position,aa_logits_at_min_logit_position
    
    def get_absolute_aa_positions(self,aa_positions):
        return aa_positions + self.token_offset
    
        
    def validate_potential_mutations(self,current_seq,min_logit_pos,potential_aa_positions):
        current_aa_chars = list(current_seq)[min_logit_pos]

        potential_aa_positions = np.array(potential_aa_positions)
        adjusted_potential_aa_positions = potential_aa_positions+self.token_offset # match aa positions to esm alphabet aa indices

        alphabet_tokens = np.array(self.alphabet.all_toks)
        new_aa_chars = alphabet_tokens[adjusted_potential_aa_positions] # convert positions to aa chars

        remove_redundant_mutation_mask = new_aa_chars!=current_aa_chars
        cleaned_new_aa_chars = new_aa_chars[remove_redundant_mutation_mask]    

        min_logit_as_array = [min_logit_pos]*len(cleaned_new_aa_chars) 
        mutations = list(zip(current_aa_chars,min_logit_as_array,cleaned_new_aa_chars)) 
        return mutations[:self.mutations_per_seq] # ensure only the specified number of mutations are returned

# Mutate through substitution the position with the minimum logit 
class MinLogitPosSub(MutationStrategy):
    def __init__(self,mutations_per_seq=20,start_pos=138,end_pos=143):
        super().__init__(mutations_per_seq,start_pos,end_pos)

    def get_next_mutations(self,current_seq):
        current_seq.constrained_seq = current_seq.sequence[self.start_pos:self.end_pos+1] # set to constrained to segment of interest

        sequence = current_seq.sequence
        all_aa_probabilities = current_seq.all_aa_probabilities
        sequence_aa_probabilities = current_seq.sequence_aa_probabilities 

        # get position with minimum logit score and its logit scores
        min_logit_pos,aa_logits_at_min_logit_position = self.get_min_logit_pos_and_values(sequence_aa_probabilities,all_aa_probabilities)

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
        all_aa_probabilities = current_seq.all_aa_probabilities
        sequence_aa_probabilities = current_seq.sequence_aa_probabilities 
        min_logit_pos,aa_logits_at_min_logit_position = self.get_min_logit_pos_and_values(sequence_aa_probabilities,all_aa_probabilities)
        
        # get new weighted scores for amino acids in given pos
        current_aa = list(sequence)[min_logit_pos]
        blosum_scores = self.get_blosum_scores(current_aa)
        weighted_scores = self.weight_logits_with_blosum(blosum_scores,aa_logits_at_min_logit_position)

        # generate mutations
        potential_aa_positions = self.get_top_n_mutations(np.array(weighted_scores))
        mutations = self.validate_potential_mutations(sequence,min_logit_pos,potential_aa_positions)
        return mutations

class MetropolisHastings(MutationStrategy):
    def __init__(self,iterations=10,mutations_per_seq=3,start_pos=138,end_pos=143):
        super().__init__(mutations_per_seq,start_pos,end_pos)
        self.iterations = iterations

    def is_new_mutation(self,current_seq,absolute_pos,potential_aa_pos):
        current_aa_char = list(current_seq)[absolute_pos]

        adjusted_potential_aa_pos = potential_aa_pos+self.token_offset # match aa positions to esm alphabet aa indices

        alphabet_tokens = np.array(self.alphabet.all_toks)
        potential_aa_char = alphabet_tokens[adjusted_potential_aa_pos] # convert positions to aa chars

        if current_aa_char!=potential_aa_char:
            return (current_aa_char,absolute_pos,potential_aa_char)
        print(f"Same mutation at {absolute_pos}: {current_aa_char} → {potential_aa_char}")
        return None

    def get_probability_distro(self,probabilities):
        probabilities = probabilities.numpy()
        recomputed_softmax = np.exp(probabilities - np.max(probabilities)) # stabilise softmax
        relevant_probs = recomputed_softmax/recomputed_softmax.sum() # normalise to get probabilities and ensure sum to 1
        return  relevant_probs 
    
    def get_previously_mutated_positions(self,parent_constrained_seq,current_constrained_seq):
        previously_mutated_positions = []
        for i in range(len(parent_constrained_seq)):
            if parent_constrained_seq[i]!=current_constrained_seq[i]: # mutation detected
                absolute_pos = self.start_pos+i
                previously_mutated_positions.append(absolute_pos)
        return previously_mutated_positions
    
    def mask_previously_mutated_positions(self,probabilities,absolute_pos,previously_mutated_positions):
        if absolute_pos in previously_mutated_positions:
            return np.zeros_like(probabilities) # mask out p for previously mutated position
        return probabilities # original p remains for unmutated position
    
    def calculate_acceptance_ratio(self,current_val,new_val,probability_distro):
        current_p = probability_distro[current_val]
        new_p = probability_distro[new_val]
        acceptance_ratio = min(1,new_p/current_p)
        return acceptance_ratio

    def should_accept(self,acceptance_ratio):
        random_number = rng.uniform(0,1)
        if random_number<=acceptance_ratio:
            return True 
        else:
            return False
    
    def sample_a_mutation(self,probability_distro):
        relative_pos = rng.choice(len(probability_distro),p=probability_distro)
        return relative_pos
    
    def get_mutation_via_mh(self,possible_mutations,probability_distro):
        current_index = self.sample_a_mutation(probability_distro)
        current_mutation = possible_mutations[current_index]
        for _ in range(self.iterations):
            new_index = self.sample_a_mutation(probability_distro)
            acceptance_ratio = self.calculate_acceptance_ratio(current_index,new_index,probability_distro)
            if self.should_accept(acceptance_ratio):
                current_index = new_index
        return current_mutation


    def get_mutations_and_probability_distro(self,current_seq):
        print(f"Constrained seq = {current_seq.sequence[self.start_pos:self.end_pos+1]}")
        current_seq.constrained_seq = current_seq.sequence[self.start_pos:self.end_pos+1] # set to constrained to segment of interest
        parent_constrained_seq = None

        previously_mutated_positions = []
        if current_seq.parent_obj: # check that it is not the reference sequence
            current_constrained_seq = current_seq.constrained_seq
            parent_constrained_seq = current_seq.parent_obj.constrained_seq
            previously_mutated_positions = self.get_previously_mutated_positions(parent_constrained_seq,current_constrained_seq)

        relative_aa_probabilities = current_seq.all_aa_probabilities[self.start_pos:self.end_pos+1] # get probabilities for constrained segment
        possible_mutations = []
        possible_mutations_probabilities = []

        print(f"Previously mutated positions = {previously_mutated_positions}")
    
        for relative_pos, absolute_pos in enumerate(range(self.start_pos,self.end_pos+1)):
            aa_probabilities = relative_aa_probabilities[relative_pos].numpy()
            masked_aa_probabilities = self.mask_previously_mutated_positions(aa_probabilities,absolute_pos,previously_mutated_positions)

            for aa_index,probability in enumerate(masked_aa_probabilities):
                if probability!=0: # omit zero probabilities
                    new_mutation = self.is_new_mutation(current_seq.sequence,absolute_pos,aa_index)
                    if new_mutation: # exclude mutation if amino acid is same as current amino acid 
                        possible_mutations.append(new_mutation)
                        possible_mutations_probabilities.append(probability)
                 
        possible_mutations_probabilities_np = np.array(possible_mutations_probabilities)
        probability_distro = possible_mutations_probabilities_np/possible_mutations_probabilities_np.sum() # normalise to get probabilities and ensure sum to 1

        return possible_mutations,probability_distro

    def get_next_mutations(self,current_seq):
        possible_mutations,probability_distro = self.get_mutations_and_probability_distro(current_seq)
        if len(possible_mutations)==0 or len(probability_distro)==0: 
            return [] # no mutations found
        
        print(f"Len possible_mutations = {len(possible_mutations)}")
        print(f"Len probability distro = {len(probability_distro)}")

        mutations_unvalidated = []
        for _ in range(self.mutations_per_seq):
            mutation = self.get_mutation_via_mh(possible_mutations,probability_distro)
            mutations_unvalidated.append(mutation)

        mutations = list(dict.fromkeys(mutations_unvalidated)) # remove duplicates while maintaining order
        return mutations
