import torch

class ProteinSequence:
    def __init__(self,id="unknown",sequence="",parent_seq=None,parent_obj=None,child_seqs=[],constrained_seq=None,mutation=None,mutation_score=None,
                 batch_tokens=None,all_aa_probabilities=None,sequence_aa_probabilities=None,embeddings=None, is_reverse=False, is_flip_flop=False):
        self.id = id
        self.sequence = sequence # full amino acid sequence
        self.parent_seq = parent_seq 
        self.parent_obj = parent_obj
        self.child_seqs = child_seqs
        self.constrained_seq = None # to be set in MutationStrategy
        self.mutation = mutation # to be set using a MutationStrategy 
        self.mutation_score = None # to be set using an EvaluationStrategy
        self.probability = None # to be set using a EvaluationStrategy
        self.embedding_distance = None # to be set using a EvaluationStrategy
        # plm processed data
        self.batch_tokens = batch_tokens
        self.all_aa_probabilities = all_aa_probabilities
        self.sequence_aa_probabilities = sequence_aa_probabilities
        self.embeddings = embeddings
        # termination indicators
        self.is_reverse = is_reverse
        self.is_flip_flop = is_flip_flop
    
    def set_parent_seq(self,parent_seq_id):
        self.parent_seq = parent_seq_id

    def set_parent_obj(self,parent_obj):
        self.parent_obj = parent_obj
        
    def add_child_seq(self,child_seq_id):
        self.child_seqs.append(child_seq_id)

    def set_mutation(self,mutation): # chosen and set by a MutationStrategy
        self.mutation = mutation
    
    def set_mutation_score(self,score): # chosen and set by an EvaluationStrategy
        self.mutation_score = score

    def generate_mutated_sequence(self,pos,aa_char):
        list_seq = list(self.sequence)
        list_seq[pos-1] = aa_char # convert mutation to 0-indexing
        mutated_seq = "".join(list_seq)
        return mutated_seq

        
