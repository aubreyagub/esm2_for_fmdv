import torch
from model_singleton import ModelSingleton

class ProteinSequence:
    def __init__(self,id,sequence,parent_seq=None,mutation=None):
        self.id = id
        self.parent_seq = parent_seq # previous seq
        self.left_seq = None # child seq with lower score
        self.right_seq = None # child seq with higher score
        # esm data
        self.model = ModelSingleton().get_model()
        self.alphabet = ModelSingleton().get_alphabet()
        self.batch_converter = ModelSingleton().get_batch_converter()
        # sequence data
        self.sequence = sequence
        self.mutation = mutation # to be set using a MutationStrategy 
        self.mutation_score = None # to be set using an EvaluationStrategy
        # plm processed data
        self.batch_tokens = None
        self.set_batch_tokens()
        self.all_logits = None
        self.set_all_logits()
        self.sequence_only_logits = None
        self.set_sequence_only_logits()
        self.embeddings = None
        self.set_embeddings()
    
    def set_parent_seq(self,seq):
        self.parent_seq = seq
        
    def set_left_seq(self,seq):
        self.left_seq = seq
    
    def set_right_seq(self,seq):
        self.right_seq = seq

    def set_mutation(self,mutation): # chosen and set by a MutationStrategy
        self.mutation = mutation
    
    def set_mutation_score(self,score): # chosen and set by an EvaluationStrategy
        self.mutation_score = score

    def set_batch_tokens(self):
        data = [(self.id,self.sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        self.batch_tokens = batch_tokens

    def set_all_logits(self):
        with torch.no_grad():
            logits_raw = self.model(self.batch_tokens)["logits"].squeeze(0)
            aa_tokens_len = len(self.batch_tokens[0,1:-1]) # exclude start and stop tokens)
            logits_target = logits_raw [1:(aa_tokens_len+1),4:24]
        # normalise logits to convert to probabilities 
        lsoftmax = torch.nn.LogSoftmax(dim=1)
        normalised_logits = lsoftmax(logits_target)
        self.all_logits = normalised_logits

    def set_sequence_only_logits(self):
        token_offset = 4
        cleaned_batch_tokens = self.batch_tokens[0,1:-1] # remove start and stop tokens
        sequence_only_logits = self.all_logits[torch.arange(self.all_logits.size(0)),cleaned_batch_tokens - token_offset]
        self.sequence_only_logits = sequence_only_logits
    
    def set_embeddings(self):
        with torch.no_grad():
            embeddings = self.model(self.batch_tokens, repr_layers=[33], return_contacts=True)
        self.embeddings = embeddings

    def generate_mutated_sequence(self,pos,aa_char):
        list_seq = list(self.sequence)
        list_seq[pos] = aa_char
        mutated_seq = "".join(list_seq)
        return mutated_seq

        
    # def evaluate_sequence_mutation(self): 
    #     log_p = torch.log(self.current_seq_logits)
    #     current_sequence_probability = torch.sum(log_p)
    #     # set, then in another func compare change from previous sequence
    #     # other ascpects to evaluate: increase protein fitness + antigenic pressure if applicable + functional + env via past data
    #     return sequence_probability
