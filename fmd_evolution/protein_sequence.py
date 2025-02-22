import torch
from model_singleton import ModelSingleton

class ProteinSequence:
    def __init__(self,id,sequence,parent_seqs=None,mutation=None):
        self.id = id
        # directed acyclic graph to represent evolutionary paths
        self.parent_seqs = set(parent_seqs) if parent_seqs else set()
        self.child_seqs = set()
        # esm data
        self.model = ModelSingleton().get_model()
        self.alphabet = ModelSingleton().get_alphabet()
        self.batch_converter = ModelSingleton().get_batch_converter()
        # sequence data
        self.sequence = sequence
        self.constrained_seq = None # to be set in MutationStrategy
        self.mutation = mutation # to be set using a MutationStrategy 
        self.mutation_score = None # to be set using an EvaluationStrategy
        # plm processed data
        self.batch_tokens = None
        self.set_batch_tokens()
        self.all_aa_logits = None
        self.set_all_aa_logits()
        self.sequence_aa_logits = None
        self.set_sequence_logits()
        self.embeddings = None
        self.set_embeddings()
    
    def add_parent_seq(self,parent_seq_id):
        self.parent_seqs.add(parent_seq_id)
        
    def add_child_seq(self,child_seq_id):
        self.child_seqs.add(child_seq_id)

    def set_mutation(self,mutation): # chosen and set by a MutationStrategy
        self.mutation = mutation
    
    def set_mutation_score(self,score): # chosen and set by an EvaluationStrategy
        self.mutation_score = score

    def set_batch_tokens(self):
        data = [(self.id,self.sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        self.batch_tokens = batch_tokens

    def set_all_aa_logits(self):
        with torch.no_grad():
            logits_raw = self.model(self.batch_tokens)["logits"].squeeze(0)
            aa_tokens_len = len(self.batch_tokens[0,1:-1]) # exclude start and stop tokens
            logits_target = logits_raw [1:(aa_tokens_len+1),4:24]
        # normalise logits to convert to probabilities 
        softmax = torch.nn.Softmax(dim=1)
        self.all_aa_logits = softmax(logits_target)

    def set_sequence_logits(self):
        token_offset = 4 # index of amino acids in alphabet begins at 4
        aa_tokens = self.batch_tokens[0,1:-1]  # exclude start and stop tokens
        sequence_aa_logits = self.all_aa_logits[torch.arange(self.all_aa_logits.size(0)),aa_tokens - token_offset]
        self.sequence_aa_logits = sequence_aa_logits
    
    def set_embeddings(self):
        with torch.no_grad():
            results = self.model(self.batch_tokens, repr_layers=[33], return_contacts=True)
        self.embeddings = results["representations"][33][0]

    def generate_mutated_sequence(self,pos,aa_char):
        list_seq = list(self.sequence)
        list_seq[pos] = aa_char
        mutated_seq = "".join(list_seq)
        return mutated_seq

        
