import torch
from .protein_sequence import ProteinSequence
from.model_singleton import ModelSingleton

class ProteinLanguageModel:
    def __init__(self):
        self.model = ModelSingleton().get_model()
        self.alphabet = ModelSingleton().get_alphabet()
        self.batch_converter = ModelSingleton().get_batch_converter()

    def get_tokens(self,sequence_id,sequence):
        data = [(sequence_id,sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        return batch_tokens
    
    def get_all_aa_probabilities(self,batch_tokens):
        with torch.no_grad():
            logits_raw = self.model(batch_tokens)["logits"].squeeze(0)
            aa_tokens_len = len(batch_tokens[0,1:-1]) # exclude start and stop tokens
            logits_target = logits_raw[1:(aa_tokens_len+1),4:24] # only extract logits for standard amino acids
        softmax = torch.nn.Softmax(dim=1) # apply by position
        return softmax(logits_target)
    
    def get_sequence_aa_probabilities(self,all_aa_probabilities,batch_tokens):
        token_offset = 4
        aa_tokens = batch_tokens[0,1:-1] # exclude start and stop tokens
        sequence_aa_probabilities = all_aa_probabilities[torch.arange(all_aa_probabilities.size(0)),aa_tokens - token_offset]
        return sequence_aa_probabilities
    
    def get_embeddings(self,batch_tokens):
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=False)
        embeddings = results["representations"][33][0]
        return embeddings
    
    def create_protein_sequence(self,id,sequence,reference_seq=None,parent_seq=None,mutation=None,mutation_score=None):
        batch_tokens = self.get_tokens(id,sequence)
        all_aa_probabilities = self.get_all_aa_probabilities(batch_tokens)
        sequence_aa_probabilities = self.get_sequence_aa_probabilities(all_aa_probabilities,batch_tokens)
        embeddings = self.get_embeddings(batch_tokens)
        return ProteinSequence(
            id=id,
            sequence=sequence,
            reference_seq=reference_seq,
            parent_seq=parent_seq,
            batch_tokens=batch_tokens,
            all_aa_probabilities=all_aa_probabilities,
            sequence_aa_probabilities=sequence_aa_probabilities,
            embeddings=embeddings,
            mutation=mutation,
            mutation_score=mutation_score)
    
