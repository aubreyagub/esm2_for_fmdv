import pytest
from ..protein_sequence import ProteinSequence

# run command: pytest tests/test_protein_sequence.py
class TestProteinSequence:
    def setup_method(self):
        self.reference_seq = "ARNDCQEGHILKMFPSTWYV"
        self.mutation_one = "A1C"
        self.mutation_two = "V20Y"
        self.protein = ProteinSequence(self.reference_seq)
        self.seq_len = len(self.reference_seq)

    def test_initialization(self):
        assert self.protein.reference_seq == self.reference_seq 
        assert self.protein.current_seq == self.reference_seq
        assert self.protein.mutations == []

    def test_get_reference_seq(self):
        assert self.protein.get_reference_seq() == self.reference_seq 

    def test_get_current_seq(self):
        assert self.protein.get_current_seq() == self.reference_seq 

    def test_add_mutation(self):
        self.protein.add_mutation(self.mutation_one)
        self.protein.add_mutation(self.mutation_two)
        assert len(self.protein.mutations) == 2
        assert self.protein.mutations[0] == self.mutation_one
        assert self.protein.mutations[1] == self.mutation_two
        
    def test_get_mutations(self):
        self.protein.mutations = [self.mutation_one,self.mutation_two]
        assert self.protein.get_mutations() == [self.mutation_one,self.mutation_two]

    def test_get_batch_tokens(self):
        tokens = self.protein.get_tokens()
        assert len(tokens) == self.seq_len
        
    def test_get_aa_logits(self):
        all_logits = self.protein.get_all_logits()
        num_of_aa = 20
        assert all_logits.shape == (self.seq_len,num_of_aa)
        
    def test_get_current_seq_logits(self):
        current_seq_logits = self.protein.get_current_seq_logits()
        assert current_seq_logits.shape == (self.seq_len)