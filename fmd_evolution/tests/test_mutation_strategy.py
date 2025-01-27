import pytest
import esm
from ..mutation_strategy import MutationStrategy, MinLogitPosSub
from ..protein_sequence import ProteinSequence

class TestMinLogitPosSub:
    def setup_method(self,alphabet):
        _,alphabet = alphabet 
        self.strategy = MinLogitPosSub(alphabet)
        self.reference_seq = "ARNDCQEGHILKMFPSTWYV"
        self.protein = ProteinSequence(self.reference_seq)

    def test_initialization(self):
        assert self.strategy.alphabet is not None
        assert self.strategy.start_pos == 138
        assert self.strategy.end_pos == 143
        assert self.strategy.token_offset == 4

    def test_get_next_mutation(self):
        pass