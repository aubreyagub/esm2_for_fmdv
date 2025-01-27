from protein_sequence import ProteinSequence
from mutation_strategy import MutationStrategy, MinLogitPosSub

class Evolution:
    def __init__(self,protein_sequence,mutation_strategy,model,alphabet,batch_converter):
        self.protein_sequence = protein_sequence
        self.mutation_strategy = mutation_strategy
        self.model = model
        self.alphabet = alphabet
        self.batch_converter = batch_converter

    def evolve_sequence(self,steps):
        for i in range(steps):
            all_logits,current_seq_logits = self.protein_sequence.get_logits(self.model,self.alphabet,self.batch_converter)
            current_sequence = self.protein_sequence.get_current_seq()
            pos,aa_char = self.mutation_strategy.get_next_mutation(current_sequence,all_logits,current_seq_logits)
            self.protein_sequence.mutate_seq(pos,aa_char) 
            print(f"Position mutated = {pos}")

    

    