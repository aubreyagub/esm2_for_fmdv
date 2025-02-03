from protein_sequence import ProteinSequence
from mutation_strategy import MutationStrategy, MinLogitPosSub
from model_singleton import ModelSingleton

class Evolution:
    def __init__(self,root_sequence,mutation_strategy,max_generations): #evaluation_strategy
        self.root_sequence = root_sequence
        self.mutation_strategy = mutation_strategy
        #self.evaluation_strategy = evaluation_strategy
        self.max_generations = max_generations
        self.model = ModelSingleton().get_model()
        self.alphabet = ModelSingleton().get_alphabet()
        self.batch_converter = ModelSingleton().get_batch_converter()
        # population to select from + be able to go back and choose another path 
        # track sequence_probability over time and ensure this doesn't go down too much (fitness landscape) - sampling then search & constrain

    def evolve_sequence(self,current_seq=None,generation=0):
        if current_seq is None:
            current_seq = self.root_sequence
        if generation<self.max_generations:
            all_logits = current_seq.all_logits
            sequence_only_logits = current_seq.sequence_only_logits 
            sequence = current_seq.sequence
            pos,aa_char = self.mutation_strategy.get_next_mutation(sequence,all_logits,sequence_only_logits) # refactor to send whole object instead + add inside evaluation strategy
            mutated_sequence = current_seq.generate_mutated_sequence(pos,aa_char) 
            mutation = f"{pos}{aa_char}"
            mutated_seq = self.add_new_seq(id=mutation,mutated_sequence=mutated_sequence,parent_seq=current_seq,mutation=mutation) # create new node for mutated seq 
            print(f"Position mutated = {pos} with amino acid {aa_char}")
            self.evolve_sequence(current_seq=mutated_seq,generation=generation+1) 

    def add_new_seq(self,id,mutated_sequence,parent_seq,mutation):
        mutated_seq = ProteinSequence(sequence=mutated_sequence,parent_seq=parent_seq,id=id,mutation=mutation)
        mutated_seq.set_parent_seq(parent_seq)
        parent_seq.set_right_seq(mutated_seq)
        #parent_seq.set_left_seq(mutated_seq) ?
        return mutated_seq
        
    

# compare logits with next step and see the change
    

    