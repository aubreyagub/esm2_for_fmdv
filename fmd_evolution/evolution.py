from protein_sequence import ProteinSequence
from mutation_strategy import MutationStrategy, MinLogitPosSub
from evaluation_strategy import EvaluationStrategy
from model_singleton import ModelSingleton

class Evolution:
    def __init__(self,root_sequence,mutation_strategy,evaluation_strategy,max_generations): #evaluation_strategy
        self.root_sequence = root_sequence
        self.mutation_strategy = mutation_strategy
        self.evaluation_strategy = evaluation_strategy
        self.max_generations = max_generations
        self.dag = {} # id = unique sequence id, value = sequence object
        self.dag[root_sequence.id] = root_sequence
        # esm data
        self.model = ModelSingleton().get_model()
        self.alphabet = ModelSingleton().get_alphabet()
        self.batch_converter = ModelSingleton().get_batch_converter()
        
        # population to select from + be able to go back and choose another path 
        # track sequence_probability over time and ensure this doesn't go down too much (fitness landscape) - sampling then search & constrain

    def evolve_sequence(self,current_seq=None,generation=0):
        if current_seq is None:
            current_seq = self.root_sequence
        if generation<self.max_generations: # stop evovling when max generations reached
            all_logits = current_seq.all_logits
            sequence_only_logits = current_seq.sequence_only_logits 
            sequence = current_seq.sequence
            # process potential mutations
            mutations = self.mutation_strategy.get_next_mutations(sequence,all_logits,sequence_only_logits)
            for pos,aa_char in mutations:
                mutated_sequence = current_seq.generate_mutated_sequence(pos,aa_char) 
                mutation = f"{pos}{aa_char}"
                mutated_seq = self.add_new_seq(id=mutation,mutated_sequence=mutated_sequence,parent_seq=current_seq,mutation=mutation) # create new node for mutated seq 
                print(f"Gen {generation}: current seq muatted at pos {pos} with aa {aa_char}, parent seq has mutation {current_seq.mutation}")
                # check if mutated sequence is probable and functional
                should_continue_mutating = self.evaluation_strategy.should_continue_mutating(mutated_seq)
                print(f"Mutation Score: {mutated_seq.mutation_score}")
                if should_continue_mutating:
                    self.evolve_sequence(current_seq=mutated_seq,generation=generation+1)

            # pos,aa_char = self.mutation_strategy.get_next_mutation(sequence,all_logits,sequence_only_logits) # refactor to send whole object instead + add inside evaluation strategy
             

    def add_new_seq(self,id,mutated_sequence,parent_seq,mutation):
        # mutation resulted in a new sequence 
        if mutated_sequence not in self.dag:
            mutated_seq = ProteinSequence(id=id,sequence=mutated_sequence,parent_seqs=[parent_seq],mutation=mutation) # create new node
            self.dag[mutated_sequence] = mutated_seq
            parent_seq.add_child_seq(mutated_seq)
        # mutation resulted in an existing sequence
        else:
            mutated_seq = self.dag[mutated_sequence] # retrieve existing node 
            mutated_seq.add_parent_seq(parent_seq)
            parent_seq.add_child_seq(mutated_seq)
        return mutated_seq
        
    

# compare logits with next step and see the change
    

    