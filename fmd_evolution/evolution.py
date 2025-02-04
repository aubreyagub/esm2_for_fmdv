from protein_sequence import ProteinSequence
from mutation_strategy import MutationStrategy, MinLogitPosSub
from evaluation_strategy import EvaluationStrategy
from model_singleton import ModelSingleton
import networkx as nx
import matplotlib.pyplot as plt

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
            potential_mutations = self.mutation_strategy.get_next_mutations(sequence,all_logits,sequence_only_logits)
            for pos,aa_char in potential_mutations:
                mutated_sequence = current_seq.generate_mutated_sequence(pos,aa_char) 
                mutation = f"{pos}{aa_char}"
                mutated_seq = self.add_new_seq(id=mutation,mutated_sequence=mutated_sequence,parent_seq=current_seq,mutation=mutation) # create new node for mutated seq 
                # print(f"Gen {generation}: current seq muatted at pos {pos} with aa {aa_char}, parent seq has mutation {current_seq.mutation}")
                # check if mutated sequence is probable and functional
                should_continue_mutating = self.evaluation_strategy.should_continue_mutating(mutated_seq)
                # print(f"Mutation Score: {mutated_seq.mutation_score}")
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
        
    def visualise_evolution_dag(self):
        graph = nx.DiGraph()
        for seq,seq_obj in self.dag.items():
            graph.add_node(seq_obj.id)
            for child in seq_obj.child_seqs:
                graph.add_edge(seq_obj.id,child.id,weight=child.mutation_score)

        # weight the edges
        edge_weights = nx.get_edge_attributes(graph,"weight")
        max_weight=max(edge_weights.values())
        min_weight=min(edge_weights.values())
        norm_weights = {
            k: 0.1 + 0.9 * (v - min_weight) / (max_weight - min_weight) if max_weight != min_weight else 1
            for k, v in edge_weights.items()
        }

        plt.figure(figsize=(12,7))
        pos = nx.spring_layout(graph, seed=42)
        nx.draw(graph, pos, with_labels=True,node_color="lightgreen",edge_color="grey",node_size=2500,font_size=8)

        labels = nx.get_node_attributes(graph,"label")
        nx.draw_networkx_labels(graph,pos,labels=labels,font_size=10,font_color="darkgreen")

        edge_widths = [norm_weights.get(edge, 1) * 2 for edge in graph.edges()]
        nx.draw_networkx_edges(graph, pos, width=edge_widths, edge_color="black")

        plt.title("Evolutionary DAG of FMDVP1")
        plt.show()
        
# compare logits with next step and see the change
    

    