import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt

class Evaluation():
    def __init__(self,tree,alignment_file_path,start_pos,ref_seq_id):
        self.alignment_file_path = alignment_file_path
        self.start_pos = start_pos
        self.ref_seq_id = ref_seq_id
        self.tree = tree
        self.np_alignments_seq_records = self.process_alignment_data(alignment_file_path)
        self.alignments_length = len(self.np_alignments_seq_records)
    
    def process_alignment_data(self,file_path):
        seq_records = list(SeqIO.parse(file_path, "fasta"))
        np_alignments_seq_records = np.array([str(record.seq) for record in seq_records])
        return np_alignments_seq_records

    def evaluate_path_using_alignments(self,path):
        mutation_matches_list = []
        for seq_id in path:
            if seq_id.strip()==self.ref_seq_id.strip():
                continue # root seq is unmutated so no mutation to evaluate
            mutated_seq_node = self.tree.nodes[seq_id]["object"]
            mutated_seq_mutation = mutated_seq_node.mutation
            percentage_of_mutation_matches = self.evaluate_mutation_only_using_alignments(mutated_seq_mutation)
            mutation_matches_list.append(percentage_of_mutation_matches)
        return mutation_matches_list

    # position-specific identity calculator
    def evaluate_mutation_only_using_alignments(self,seq_mutation):
        mutated_pos = int(seq_mutation[1:-1])-1 # adjust pos to 0-indexing
        new_aa = seq_mutation[-1] 
        # check percentage of matches for the mutated position containing the new amino acid
        relative_mutated_pos = mutated_pos - self.start_pos -1# start pos is already 0-indexed so add 1
        alignment_amino_acids = np.array([seq_record[relative_mutated_pos] for seq_record in self.np_alignments_seq_records])
        num_of_matches = np.sum(alignment_amino_acids==new_aa)
        percentage_of_matches = round((num_of_matches/self.alignments_length)*100,2)
        return percentage_of_matches

    def get_paths_metric_data(self,paths):
        paths_metrics = [self.evaluate_path_using_alignments(path) for path in paths]
        means = [np.mean(path) for path in paths_metrics]
        std_devs = [np.std(path) for path in paths_metrics]
        return means,std_devs

    @staticmethod
    def plot_path_mutation_matches(num_of_paths,means,std_devs):
        # create graphs of path means with error bars
        x_positions = range(num_of_paths)
        path_names = [f"Path {n+1}" for n in x_positions]
        plt.errorbar(x=x_positions,y=means,yerr=std_devs,fmt='o',linewidth=2,capsize=5)
        plt.xticks(x_positions,path_names)
        plt.ylim(0,max(means)+max(std_devs))
        plt.ylabel("Avg. mutation match % across path")
        plt.title("Average Mutuation Match % Across Generated Evolutionary Paths")