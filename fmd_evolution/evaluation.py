import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt

class Evaluation():
    def __init__(self,tree,alignment_file_path,start_pos,ref_seq_id):
        self.alignment_file_path = alignment_file_path
        self.start_pos = start_pos-1 # convert to 0-indexing
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
        relative_mutated_pos = mutated_pos - self.start_pos # adjust start pos to 0-indexing
        print(f"mutated pos: {mutated_pos}, start: {self.start_pos}, relative mutated pos: {relative_mutated_pos}")
        alignment_amino_acids = np.array([seq_record[relative_mutated_pos] for seq_record in self.np_alignments_seq_records])
        print(f"new aa: {new_aa}, alignment: {alignment_amino_acids}")
        num_of_matches = np.sum(alignment_amino_acids==new_aa)
        percentage_of_matches = round((num_of_matches/self.alignments_length),2)
        return percentage_of_matches
    
    # first two mutations are present in same sequence
    def evaluate_final_segment_using_alignments(self,path):
        final_seq_id = path[-1] 
        mutated_seq_node = self.tree.nodes[final_seq_id]["object"]
        mutated_seq_mutation = mutated_seq_node.mutation
        num_of_matches = np.sum(self.np_alignments_seq_records==mutated_seq_mutation)
        percentage_of_matches = round((num_of_matches/self.alignments_length),2)
        return percentage_of_matches

    def evaluate_consecutive_pairs_using_alignments(self,path):
        consective_pairs_matches = []
        for i in range(len(path)-1):
            seq_id = path[i]
            next_seq_id = path[i+1]
            seq_node = self.tree.nodes[seq_id]["object"] 
            next_seq_node = self.tree.nodes[next_seq_id]["object"]
            seq_mutation = seq_node.mutation
            next_seq_mutation = next_seq_node.mutation
            seq_mutated_pos,seq_aa = int(seq_mutation[1:-1])-1,seq_mutation[-1] # adjust pos to 0-indexing
            relative_seq_mutated_pos = seq_mutated_pos - self.start_pos  
            next_seq_mutated_pos,next_seq_aa = int(next_seq_mutation[1:-1])-1,next_seq_mutation[-1] # adjust pos to 0-indexing
            relative_next_seq_mutated_pos = next_seq_mutated_pos - self.start_pos 

            print(f"seq mutated pos: {seq_mutated_pos}, relative seq mutated pos: {relative_seq_mutated_pos}")
            print(f"next mutated pos: {next_seq_mutated_pos}, relative seq mutated pos: {relative_next_seq_mutated_pos}")
            print(f"seq aa: {seq_aa}, next seq aa: {next_seq_aa}")

            num_of_matches = np.sum([
                seq_record[relative_seq_mutated_pos]==seq_aa and seq_record[relative_next_seq_mutated_pos]==next_seq_aa 
                for seq_record in self.np_alignments_seq_records])
            percentage_of_matches = round((num_of_matches/self.alignments_length),2)
            consective_pairs_matches.append(percentage_of_matches)
        return consective_pairs_matches

    def get_paths_metric_data(self,paths_with_scores):
        paths = [path[1:] for _,path in paths_with_scores] # remove root seq from mutations list
        print(f"Paths: {paths}")
        paths_mutation_matches = [self.evaluate_path_using_alignments(path) for path in paths]
        paths_final_segment_matches = [self.evaluate_final_segment_using_alignments(path) for path in paths]
        paths_consecutive_pair_matches = [self.evaluate_consecutive_pairs_using_alignments(path) for path in paths]
        return paths_mutation_matches,paths_final_segment_matches,paths_consecutive_pair_matches
    
    def get_path_mean_and_std_dev(self,paths_metrics):
        means = [np.mean(path) for path in paths_metrics]
        std_devs = [np.std(path) for path in paths_metrics]
        return means,std_devs
    
    @staticmethod
    def errorbar_plot_path_mutation_matches(num_of_paths,means,std_devs):
        plt.figure(figsize=(6,3))
        # create graphs of path means with error bars
        x_positions = range(num_of_paths)
        path_names = [f"{n+1}" for n in x_positions]
        plt.errorbar(x=x_positions,y=means,yerr=std_devs,fmt='o',linewidth=2,capsize=5)
        plt.xlabel("Path Index")
        plt.xticks(x_positions,path_names,ha="right",rotation=45)
        plt.ylim(0,max(means)+max(std_devs))
        plt.ylabel("Avg. mutation match % across path")
        plt.title("Average Mutuation Match % Across Generated Evolutionary Paths")

    @staticmethod
    def box_plot_path_mutation_matches(paths_metrics):
        plt.figure(figsize=(6,3))
        plt.boxplot(paths_metrics, patch_artist = True, notch ='True')
        plt.xlabel("Path Index")
        plt.ylabel("Mutation match %")
        plt.title("Distribution of Mutuation Match % Across Generated Evolutionary Paths")

    @staticmethod
    def bar_plot_path_final_segment_matches(path_metrics):
        plt.figure(figsize=(6,3))
        x_positions = range(len(path_metrics))
        path_names = [f"{n+1}" for n in x_positions] # start from path 1
        plt.bar(path_names,path_metrics)
        plt.xlabel("Path Index")
        plt.ylabel("Final segment match %")
        plt.title("Final Segment Match % Across Generated Evolutionary Paths")

    @staticmethod  
    def box_plot_path_consecutive_pair_matches(paths_metrics):
        plt.figure(figsize=(6,3))
        plt.boxplot(paths_metrics, patch_artist = True, notch ='True')
        plt.xlabel("Path Index")
        plt.ylabel("Consecutive pair match %")
        plt.title("Distribution of Consecutive Pair Match % Across Evolutionary Paths")
