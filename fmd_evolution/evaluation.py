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
    
    def process_alignment_data_to_graph(self):
        seq_records = list(SeqIO.parse(self.alignment_file_path, "fasta"))
        np_alignments_seq_records = np.array([list(str(record.seq)) for record in seq_records])
        return np_alignments_seq_records
    
    def get_path_as_mutations(self,path):
        path_as_mutations = []
        for seq_id in path:
            mutated_seq_node = self.tree.nodes[seq_id]["object"]
            mutated_seq_mutation = mutated_seq_node.mutation
            path_as_mutations.append(mutated_seq_mutation)
        return path_as_mutations

    def evaluate_path_mutations(self,path):
        mutation_matches_list = []
        for seq_id in path:
            mutated_seq_node = self.tree.nodes[seq_id]["object"]
            mutated_seq_mutation = mutated_seq_node.mutation
            percentage_of_mutation_matches = self.evaluate_mutation_only(mutated_seq_mutation)
            mutation_matches_list.append(percentage_of_mutation_matches)
        return mutation_matches_list

    # position-specific identity calculator
    def evaluate_mutation_only(self,seq_mutation):
        mutated_pos = int(seq_mutation[1:-1])-1 # adjust pos to 0-indexing
        new_aa = seq_mutation[-1] 
        # check percentage of matches for the mutated position containing the new amino acid
        relative_mutated_pos = mutated_pos - self.start_pos # adjust start pos to 0-indexing
        alignment_amino_acids = np.array([seq_record[relative_mutated_pos] for seq_record in self.np_alignments_seq_records])
        num_of_matches = np.sum(alignment_amino_acids==new_aa)
        percentage_of_matches = round((num_of_matches/self.alignments_length),3)
        return percentage_of_matches
    
    def evaluate_final_segment(self,path):
        final_seq_id = path[-1] 
        mutated_seq_node = self.tree.nodes[final_seq_id]["object"]
        mutated_seq_mutation = mutated_seq_node.mutation
        num_of_matches = np.sum(self.np_alignments_seq_records==mutated_seq_mutation)
        percentage_of_matches = round((num_of_matches/self.alignments_length),3)
        return percentage_of_matches

    def evaluate_consecutive_pairs(self,path):
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

            num_of_matches = np.sum([
                seq_record[relative_seq_mutated_pos]==seq_aa and seq_record[relative_next_seq_mutated_pos]==next_seq_aa 
                for seq_record in self.np_alignments_seq_records])
            percentage_of_matches = round((num_of_matches/self.alignments_length),3)
            consective_pairs_matches.append(percentage_of_matches)
        return consective_pairs_matches
    
    def get_average_proportion_of_paths_with_nonzero_values(self,paths_metrics):
        proportions = []
        for path_metrics in paths_metrics:
            path_len = len(path_metrics)
            if path_len==0:
                continue
            mutations_with_nonzero_value = 0
            for mf in path_metrics:
                if mf != 0.0:
                    mutations_with_nonzero_value += 1
            proportion = round(mutations_with_nonzero_value/path_len,2)
            proportions.append(proportion)

        average_proportion = round(sum(proportions)/len(proportions),2)
        return average_proportion


    def get_paths_metric_data(self,paths_with_scores):
        paths = [path[1:] for _,path in paths_with_scores] # remove root seq from mutations list
        print(f"Paths: {paths}")
        paths_mutation_matches = [self.evaluate_path_mutations(path) for path in paths]
        paths_final_segment_matches = [self.evaluate_final_segment(path) for path in paths]
        paths_consecutive_pair_matches = [self.evaluate_consecutive_pairs(path) for path in paths]
        return paths_mutation_matches,paths_final_segment_matches,paths_consecutive_pair_matches
    
    def get_metrics_by_mutation(self,metrics,paths_with_scores,pair=False):
        paths = [path[1:] for _,path in paths_with_scores] # remove root seq from mutations list
        paths_as_mutations = [self.get_path_as_mutations(path) for path in paths]
        metrics_by_mutation = {}

        if pair:
            for i in range(len(paths_as_mutations)):
                metrics_per_path = []
                path = paths_as_mutations[i]
                metric = metrics[i]
                for j in range(len(path)-1):
                    mutation = path[j]
                    next_mutation = path[j+1]
                    metric_value = metric[j]
                    metrics_per_path.append((mutation,next_mutation,metric_value))
                path_name = f"Path {i+1}"
                metrics_by_mutation[path_name] = metrics_per_path
        else:
            for i in range(len(paths_as_mutations)):
                metrics_per_path = []
                path = paths_as_mutations[i]
                metric = metrics[i]
                for j in range(len(path)):
                    mutation = path[j]
                    metric_value = metric[j]
                    metrics_per_path.append((mutation,metric_value))
                path_name = f"Path {i+1}"
                metrics_by_mutation[path_name] = metrics_per_path

        return metrics_by_mutation

    def get_aa_frequencies(self):
        alignment_data = self.process_alignment_data_to_graph()
        num_positions = alignment_data.shape[1]
        num_seqs = alignment_data.shape[0]
        column_frequencies = {}
        
        for pos in range(num_positions):
            column = alignment_data[:,pos]
            unique_aa,counts = np.unique(column,return_counts=True)
            aa_frequencies = {aa:round(count/num_seqs,3) for aa,count in zip(unique_aa,counts)} # get frequency for each aa in the column
            column_frequencies[pos] = aa_frequencies
        
        return column_frequencies
    
    def heatmap_plot_aa_frequencies(self):
        aa_frequencies = self.get_aa_frequencies()
        positions = sorted(aa_frequencies.keys())
        unique_aas = sorted(set([aa for aa_frequencies in aa_frequencies.values() for aa in aa_frequencies]))

        num_positions = len(aa_frequencies)
        num_aas = len(unique_aas)
        aa_frequencies_matrix = np.zeros((num_aas,num_positions))

        for column,pos in enumerate(positions):
            for row,aa in enumerate(unique_aas):
                aa_frequencies_matrix[row,column] = aa_frequencies[pos].get(aa,0)

        plt.figure(figsize=(6,6))
        plt.imshow(aa_frequencies_matrix, cmap='viridis', interpolation='nearest')
        cbar = plt.colorbar()
        cbar.set_label('Frequency')
        plt.yticks(np.arange(len(unique_aas)),unique_aas)
        plt.xticks(np.arange(len(positions)),[138,139,140,141,142,143],rotation=45)

        for column in range(num_positions):
            for row in range(num_aas):
                aa_frequency = aa_frequencies_matrix[row,column]
                plt.text(column, row, aa_frequency, ha='center', va='center', color='white', fontsize=7)

        plt.ylabel("Amino Acid")
        plt.xlabel("Position")
        plt.title("Amino Acid Frequency Heatmap for MSA Data")
    
    @staticmethod
    def box_plot_path_mutation_matches(paths_metrics):
        plt.figure(figsize=(6,3))
        plt.boxplot(paths_metrics, patch_artist = True, notch ='True')
        plt.xlabel("Path Index")
        plt.ylabel("Mutation frequency")
        plt.title("Distribution of Mutuation Frequency Across Generated Evolutionary Paths")

    @staticmethod  
    def box_plot_path_consecutive_pair_matches(paths_metrics):
        plt.figure(figsize=(6,3))
        plt.boxplot(paths_metrics, patch_artist = True)

        plt.xlabel("Path Index")
        plt.ylabel("Consecutive mutation frequency")
        plt.title("Distribution of Consecutive Mutation Co-occurence Across Evolutionary Paths")
    
    @staticmethod
    def get_box_plot_medians(paths_metrics):
        medians = [median.get_ydata()[0] for median in plt.boxplot(paths_metrics)['medians']]
        return medians
