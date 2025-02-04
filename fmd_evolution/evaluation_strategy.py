import torch
import torch.nn.functional as F


class EvaluationStrategy:
    def __init__(self,probable_threshold=0.8,functional_threshold=0.3):
        self.probable_threshold = probable_threshold
        self.functional_threshold = functional_threshold
    
    def is_sequence_probable(self,sequence):
        sequence_only_logits = sequence.sequence_only_logits
        log_p = torch.log(sequence_only_logits) # epsilon to avoid nan
        mean_log_p = torch.mean(log_p)
        sequence_p = torch.exp(mean_log_p)
        return sequence_p>self.probable_threshold,sequence_p

    #     # set, then in another func compare change from previous sequence
    #     # other ascpects to evaluate: increase protein fitness + antigenic pressure if applicable + functional + env via past data

    def is_sequence_functional(self,sequence):
        sequence_embedding = sequence.embeddings
        previous_sequence_embedding = sequence.parent_seq.embeddings
        # compare with previous embeddings using a distance metric: cosine similarity
        # cosine_sim = F.cosine_similarity(sequence_embedding,previous_sequence_embedding,dim=1)
        # mean_cosine_sim = torch.mean(cosine_sim).item()
        # print(f"Cosine similarity = {mean_cosine_sim}")
        # return mean_cosine_sim> self.functional_threshold,mean_cosine_sim
        l2_distance = torch.norm(sequence_embedding - previous_sequence_embedding, p=2, dim=1)
        mean_l2_distance = torch.mean(l2_distance).item()
        functional_score = 1 - mean_l2_distance # so both metrics are in the same direction
        return functional_score>self.functional_threshold,functional_score
    
    def should_continue_mutating(self,sequence):
        is_probable,sequence_p = self.is_sequence_probable(sequence)
        is_functional,cosine_sim = self.is_sequence_functional(sequence)
        sequence.mutation_score = (sequence_p + cosine_sim)/2 # average of scores
        if is_probable and is_functional:
            return True
        else:
            return False
        

    # next steps: instead of just checking it falls above a threshold, I need to cehck the direction of change (increased or decrese) to correspond with increased or decreased fitness