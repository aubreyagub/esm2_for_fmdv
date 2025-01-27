import torch

class ProteinSequence:
    def __init__(self,id,reference_seq):
        self.id = id
        self.reference_seq = reference_seq
        self.current_seq = reference_seq
        # self.current_batch_tokens = []
        # self.current_all_logits = []
        # self.current_seq_logits = []
        self.mutations = []

    def set_current_seq(self,new_seq):
        self.current_seq = new_seq

    # def set_current_batch_tokens(self,current_batch_tokens):
    #     self.current_batch_tokens = current_batch_tokens

    # def set_current_all_logits(self,current_all_logits)
    #     self.current_all_logits = current_all_logits

    # def set_current_seq_logits(self, current_seq_logits):
    #     self.current_seq_logits = current_seq_logits
    
    def get_reference_seq(self):
        return self.reference_seq

    def get_current_seq(self):
        return self.current_seq

    # def get_current_batch_tokens(self):
    #     return self.current_batch_tokens 

    # def get_current_all_logits(self)
    #     return self.current_all_logits 

    # def get_current_seq_logits(self):
    #     return self.current_seq_logits 
        
    def get_mutations(self):
        return self.mutations

    def add_mutation(self, mutation):
        self.mutations.append(mutation)

    def get_batch_tokens(self,model,alphabet,batch_converter):
        data = [(self.id,self.current_seq)]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        return batch_tokens
         
    def get_all_logits(self,batch_tokens,model,start_pos=138, end_pos=143):
        if torch.cuda.is_available():
            batch_tokens = batch_tokens.to(device=device, non_blocking=True)
        with torch.no_grad():
            logits_raw = model(batch_tokens)["logits"].squeeze(0)
            aa_tokens_len = len(batch_tokens[0,1:-1]) # without start and stop tokens
            logits_target = logits_raw [1:(aa_tokens_len+1),4:24]
    
        # normalise logits to convert to probabilities 
        lsoftmax = torch.nn.LogSoftmax(dim=1)
        logits = lsoftmax(logits_target)
    
        return logits

    def get_current_seq_logits(self,batch_tokens,all_logits,token_offset=4):
        current_seq_tokens = batch_tokens[0,1:-1]
        current_seq_logits = all_logits[torch.arange(all_logits.size(0)),current_seq_tokens - token_offset]
        return current_seq_logits

    def mutate_seq(self,pos,aa_char):
        list_seq = list(self.current_seq)
        list_seq[pos] = aa_char
        self.current_seq = "".join(list_seq)

    def get_logits(self,model,alphabet,batch_converter):
        batch_tokens = self.get_batch_tokens(model,alphabet,batch_converter)
        all_logits = self.get_all_logits(batch_tokens,model)
        current_seq_logits = self.get_current_seq_logits(batch_tokens,all_logits)
        return all_logits,current_seq_logits
        

