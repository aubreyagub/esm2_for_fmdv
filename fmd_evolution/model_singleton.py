class ModelSingleton:
    def __new__(cls, model=None, alphabet=None, batch_converter=None):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ModelSingleton, cls).__new__(cls)
            cls.instance.model = model
            cls.instance.alphabet = alphabet
            cls.instance.batch_converter = batch_converter
        return cls.instance
    
    def get_model(self):
        if self.model is None:
            raise ValueError("Model needs to be set.")
        return self.model
    
    def get_alphabet(self):
        if self.alphabet is None:
            raise ValueError("Alphabet needs to be set.")
        return self.alphabet

    def get_batch_converter(self):
        if self.batch_converter is None:
            raise ValueError("Batch converter needs to be set.")
        return self.batch_converter
