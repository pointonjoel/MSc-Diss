from config import *


class ChatBot:
    def __init__(self, chatbot_topic: str, hf_reference: str = None, embedding: str = None, max_new_tokens: int = 150):
        self.chatbot_topic: str = chatbot_topic
        self.knowledge = None
        self.hf_reference: str = hf_reference
        self.max_new_tokens: int = max_new_tokens
        self.embedding_model = GPT_EMBEDDING_MODEL if embedding == 'gpt' else GENERAL_EMBEDDING_MODEL
        self.model = self.get_model()  # If None then the GPT model will be used
        self.tokeniser = self.get_tokeniser()  # If None then the GPT tokeniser will be used
        self.load_data()

    def load_data(self):
        """
        Loads the knowledge df, appends a prefix, and calculates the number of tokens per section of knowledge
        """

        # load data from csv
        if self.embedding_model == GPT_EMBEDDING_MODEL:
            self.knowledge = pd.read_csv(f'assets/{self.chatbot_topic}_knowledge_gpt.csv')
        else:
            self.knowledge = pd.read_csv(f'assets/{self.chatbot_topic}_knowledge.csv')
        # convert embeddings from CSV str type back to list type
        self.knowledge['Embedding'] = self.knowledge['Embedding'].apply(ast.literal_eval)

    def get_model(self):
        """
        Loads the finetuned model if appropriate
        """

        if self.hf_reference:
            if self.hf_reference == MLM_HF_REFERENCE:
                return pipeline(model=self.hf_reference, max_new_tokens=self.max_new_tokens)
            else:
                return AutoModelForSeq2SeqLM.from_pretrained(self.hf_reference)
        else:
            return

    def get_tokeniser(self):
        """
        Loads the finetuned tokeniser if appropriate
        """
        if self.hf_reference:
            if self.hf_reference == MLM_HF_REFERENCE:
                return GPT2Tokenizer.from_pretrained(self.hf_reference)
            else:
                return AutoTokenizer.from_pretrained(self.hf_reference)
        else:
            return
