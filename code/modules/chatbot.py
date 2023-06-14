from modules.config import *


class ChatBot:
    def __init__(self, chatbot_topic: str, knowledge_path: str):
        self.knowledge = None
        self.load_data(knowledge_path)
        self.chatbot_topic = chatbot_topic

    def load_data(self, path: str):
        """Loads the knowledge df, appends a prefix, and calculates the number of tokens per section of knowledge"""

        # load data from csv
        self.knowledge = pd.read_csv(path)
        # convert embeddings from CSV str type back to list type
        self.knowledge['Embedding'] = self.knowledge['Embedding'].apply(ast.literal_eval)

        # Format the knowledge df by adding section prefix and token sizes
        # self.knowledge['Content'] = 'Article section:\n\n' + self.knowledge['Content']
        # self.knowledge['Tokens'] = self.knowledge["text"].apply(lambda x: num_tokens(x))
        # self.knowledge['Section'] = 'Wikipedia'
