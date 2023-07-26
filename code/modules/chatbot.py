from config import *


class ChatBot:
    def __init__(self, chatbot_topic: str):
        self.chatbot_topic = chatbot_topic
        self.knowledge = None
        self.load_data()

    def load_data(self):
        """
        Loads the knowledge df, appends a prefix, and calculates the number of tokens per section of knowledge
        """

        # load data from csv
        self.knowledge = pd.read_csv(f'assets/{self.chatbot_topic}_knowledge.csv')
        # convert embeddings from CSV str type back to list type
        self.knowledge['Embedding'] = self.knowledge['Embedding'].apply(ast.literal_eval)

        # Format the knowledge df by adding section prefix and token sizes
        # self.knowledge['Content'] = 'Article section:\n\n' + self.knowledge['Content']
        # self.knowledge['Tokens'] = self.knowledge["text"].apply(lambda x: num_tokens(x))
        # self.knowledge['Section'] = 'Wikipedia'
