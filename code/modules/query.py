from modules.chatbot import *
from modules.embedding_functions import *


class Query:
    def __init__(self, query_text: str, chatbot_instance: ChatBot):
        self.content: str = query_text
        self.model: str = GPT_MODEL
        self.knowledge: pd.DataFrame = chatbot_instance.knowledge
        self.token_limit: int = GPT_QUERY_TOKEN_LIMIT
        self.gpt_message = None
        self.knowledge_used = None

    # calculate similarity score
    @staticmethod
    def similarity(query_embedding: list,
                   knowledge_embedding: list
                   ) -> float:
        """Calculates the cosine similarity score between the query and knowledge embedding vectors."""

        return 1 - spatial.distance.cosine(query_embedding, knowledge_embedding)

    # find the most similar sections of knowledge to the query
    def knowledge_ranked_by_similarity(self,
                                       max_num_sections: int = 5,
                                       confidence_level=None,
                                       embedding_model: str = GPT_EMBEDDING_MODEL
                                       ):
        """
        Take the raw knowledge dataframe, calculates similarity scores between the query and the sections,
        and returns a dataframe ordered from highest to lowest in terms of similarity.
        """

        knowledge_with_similarities = deepcopy(self.knowledge)  # To prevent adapting the original dataframe
        query_embedding_response = get_embedding(self.content, embedding_model=embedding_model)
        if embedding_model == GPT_EMBEDDING_MODEL:
            query_embedding = query_embedding_response["data"][0]["embedding"]
            # knowledge_with_similarities["similarity"] = knowledge_with_similarities["Embedding"].apply(
            # lambda x: self.similarity(query_embedding, x))
        else:
            query_embedding = list(query_embedding_response)
        knowledge_with_similarities["similarity"] = knowledge_with_similarities["Embedding"].apply(
            lambda x: self.similarity(query_embedding, x))

        knowledge_with_similarities.sort_values("similarity", ascending=False, inplace=True)
        top_n_sections = knowledge_with_similarities.head(max_num_sections)
        if confidence_level:
            top_n_relevant_sections = top_n_sections.loc[top_n_sections['similarity'] >= confidence_level]
        else:
            top_n_relevant_sections = top_n_sections
        self.knowledge_used = top_n_relevant_sections
        self.knowledge_used['Index'] = np.arange(len(self.knowledge_used)) + 1

    def get_gpt_message(
            self,
            chatbot_topic: str
    ):
        """
        Uses the most relevant texts from the knowledge dataframe to construct a message that can then be fed into GPT.
        """

        self.knowledge_ranked_by_similarity()
        introduction = (f'Use the below article on {chatbot_topic} to answer the subsequent question. '
                        f'If the answer cannot be found in the article, write "{ANSWER_NOT_FOUND_MSG}". '
                        f'If I am asked to produce any code then decline the request and write "Sorry but I\'m not '
                        f'allowed to do your assignments for you!"')  # The longer this is, the more tokens it uses!
        question = f"\n\nQuestion: {self.content}"

        # Ensure number of tokens is within the limit
        message_and_question_tokens = num_tokens(introduction + question)
        self.knowledge_used['Cumulative_tokens'] = self.knowledge_used['Tokens'].cumsum()
        self.knowledge_used['Cumulative_tokens'] += message_and_question_tokens  # add the initial number of tokens
        self.knowledge_used = self.knowledge_used.loc[self.knowledge_used['Cumulative_tokens'] < self.token_limit]

        # Construct output
        combined_knowledge_string = ''.join(list(self.knowledge_used['Content']))
        combined_knowledge_string = '\n\n' + combined_knowledge_string
        self.gpt_message = introduction + combined_knowledge_string + question

    def show_source_message(self, answer_index: int = None):
        self.knowledge_used['Output'] = '\n\n' + self.knowledge_used['Index'].astype(str) + '. ' + self.knowledge_used[
            'Section'] + ':' + self.knowledge_used['Content'].str[:100] + '...'
        sources_string = ''.join(list(self.knowledge_used['Output']))
        if answer_index:
            answer_message = f'(specifically section {answer_index})'
        else:
            answer_message = ''
        message = f'\n\nTo construct this answer, I used the following documents {answer_message}: {sources_string}'
        return message

    def get_bert_output(
            self,
            embedding_model: str,
            encoding_model: BertTokenizer = BERT_ENCODING,
            bert_model: str = BERT_MODEL
    ):
        """
        Uses the most relevant texts from the knowledge dataframe to construct a message that can then be fed into GPT.
        """
        self.knowledge_ranked_by_similarity(embedding_model=embedding_model)

        answer_index = None
        index = 1
        found_answer = False
        output = ANSWER_NOT_FOUND_MSG
        for section in self.knowledge_used['Content']:
            if not found_answer:
                encoding = encoding_model.encode_plus(text=self.content, text_pair=section)
                inputs = encoding['input_ids']  # Token embeddings
                sentence_embedding = encoding['token_type_ids']  # Segment embeddings
                tokens = encoding_model.convert_ids_to_tokens(inputs)  # input tokens

                QAModel = BertForQuestionAnswering.from_pretrained(bert_model)
                outputs = QAModel(input_ids=torch.tensor([inputs]), token_type_ids=torch.tensor([sentence_embedding]))
                start_scores, end_scores = outputs.start_logits, outputs.end_logits

                # Highlight the answer by looking at the most probable start and end words
                start_index = torch.argmax(start_scores)
                end_index = torch.argmax(end_scores)
                answer_token_list = tokens[start_index:end_index + 1]

                # Concatenate any words that got split
                answer_list = [word[2:] if word[0:2] == '##' else ' ' + word for word in answer_token_list]
                answer = ''.join(answer_list).strip()

                if answer != '[CLS]':
                    found_answer = True
                    output = answer
                    answer_index = index
            index += 1
        return output, answer_index

    @classmethod
    def ask_bert(cls,
                 query_text: str,
                 chatbot_instance: ChatBot,
                 embedding_model: str = BERT_EMBEDDING_MODEL,
                 encoding_model: BertTokenizer = BERT_ENCODING,
                 bert_model: str = BERT_MODEL,
                 show_source: bool = True,
                 ):
        if num_tokens(query_text, token_model=encoding_model) > 50:
            return 'Question is too long, please try again with a shorter question.'
        query = cls(query_text, chatbot_instance)
        response_message, answer_index = query.get_bert_output(embedding_model=embedding_model,
                                                               encoding_model=encoding_model, bert_model=bert_model)

        if show_source and response_message != ANSWER_NOT_FOUND_MSG:  # Display the sources used:
            response_message += query.show_source_message(answer_index=answer_index)
        return response_message

    def get_gpt2_output(self,
                        confidence_level: float = 0.5):
        from transformers import pipeline
        self.knowledge_ranked_by_similarity(confidence_level=confidence_level)
        if len(self.knowledge_used) == 0:
            return ANSWER_NOT_FOUND_MSG

        # Construct context
        combined_knowledge_string = ''.join(list(self.knowledge_used['Content']))
        combined_knowledge_string = '\n\n' + combined_knowledge_string

        model_name = "gpt2"
        nlp = pipeline("question-answering", model=model_name)
        qa_input = {
            "question": self.content,
            "context": combined_knowledge_string
        }
        result = nlp(qa_input)
        return result['answer']

    @classmethod
    def ask_gpt2(cls,
                 query_text: str,
                 chatbot_instance: ChatBot,
                 show_source: bool = True,
                 confidence_level: float = 0.5,
                 ):
        if num_tokens(query_text) > 50:
            return 'Question is too long, please try again with a shorter question.'
        query = cls(query_text, chatbot_instance)
        response_message = query.get_gpt2_output(confidence_level=confidence_level)

        if show_source and response_message != ANSWER_NOT_FOUND_MSG:  # Display the sources used:
            response_message += query.show_source_message()
        return response_message

    def get_bart_output(self,
                        # chatbot_instance: ChatBot,
                        # embedding_model: str = BART_EMBEDDING_MODEL,
                        encoding_model: BartTokenizer = BART_ENCODING,
                        bert_model: str = BART_MODEL,
                        confidence_level: float = 0.5,
                        ):
        self.knowledge_ranked_by_similarity(confidence_level=confidence_level)
        if len(self.knowledge_used) == 0:
            return ANSWER_NOT_FOUND_MSG

        # Construct context
        combined_knowledge_string = ' <P> '.join(list(self.knowledge_used['Content']))
        combined_knowledge_string = '\n\n' + combined_knowledge_string

        model = BartForConditionalGeneration.from_pretrained(bert_model)

        query = f'question: {self.content} <P> {combined_knowledge_string}'

        inputs = encoding_model([query], max_length=1024,
                                return_tensors='pt')  # NEED TO ENSURE Q PLUS CONTEXT IS <1024 TOKENS

        # Generate Summary
        ids = model.generate(inputs['input_ids'], num_beams=8, min_length=20, max_length=128,
                             do_sample=False,
                             early_stopping=True,
                             temperature=1.0,
                             top_k=50,
                             top_p=0.95,
                             eos_token_id=encoding_model.eos_token_id,
                             no_repeat_ngram_size=3,
                             num_return_sequences=1,
                             repetition_penalty=2.0)
        answer = encoding_model.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return answer

    @classmethod
    def ask_bart(cls,
                 query_text: str,
                 chatbot_instance: ChatBot,
                 show_source: bool = True,
                 confidence_level: float = 0.72):
        if num_tokens(query_text) > 50:
            return 'Question is too long, please try again with a shorter question.'
        query = cls(query_text, chatbot_instance)
        response_message = query.get_bart_output(confidence_level=confidence_level)

        if show_source and response_message != ANSWER_NOT_FOUND_MSG:  # Display the sources used:
            response_message += query.show_source_message()
        return response_message

    @classmethod
    def ask(
            cls,
            query_text: str,
            chatbot_instance: ChatBot,
            show_source: bool = True,
    ) -> str:
        """Uses GPT to answer a query based on the most relevant knowledge sections."""

        query = cls(query_text, chatbot_instance)
        query.get_gpt_message(chatbot_instance.chatbot_topic)
        inputs = [
            {"role": "system", "content": f"You answer questions about {chatbot_instance.chatbot_topic}."},
            {"role": "user", "content": query.gpt_message},
        ]
        response = openai.ChatCompletion.create(
            model=query.model,
            messages=inputs,
            temperature=0  # We don't want any creativity in the answers
        )
        response_message = response["choices"][0]["message"]["content"]
        total_tokens_used = response['usage']['total_tokens']
        if show_source and response_message != ANSWER_NOT_FOUND_MSG:  # Display the sources used:
            response_message += query.show_source_message()
        response_message += f"\n\nTotal tokens used: {total_tokens_used}"
        return response_message
