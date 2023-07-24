from config import *

def remove_unnecessary_whitespace(sentence):
    # Remove whitespace before punctuation
    sentence = re.sub(r'\s+([.,!?)])', r'\1', sentence)
    sentence = re.sub(r'\s+\'', '\'', sentence)

    # Remove whitespace after punctuation
    sentence = re.sub(r'([(])\s+', r'\1', sentence)

    # Remove whitespace either side of '-'
    sentence = re.sub(r'\s*-\s*', '-', sentence)

    # Remove any double/triple etc spaces
    sentence = re.sub(r'\s+', ' ', sentence.strip())

    return sentence.strip()


def remove_references(sentence):
    sentence = re.sub(r'\[\d+]', '', sentence)
    return sentence


class AllData:
    def __init__(self, default='dev', cache_dir=None):
        self.dataset = load_dataset("natural_questions", default, cache_dir=cache_dir)
        if default != 'dev':
            self.dataset = self.dataset['train']
            self.dataset = self.dataset.train_test_split(test_size=0.5, shuffle=False)
        self.simplified_dataset = None
        self.training_data = None
        self.data_split = default
        self.unwanted_tags = ['table', 'tr', 'th', 'td', 'ul', 'li', 'dl', 'dd', 'ol', 'dt']
        # self.max_output_tokens = 120
        # self.max_input_tokens = tokeniser.model_max_length - self.max_output_tokens
        self.extract_data()

    def extract_data(self):
        # many potential answers, took the main long answer one for simplicity and limitations in computing power
        self.simplified_dataset = []

        log_and_print_message('Extracting the dataset')
        datacol = 'validation' if self.data_split == 'dev' else 'test'
        for item in tqdm(self.dataset[datacol]):
            # Initially checking whether the answer contains unwanted contents

            # Getting the document tokens, to be able to extract the answer
            wiki_doc_tokens = item['document']['tokens']['token']
            wiki_doc = ' '.join(wiki_doc_tokens)

            # Getting the main answer
            wiki_answer_meta = item['annotations']['long_answer'][0]
            wiki_answer_tokens = wiki_doc_tokens[wiki_answer_meta['start_token']:wiki_answer_meta['end_token']]
            wiki_answer = ' '.join(wiki_answer_tokens)

            # Extract HTML tags
            soup = BeautifulSoup(wiki_answer, 'html.parser')
            tags = [tag.name for tag in soup.find_all()]
            common_elements = any(item in tags for item in self.unwanted_tags)

            # Only continue to add the item if there are no unwanted tags in the answer
            if not common_elements:
                # Remove <P> tags and any unwanted whitespace
                wiki_answer = ' '.join([p.get_text() for p in soup.find_all('p')]).strip()
                wiki_answer = remove_unnecessary_whitespace(wiki_answer)
                answer = wiki_answer if wiki_answer != '' else '[NO_ANS]'

                # Getting the document ID and title
                id = item['id']
                title = item['document']['title']
                url = item['document']['url']

                # Getting main content to feed as a context
                page = item['document']['html']  # variable was called: page
                document = html.document_fromstring(page)
                content = ' '.join([p.text_content() for p in document.xpath('//p')])
                content = remove_references(content)

                # Getting the question
                wiki_question = item['question']['text']

                # Formatting - NEED TO REMOVE THIS
                formatted_text = f'[Question: ] {wiki_question} \n\n [Context: ] {content}'

                # num_tokens_content = len(encoding.encode(formatted_text))
                # num_tokens_answer = len(encoding.encode(answer))
                # total_tokens = num_tokens_content + num_tokens_answer

                # if num_tokens_answer<=self.max_output_tokens and num_tokens_content<=self.max_input_tokens and num_tokens_content > 100: # Used instead of truncating
                # tokenised_content = tokeniser(wiki_question, content, truncation=False)
                # tokenised_answer = tokeniser(answer, truncation=False)

                # NEED TO CHECK TO SEE IF ANY ROWS HAVE BLANK COLUMN ENTRIES
                self.simplified_dataset.append({'id': id,
                                                'title': title,
                                                'url': url,
                                                'content': content,
                                                'question': wiki_question,
                                                'answer': answer,
                                                'start_token': wiki_answer_meta['start_token'],
                                                'end_token': wiki_answer_meta['end_token'],
                                                'length': wiki_answer_meta['end_token'] - wiki_answer_meta[
                                                    'start_token'],
                                                # 'formatted_text': formatted_text, # NEED TO REMOVE THIS
                                                # 'num_tokens': num_tokens_content,
                                                # 'tokenised_content': tokenised_content,
                                                # 'labels': tokenised_answer['input_ids']
                                                })

    def export_simplified_dataset(self, path="/content/drive/MyDrive/Diss/Output/simplified_dataset.csv"):
        df = pd.DataFrame(self.simplified_dataset)
        df.to_csv(path, index=False)