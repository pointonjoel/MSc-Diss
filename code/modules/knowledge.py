from embedding_functions import *
from config import GENERAL_EMBEDDING_MODEL, MIN_LENGTH, GPT_MAX_SECTION_TOKENS, pd, mwparserfromhell
from config import SECTIONS_TO_IGNORE, DocTypeNotFoundError, wikipedia, log_and_print_message, fitz, unidecode, re


class Knowledge:
    def __init__(self, chatbot_topic: str, model_family: str = 'T5'):
        self.chatbot_topic: str = chatbot_topic  # The chatbot domain, used to export the knowledge as a csv file
        self.model_family: str = model_family  # The model type, used to detect the tokeniser
        self.token_model = self.get_token_model()  # Used to calculate the number of tokens per section
        self.embedding_model: str = GENERAL_EMBEDDING_MODEL  # Needs to be consistent with the query embedding model
        self.df: pd.DataFrame = self.get_blank_knowledge_df()  # need to add code to remove small sections (<16 chars?)
        self.max_tokens: int = self.get_max_tokens()  # max number of tokens per section
        self.min_section_length: int = MIN_LENGTH  # min character length for each section

    def get_token_model(self):
        """
        Checks the model_family is appropriate and returns the relevant tokeniser.
        """
        if self.model_family == 'GPT':
            return GPT_TOKENISER
        elif self.model_family == 'T5':
            return T5_TOKENISER
        elif self.model_family == 'BART':
            return BART_TOKENISER
        else:
            raise ModelNotSupportedError('The model type doesn\'t have an associated Tokeniser and as such isn\'t '
                                         'currently supported. Please select from GPT, T5 and BART.')

    def get_max_tokens(self):
        """
        Checks the model_family is appropriate and returns the number of tokens the model can handle per section.
        """
        if self.embedding_model == GPT_EMBEDDING_MODEL:
            return GPT_MAX_SECTION_TOKENS
        elif self.embedding_model == GENERAL_EMBEDDING_MODEL:
            return GENERAL_EMBEDDING_MODEL.max_seq_length
        else:
            raise ModelNotSupportedError('The EMBEDDING model type isn\'t currently supported. Please select from GPT, T5 and '
                                         'BART.')

    # def get_embedding_model(self):
    #     return GPT_EMBEDDING_MODEL if self.model_family == 'GPT' else GENERAL_EMBEDDING_MODEL

    @staticmethod
    def get_blank_knowledge_df() -> pd.DataFrame:
        """
        Creates a blank dataframe to contain the sections of knowledge.
        """
        return pd.DataFrame(columns=['Source', 'Heading', 'Subheading', 'Page', 'Content'])

    # @staticmethod
    def get_populated_knowledge_df(self,
                                   source: list = (),
                                   heading: list = (),
                                   subheading: list = (),
                                   page: list = (),
                                   content: list = ()
                                   ) -> pd.DataFrame:
        """
        Converts a list of data into a populated dataframe of sections of knowledge.
        """
        knowledge = self.get_blank_knowledge_df()
        if source:
            knowledge['Source'] = source
        if heading:
            knowledge['Heading'] = heading
        if subheading:
            knowledge['Subheading'] = subheading
        if page:
            knowledge['Page'] = page
        if content:
            knowledge['Content'] = content

        return knowledge

    def extract_wiki_sections(self,
                              page_name: str,
                              content: mwparserfromhell.wikicode.Wikicode,
                              sections_to_ignore: list = SECTIONS_TO_IGNORE
                              ) -> pd.DataFrame:
        """
        Creates a df of sections by extracting section content from a Wikicode.
        """

        knowledge = self.get_blank_knowledge_df()
        for section in content.get_sections(levels=[2]):
            section_headings = section.filter_headings()
            section_header = str(section_headings[0])
            if len(section_headings) == 1:  # therefore a section title, not a subsection
                section = section.strip(section_header)
                if section_header.strip("=" + " ") not in sections_to_ignore:  # append to df
                    new_row = {'Source': f'Wikipedia ({page_name})', 'Heading': section_header.strip("=" + " "),
                               'Content': section}
                    knowledge = pd.concat([knowledge, pd.DataFrame.from_records([new_row])])
            elif len(section_headings) > 1 and section_header.strip(
                    "=" + " ") not in sections_to_ignore:  # therefore subsections
                # Append the text before the first subsection
                initial_text = section.split(str(section_headings[1]))[0]
                initial_text = initial_text.strip(section_header)
                new_row = {'Source': f'Wikipedia ({page_name})', 'Heading': section_header.strip("=" + " "),
                           'Content': initial_text}
                knowledge = pd.concat([knowledge, pd.DataFrame.from_records([new_row])])
                for subsection in section.get_sections(levels=[3]):
                    subsection_sections = subsection.get_sections(levels=[3])[0]
                    subsection_headings = subsection_sections.filter_headings()
                    subsection_header = str(subsection_headings[0])
                    subsection = subsection.strip(subsection_header)
                    if subsection_header.strip("=" + " ") not in sections_to_ignore:  # append to df
                        new_row = {'Source': f'Wikipedia ({page_name})', 'Heading': section_header.strip("=" + " "),
                                   'Subheading': subsection_header.strip("=" + " "), 'Content': subsection}
                        knowledge = pd.concat([knowledge, pd.DataFrame.from_records([new_row])])
        return knowledge

    @staticmethod
    def generate_source_column(df: pd.DataFrame, doc_type: str = 'Wiki') -> pd.DataFrame:
        """
        Creates a new column in the df which contains a summary of the source location.
        """

        df.fillna('', inplace=True)

        if doc_type == 'Wiki':
            df['Section'] = df['Source'] + '->' + df['Heading'] + '->' + df['Subheading']
            df['Section'] = df['Section'].str.replace('->->', '')
            df['Section'] = df['Section'].str.rstrip('_->')
        elif doc_type == 'PDF':
            df['Section'] = df['Source'] + '->Page(s)' + df['Page']
        else:
            raise DocTypeNotFoundError('DocType not specified - could not produce the source column.')
        return df

    @staticmethod
    def clean_section_contents(df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a cleaned up section with <ref>xyz</ref> patterns and leading/trailing whitespace removed.
        """

        # text = re.sub(r"<ref.*?</ref>", "", text)
        df['Content'] = df['Content'].str.replace(r"<ref.*?</ref>", "", regex=True)
        df['Content'] = df['Content'].str.strip()  # removes whitespace
        df['Content'] = '\n' + df['Content']  # need to add the \n back to the start of each title
        return df

    def merge_elements_of_list(self, list_of_strings: list, delimiter: str = "\n"):
        """
        Merges a list of strings together where possible, as long as each string is less than the max_tokens limit.
        """
        potential_for_more_merging = False
        merged_list = []
        skip_item = False
        for i in range(len(list_of_strings)):
            if not skip_item:
                if i == len(list_of_strings) - 1:
                    merged_list.append(list_of_strings[i])
                else:
                    merged_strings = list_of_strings[i] + delimiter + list_of_strings[i + 1]
                    if num_tokens(merged_strings) <= self.max_tokens:
                        merged_list.append(merged_strings)
                        skip_item = True  # make it skip the element we just merged
                        potential_for_more_merging = True
                    else:
                        merged_list.append(list_of_strings[i])
            else:
                skip_item = False  # set the default back to False unless otherwise specified
        return merged_list, potential_for_more_merging

    def force_split_string(self,
                           string: str,
                           encoding=GPT_TOKENISER) -> list:
        """
        Force a section to be split into 2 (to be used if it has no delimiter).
        """

        list_of_strings = []
        if num_tokens(string) <= self.max_tokens:
            return [string]
        else:
            needs_truncating = True
            while needs_truncating:
                encoded_string = encoding.encode(string)
                truncated_string = encoding.decode(encoded_string[:self.max_tokens])
                remainder_of_string = encoding.decode(encoded_string[self.max_tokens:])
                list_of_strings.append(truncated_string)
                string = remainder_of_string
                if num_tokens(remainder_of_string) <= self.max_tokens:
                    needs_truncating = False
                    list_of_strings.append(remainder_of_string)
        return list_of_strings

    def split_long_sections(self, df: pd.DataFrame, delimiter: str = '\n') -> pd.DataFrame:
        """
        Splits long sections of text into smaller ones, using each delimiter.
        """

        new_dict_of_shorter_sections = self.get_blank_knowledge_df().to_dict('records')
        df_as_dict = df.to_dict('records')
        for section in df_as_dict:
            # for delimiter in delimiters:
            if section['Tokens'] <= self.max_tokens:
                new_dict_of_shorter_sections.append(section)
            else:
                # needs to be split up
                if delimiter == '':  # meaning that we just need to truncate it.
                    text = self.force_split_string(section['Content'])
                else:
                    text = section['Content'].split(delimiter)
                    if delimiter == '. ':
                        for i in range(len(text) - 1):
                            text[i] += delimiter
                potential_for_more_merging = True
                i = 0
                while potential_for_more_merging:
                    if i > 20:
                        break
                    else:
                        text, potential_for_more_merging = self.merge_elements_of_list(text)

                # The sections should be merged into acceptable sizes:
                if len(text) > 1:
                    for string in text:
                        item_to_append = {'Source': section['Source'], 'Heading': section['Heading'],
                                          'Subheading': section['Subheading'], 'Content': string,
                                          'Section': section['Section'], 'Tokens': num_tokens(string)}

                        new_dict_of_shorter_sections.append(item_to_append)
                else:
                    item_to_append = {'Source': section['Source'], 'Heading': section['Heading'],
                                      'Subheading': section['Subheading'], 'Content': text[0],
                                      'Section': section['Section'], 'Tokens': num_tokens(text[0])}
                    # we shouldn't have this because the text should be more than the acceptable number of tokens
                    new_dict_of_shorter_sections.append(item_to_append)
        return pd.DataFrame(new_dict_of_shorter_sections)

    def format_and_get_embeddings(self, knowledge: pd.DataFrame) -> pd.DataFrame:
        """
        Formats the knowledge dataframe 'content' column and obtains embeddings for each section content.
        """
        # Append '\n' to the start if it doesn't already have one
        knowledge.loc[~knowledge['Content'].str.startswith('\n'), 'Content'] = '\n' + knowledge.loc[
            ~knowledge['Content'].str.startswith('\n'), 'Content']

        # Get embeddings
        if self.model_family == 'GPT':
            response = get_embedding(list(knowledge['Content']), embedding_model=self.embedding_model)
            for i, be in enumerate(response["data"]):
                assert i == be["index"]  # double check embeddings are in same order as input
            batch_embeddings = [e["embedding"] for e in response["data"]]
            knowledge['Embedding'] = batch_embeddings
        else:
            knowledge['Embedding'] = get_embedding(list(knowledge['Content']),
                                                   embedding_model=self.embedding_model).tolist()

        knowledge['Tokens'] = knowledge["Content"].apply(lambda x: num_tokens(x, token_model=self.token_model))
        return knowledge

    def append_wikipedia_page(self, page_name: str,
                              sections_to_ignore: list = SECTIONS_TO_IGNORE):
        """
        Takes a wikipedia page and appends the sections to the knowledge df.
        """

        try:
            site = wikipedia.page(page_name, auto_suggest=False)
            text = site.content
            parsed_text = mwparserfromhell.parse(text)

            # Creating initial df and appending the introduction paragraph (the text up to the first heading)
            intro = str(parsed_text).split(str(parsed_text.filter_headings()[0]))[0]
            knowledge = self.get_blank_knowledge_df()
            new_row = {'Source': f'Wikipedia ({page_name})', 'Content': '\n' + intro}
            knowledge = pd.concat([knowledge, pd.DataFrame.from_records([new_row])])

            section_content = self.extract_wiki_sections(page_name=page_name, content=parsed_text,
                                                         sections_to_ignore=sections_to_ignore)
            knowledge = pd.concat([knowledge, section_content])

            # Generate succinct heading information
            knowledge = self.generate_source_column(knowledge, doc_type='Wiki')

            # Remove unwanted strings and whitespace
            knowledge = self.clean_section_contents(knowledge)

            # Generate number of tokens in each section
            knowledge['Tokens'] = knowledge["Content"].apply(lambda x: num_tokens(x, token_model=self.token_model))

            # Split long sections
            for delim in ["\n\n", "\n", ". ", '']:
                knowledge = self.split_long_sections(knowledge, delimiter=delim)

            # Remove short sections
            knowledge = knowledge.loc[knowledge['Content'].str.len() > self.min_section_length]

            # conduct final formatting and getting embeddings
            knowledge = self.format_and_get_embeddings(knowledge)

            # Concat with main self.df
            self.df = pd.concat([self.df, knowledge])

            log_and_print_message(
                f'The following Wikipedia page has been successfully added to the knowledge database: {page_name}')

        except wikipedia.exceptions.PageError:  # The wiki page doesn't exist
            log_and_print_message(f'The wiki page {page_name} can\'t be found. Please check and try again.')

    @staticmethod
    def sentence_end(text: str, separators: list = ('.', '!', '?')) -> bool:
        """
        Detects whether there is an end of a sentence.
        """
        for sep in separators:
            if text.endswith(sep):
                return True
            else:
                pass
        # So if none of the separators are used to end the string, then the sentence has come to an unnatural end
        return False

    def extract_pdf_text(self,
                         filename_path: str,
                         document_name: str,
                         page_limit: int = None) -> pd.DataFrame:
        """
        Parses the PDF using Fitz, extracting and cleaning the text.
        """
        # Open document
        doc = fitz.open(filename_path)
        content = self.get_blank_knowledge_df()

        # Iterate through the content
        for page in doc:
            page_limit = doc.page_count if not page_limit else page_limit
            if page.number <= page_limit:
                block_content = page.get_text("blocks")  # .encode("utf8") # "blocks"
                for block in block_content:
                    if block[6] == 0:  # I.e. only extract text
                        plain_text = unidecode(block[4])  # .decode('latin1') #.decode('utf-8')
                        new_row = {'Source': document_name, 'Page': page.number, 'Content': plain_text}
                        content = pd.concat([content, pd.DataFrame.from_records([new_row])])
            else:
                pass

        # Remove any unwanted content - specifically websites
        content['Content'] = content['Content'].apply(lambda x: re.sub(r'http\S+', '', x))
        content['Content'] = content['Content'].apply(lambda x: re.sub(r'www.+', '', x))
        content = content.loc[content['Content'] != '\n']
        return content

    def append_pdf(self, filename_path: str, document_name: str):
        """
        Takes a PDF document and appends the sections to the knowledge df.
        """
        # Extract the text into blocks
        knowledge = self.extract_pdf_text(filename_path, document_name)

        section_texts = knowledge['Content'].tolist()
        page_numbers = knowledge['Page'].tolist()
        merged_texts = ['']
        merged_page_numbers = []

        for i in range(len(section_texts)):
            if num_tokens(section_texts[i]) <= self.max_tokens:
                merged_text = merged_texts[-1] + section_texts[i]
                if num_tokens(merged_text) <= self.max_tokens:  # i.e. we can merge the text without it being too long
                    merged_texts[-1] = merged_text  # Update the latest line with the merged text
                    if len(merged_page_numbers) == 0:  # i.e. the 1st piece of text
                        merged_page_numbers.append([page_numbers[i]])
                    elif page_numbers[i] not in merged_page_numbers[-1]:  # to find the combined page number
                        merged_page_numbers[-1] = merged_page_numbers[-1] + [
                            page_numbers[i]]  # f'{merged_page_numbers[-1]}/{page_numbers[i]}'
                    else:
                        pass
                else:  # we can't merge the current text with the previous ones
                    merged_texts.append(section_texts[i])
                    merged_page_numbers.append([page_numbers[i]])
                    pass
            else:  # very unlikely
                pass  # NEED TO UPDATE THIS! Split the long bit up?
        merged_page_numbers = [str(pages) if len(pages) == 0 else '/'.join(str(page) for page in pages) for pages in
                               merged_page_numbers]
        knowledge = self.get_populated_knowledge_df(content=merged_texts, page=merged_page_numbers)
        knowledge['Source'] = document_name

        # Generate succinct heading information
        knowledge = self.generate_source_column(knowledge, doc_type='PDF')

        # conduct final formatting and getting embeddings
        knowledge = self.format_and_get_embeddings(knowledge)

        # Concat with main self.df
        self.df = pd.concat([self.df, knowledge])

        log_and_print_message(
            f'The following PDF has been successfully added to the knowledge database: '
            f'{document_name} ({filename_path})')

    def export_to_csv(self):
        """
        Saves the knowledge df to a CSV file.
        """

        self.df.to_csv(f'assets/{self.chatbot_topic}_knowledge.csv', index=False)
