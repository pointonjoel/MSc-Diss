# from config import *
from embedding_functions import *


class Knowledge:
    def __init__(self, topic, model):
        self.topic: str = topic
        self.model: str = model
        self.token_model = self.get_token_model()
        self.embedding_model: str = self.get_embedding_model()
        self.df: pd.DataFrame = self.get_blank_knowledge_df()  # need to add code to remove small sections (<16 chars?)
        self.max_tokens: int = self.get_max_tokens()  # max number of tokens per section
        self.min_section_length = MIN_LENGTH  # min character length for each section

    def get_token_model(self):
        return GPT_ENCODING if self.model == 'GPT' else BERT_ENCODING

    def get_max_tokens(self):
        return GPT_MAX_SECTION_TOKENS if self.model == 'GPT' else BERT_MAX_SECTION_TOKENS

    def get_embedding_model(self):
        return GPT_EMBEDDING_MODEL if self.model == 'GPT' else BERT_EMBEDDING_MODEL

    @staticmethod
    def get_blank_knowledge_df() -> pd.DataFrame:
        return pd.DataFrame(columns=['Source', 'Heading', 'Subheading', 'Page', 'Content'])

    def extract_wiki_sections(self,
                              page_name: str,
                              content: mwparserfromhell.wikicode.Wikicode,
                              sections_to_ignore: list = SECTIONS_TO_IGNORE
                              ) -> pd.DataFrame:
        """
        Creates a df of sections by extracting section content from a Wikicode
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
    def generate_source_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a new column in the df which contains a summary of the source location
        """

        df.fillna('', inplace=True)
        df['Section'] = df['Source'] + '->' + df['Heading'] + '->' + df['Subheading']
        df['Section'] = df['Section'].str.replace('->->', '')
        df['Section'] = df['Section'].str.rstrip('_->')
        return df

    @staticmethod
    def clean_section_contents(df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a cleaned up section with <ref>xyz</ref> patterns and leading/trailing whitespace removed
        """

        # text = re.sub(r"<ref.*?</ref>", "", text)
        df['Content'] = df['Content'].str.replace(r"<ref.*?</ref>", "", regex=True)
        df['Content'] = df['Content'].str.strip()  # removes whitespace
        df['Content'] = '\n' + df['Content']  # need to add the \n back to the start of each title
        return df

    def merge_elements_of_list(self, list_of_strings: list, delimiter: str = "\n"):
        potential_for_more_merging = False
        merged_list = []
        skip_item = False
        for i in range(len(list_of_strings)):
            if not skip_item:
                if i == len(list_of_strings) - 1:
                    merged_list.append(list_of_strings[i])
                else:
                    merged_strings = list_of_strings[i] + delimiter + list_of_strings[i + 1]
                    if num_tokens(merged_strings) < self.max_tokens:
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
                           encoding=GPT_ENCODING) -> list:
        """
        Force a section to be split into 2 (to be used if it has no delimiter)
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
                if num_tokens(remainder_of_string) < self.max_tokens:
                    needs_truncating = False
                    list_of_strings.append(remainder_of_string)
        return list_of_strings

    def split_long_sections(self, df: pd.DataFrame, delimiter: str = '\n'):
        """
        Splits long sections of text into smaller ones, using each delimiter
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

    def append_wikipedia_page(self, page_name: str,
                              sections_to_ignore: list = SECTIONS_TO_IGNORE):
        """
        Takes a wikipedia page and appends the sections to the knowledge df
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
            knowledge = self.generate_source_column(knowledge)
            self.df = pd.concat([self.df, knowledge])

            # Remove unwanted strings and whitespace
            self.df = self.clean_section_contents(self.df)

            # Generate number of tokens in each section
            self.df['Tokens'] = self.df["Content"].apply(lambda x: num_tokens(x, token_model=self.token_model))

            # Split long sections
            for delim in ["\n\n", "\n", ". ", '']:
                self.df = self.split_long_sections(self.df, delimiter=delim)

            # Remove short sections
            self.df = self.df.loc[self.df['Content'].str.len() > self.min_section_length]

            # Append '\n' to the start if it doesn't already have one
            self.df.loc[~self.df['Content'].str.startswith('\n'), 'Content'] = '\n' + self.df.loc[
                ~self.df['Content'].str.startswith('\n'), 'Content']

            # Get embeddings
            if self.model == 'GPT':
                response = get_embedding(list(self.df['Content']), embedding_model=self.embedding_model)
                for i, be in enumerate(response["data"]):
                    assert i == be["index"]  # double check embeddings are in same order as input
                batch_embeddings = [e["embedding"] for e in response["data"]]
                self.df['Embedding'] = batch_embeddings
            else:
                self.df['Embedding'] = get_embedding(list(self.df['Content']),
                                                     embedding_model=self.embedding_model).tolist()
            log_and_print_message(
                f'The following page has been successfully added to the knowledge database: {page_name}')

        except wikipedia.exceptions.PageError:  # The wiki page doesn't exist
            log_and_print_message(f'The wiki page {page_name} can\'t be found. Please check and try again.')

    def sentence_end(self, text: str, separators: list = ['.', '!', '?']) -> bool:
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
        # Open document
        doc = fitz.open(filename_path)
        content = self.get_blank_knowledge_df()

        # Iterate through the content
        for page in doc:
            page_limit = doc.page_count if not page_limit else page_limit
            if page.number <= page_limit:
                block_content = page.get_text("blocks")
                for block in block_content:
                    if block[6] == 0:  # I.e. only extract text
                        plain_text = unidecode(block[4])
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
        Takes a PDF document and appends the sections to the knowledge df
        """
        # Extract the text into blocks
        content = self.extract_pdf_text(filename_path, document_name)

        section_texts = content['Content']
        page_numbers = content['Page']
        merged_texts = []
        merged_page_numbers = []

        for i in range(len(section_texts)):
            if num_tokens(section_texts[i]) < self.max_tokens:
                pass

        log_and_print_message(
            f'The following PDF has been successfully added to the knowledge database: {document_name} ({filename_path})')


    def export_to_csv(self, filename):
        """
        Saves the knowledge df to a CSV file
        """

        location = 'assets/' + filename
        self.df.to_csv(location, index=False)
