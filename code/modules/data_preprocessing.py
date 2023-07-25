from config import *


class TrainingData:
    def __init__(self,
                 save_dir: str,
                 load_dir: str = '/content/drive/MyDrive/Diss/Output'
                 ):
        self.seed = 9  # For only keeping a subset of unanswerable questions
        self.prop = 0.3  # desired proportion of NO_ANS in final dataset
        self.test_prop = 0.2  # Train/Test split
        self.num_chars_to_check = 15  # Number of characters at the start of the str to check for duplicates
        self.min_tokens = 100
        self.max_tokens = 1024 if tokeniser.model_max_length > 1024 else tokeniser.model_max_length
        self.save_dir = save_dir

        # Import data
        log_and_print_message('Loading data ready for preprocessing')
        self.training_df_train_half = pd.read_csv(f'{load_dir}/simplified_dataset_train_half.csv')
        self.training_df_test_half = pd.read_csv(f'{load_dir}/simplified_dataset_test_half.csv')
        self.validation_df = pd.read_csv(f'{load_dir}/simplified_dataset_validation.csv')
        self.training_df = pd.concat([self.training_df_train_half, self.training_df_test_half], ignore_index=True,
                                     sort=False)
        self.num_provided_examples = len(self.training_df)

        # Preprocess data
        log_and_print_message('Preprocessing data')
        self.training_df = self.preprocess_data(self.training_df)
        self.validation_df = self.preprocess_data(self.validation_df)
        self.training_data = self.create_dataset_object()
        # del self.training_df_train_half
        # del self.training_df_test_half

        # Final tokenisation and cleaning
        log_and_print_message('Conducting final cleaning')
        self.final_cleaning()
        self.ensure_ans_non_ans_balance()
        self.get_train_test_split()
        self.training_data.save_to_disk(self.save_dir)
        log_and_print_message(f'Dataset saved to {self.save_dir}')

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pre-processes the data by removing any answers that are simply the first line (using n chars), and shrinks
        the dataset by removing any excess unanswerable questions"""
        # Cleaning data and creating a new col for the model to use
        df['question'] = df['question'].astype(str)
        df['answer'] = df['answer'].astype(str)
        df['content'] = df['content'].astype(str)
        df['answer'] = df['answer'].str.replace('`` ', '"')  # Fixes an issue caused by the decoding

        # Fitering the dataset to only have high quality answers
        # Removing any articles where the answer is simply the first sentence/paragraph
        log_and_print_message(f'Original length: {len(df)}')
        df = df.loc[df['answer'].str[:self.num_chars_to_check] != df['content'].str[:self.num_chars_to_check]]
        log_and_print_message(f'Post-filtered df length: {len(df)}')

        # Discarding some unanswerable examples so the answer-no_ans ratio is favourable
        prop = self.prop + 0.1  # Allow some leeway to ensure self.prop value can be achieved

        # Extracting the unanswerable examples
        df_no_ans = df.loc[df['answer'] == "[NO_ANS]"]

        # Removing any articles where the answer is simply the first sentence/paragraph or there is no answer
        df_good_ans = df.loc[df['answer'] != "[NO_ANS]"]
        log_and_print_message(f'no_ans df length: {len(df_no_ans)}, good_ans df length: {len(df_good_ans)}')

        # Obtain the desired ans/non_ans split
        num_no_ans = prop * len(df_good_ans) / (1 - prop)
        df_no_ans_keep, _ = train_test_split(df_no_ans, train_size=num_no_ans / len(df_no_ans), random_state=self.seed)
        df_filtered = pd.concat([df_good_ans, df_no_ans_keep], ignore_index=True, sort=False)

        log_and_print_message(f'Reduced dataset length to {len(df_filtered)} examples.')

        # Reformat question column
        df_filtered['question'] = df_filtered['question'] + '?'

        return df_filtered

    def create_dataset_object(self) -> DatasetDict:
        """
        Creates an HF dataset object from training and validation dataframes
        """

        dataset = DatasetDict({
            "training": Dataset.from_pandas(self.training_df),
            "validation": Dataset.from_pandas(self.validation_df)
        })
        return dataset

    @staticmethod
    def tokenise(data):
        """
        Tokenises the question/context and answers for a DatasetDict object
        """

        # tokenize the inputs (questions and contexts)
        additional_cols = tokeniser(data['content'], data['question'], truncation=False)

        # tokenize the answers
        targets = tokeniser(text_target=data['answer'], truncation=False)

        # set labels
        additional_cols['labels'] = targets['input_ids']
        additional_cols['num_tokens'] = [len(row) for row in additional_cols["input_ids"]]
        return additional_cols

    def ensure_ans_non_ans_balance(self):
        """
        Removes any excess unanswerable questions so that there's only a desired proportion of them (default=0.3)
        """

        # Extracting the unanswerable examples
        no_ans = self.training_data.filter(lambda row: (row["answer"] == NO_ANS_TOKEN))
        good_ans = self.training_data.filter(lambda row: (row["answer"] != NO_ANS_TOKEN))

        # Discarding some unanswerable examples so the answer-no_ans ratio is favourable in each split
        processed_datasets_dict = {}
        splits = ['training', 'validation']
        for split in splits:
            num_no_ans = self.prop * len(good_ans[split]) / (1 - self.prop)
            no_ans_keep = no_ans[split].train_test_split(train_size=num_no_ans / len(no_ans[split]), seed=self.seed)[
                'train']
            processed_datasets_dict[split] = concatenate_datasets([no_ans_keep, good_ans[split]])

        processed_dataset = DatasetDict({
            "training": processed_datasets_dict[splits[0]],
            "validation": processed_datasets_dict[splits[1]],
        })
        shuffled_dataset = processed_dataset.shuffle(seed=self.seed)
        self.training_data = shuffled_dataset
        log_and_print_message(f'Reduced {splits[0]} dataset to {len(processed_dataset[splits[0]])} examples.')
        log_and_print_message(f'Reduced {splits[1]} dataset to {len(processed_dataset[splits[1]])} examples.')

    def final_cleaning(self):
        """
        Tokenises the dataset and removes any that are too long
        """
        del self.validation_df
        del self.training_df
        gc.collect()

        self.training_data = self.training_data.map(self.tokenise, batched=True)
        self.training_data = self.training_data.filter(
            lambda row: (row["num_tokens"] >= self.min_tokens) & (row["num_tokens"] <= self.max_tokens))
        num_training_examples = len(self.training_data['training'])
        log_and_print_message(
            f'Overall reduced dataset from {self.num_provided_examples} to {num_training_examples} examples.')

    def get_train_test_split(self):
        """
        Combines all examples/questions and creates a new train/test split
        """
        combined_dataset = concatenate_datasets([self.training_data['training'], self.training_data['validation']])
        na = combined_dataset.filter(lambda row: (row["answer"] == NO_ANS_TOKEN))
        a = combined_dataset.filter(lambda row: (row["answer"] != NO_ANS_TOKEN))

        # Get new test_train splits, for the answerable and non-answerable questions
        na_new_split = na.train_test_split(seed=self.seed, test_size=self.test_prop)
        a_new_split = a.train_test_split(seed=self.seed, test_size=self.test_prop)

        # Concat the splits back together so there's 20% test and 30% non-answerable
        training_concatenated = concatenate_datasets([na_new_split['train'], a_new_split['train']])
        test_concatenated = concatenate_datasets([na_new_split['test'], a_new_split['test']])

        self.training_data = DatasetDict({
            "train": training_concatenated,
            "test": test_concatenated,
        })
