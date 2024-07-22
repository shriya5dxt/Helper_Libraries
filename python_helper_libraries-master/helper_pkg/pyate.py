from os import cpu_count
import pandas as pd
# from tqdm import tqdm
from . import Helper
import spacy
from pyate.term_extraction_pipeline import TermExtractionPipeline
import re
import numpy as np
from tqdm import tqdm
from .modified_pyate import Modified_PYATE
from typing import List
tqdm.pandas()
spacy.prefer_gpu()


class PYATE(Helper):
    def __init__(self) -> None:
        Helper.__init__(self)

        self.test = None
        # self.nlp = spacy.load('en_core_web_lg', disable=[
        #                       'ner', 'parser'])
        # self.nlp.add_pipe(TermExtractionPipeline())

        remove_dict = {
            r'^=': '',
            r'^-': '',
            r'--+': ' ',
            r'===+': ' ',
            r'\n+': ' ',
            r'(\>\>|\=\>)': '.',
            r'\t+': ' ',
            r' +': ' ',
            r'\S+@\S+': ' ',
            #                 r'\d\d\d\d\d+': '',
            r'\*+': '',
            r'#+': '',
            r'\s•\s': ' ',
            r'[_\'\"-\(\)\^|]+': '',
            r';+': '.',
            r'[<\[\]]+': '',
            r'[\{\}]+': '',
            r'- -': '',
            #                 r'\d\d:\d\d': '',
            r'\d:\d\d (a|p)m [a-z]*:': '',
            r'\d:\d\d (a|p)m': '',
            #                 r'\d\d:\d\d': ''
            'Windows 10':'Windows10',
	        'windows 10': 'windows10',
	        'WINDOWS 10':'WINDOWS10',
	        'WIN 10':'WIN10',
	        'win 10':'win10',
	        'Win 10':'Win10',
	        'win .10':'win10',
    	    'wind 10':'wind 10',
	        'Windows 11':'Windows11',
	        'windows 11': 'windows11',
	        'WINDOWS 11':'WINDOWS11',
	        'WIN 11':'WIN11',
	        'win 11':'win11',
	        'Win 11':'Win11',
	        'win .11':'win11',
	        'wind 11':'wind 11'

        }
        self.remove_patterns = {}
        for remove_ele, sub_with in remove_dict.items():
            pattern = re.compile(remove_ele)
            self.remove_patterns.update({pattern: sub_with})

    def apply(self, df: pd.DataFrame, target_col: str, id_col: str, processed_table: str, block_size: int = 5000, verbose: bool = False) -> None:
        """Function to apply term extraction using the modified pyate module. 
            It processes the data in batches and loads them into the processed table using the load_to_vertica_function.

        Args:
            df (pd.DataFrame): Dataframe containing the data for term extraction.
            target_col (str): Name of column whose terms need to be extracted.
            id_col (str): Unique ID column. Primary key.
            processed_table (str): Table name where you want to store the processed data. (In aws 3-node, pps_quality). If it doesn't exist then this module will create it for you.
            block_size (int, optional): Size of batches. Defaults to 5000.
            verbose (bool, optional): Print intermediate outputs. Defaults to False.
        """
        for i in range(1, 21):
            df[f'term_{i}'] = np.nan
            df[f'term_{i}'] = df[f'term_{i}'].astype(str)
            df[f'score_{i}'] = np.nan
            df[f'count_{i}'] = np.nan

        num_blocks = int(len(df) / block_size) + 1

        start = 0
        end = min(len(df), block_size)

        print(f'Total blocks: {num_blocks}\n\n')

        for block in range(num_blocks):
            print(f'Block: {block+1}')
            print(f'Range: {start}:{end}')

            # Using apply method
            result = df[start:end].progress_apply(
                lambda row: self.get_terms_using_pyate_on_df(row, target_col, id_col, verbose), axis=1)

            if verbose:
                display(result)

            status = self.load_to_vertica_database(
                result, self.schema, processed_table)
            del result

            # Using nlp.pipe method (multi-processing)

            # batch = df[start:end].copy()
            # batch.reset_index(inplace=True, drop=True)
            # texts = batch[target_col].to_list()

            # for counter, doc in tqdm(enumerate(self.nlp.pipe(texts))):
            #     topn_terms = doc._.combo_basic.sort_values(
            #         ascending=False).head(20)
            #     topn_terms = topn_terms.reset_index()

            #     terms = topn_terms['index'].to_list()
            #     scores = topn_terms.loc[:, 0].to_list()

            #     i = 1
            #     for term, score in zip(terms, scores):
            #         batch.loc[counter, f'term_{i}'] = term
            #         batch.loc[counter, f'score_{i}'] = score
            #         i += 1
            # # self.analyse_df(batch)
            # status = self.load_to_vertica_database(
            #     batch, self.schema, processed_table)

            start = end
            end = min(len(df), start+block_size)

    def restructure(self, df: pd.DataFrame, id_col: str, target_col: str,
                    cols_to_retain: List[str], top_n_terms: int = 20) -> pd.DataFrame:
        """Restructures the processed table to combine term_1 through term_{top_n_terms} and score_1 through score_{top_n_terms} into a single term and score column.
            Please use this for restructuring the processed_table of the apply function.

        Args:
            df (pd.DataFrame): Dataframe containing rows from the processed_table of the apply function.
            id_col (str): ID column
            target_col (str): Text column from which the terms were extracted.
            cols_to_retain (list[str]): Additional columns you would like to retain into your restructured table like ['business', 'ip2']
            top_n_terms (int, optional): Choose number of terms to keep (max:20). Defaults to 20.

        Returns:
            pd.DataFrame: Restructured dataframe with single term and score columns. Plus word_length column to signify it's n-gram length.
        """

        term_cols = []
        score_cols = []
        count_cols = []

        for i in range(1, top_n_terms + 1):
            term_cols += [f'term_{i}']
            score_cols += [f'score_{i}']
            count_cols += [f'count_{i}']

        dfs = []

        if len(cols_to_retain) == 0:
            cols_to_retain = [id_col, target_col]

        for term_col, score_col, count_col in zip(term_cols, score_cols, count_cols):
            reshaped_df = df.loc[:, cols_to_retain +
                                 [term_col, score_col, count_col]].copy()
            reshaped_df.columns = cols_to_retain + ['term', 'score', 'count']

            dfs.append(reshaped_df)

        reshaped = pd.concat(dfs, ignore_index=True, sort=False)
        reshaped.dropna(subset=['term', 'score', 'count'], inplace=True)
        reshaped['word_length'] = reshaped['term'].str.strip().str.count(' ') + 1

        return reshaped

    def run_ews_pipeline(self, ip2s, from_date, to_date, target_col='case_subject', id_col='case_id', processed_table='EWS_PYATE_SUBJECT_PROCESSED', restructured_table='EWS_PYATE_SUBJECT_PROCESSED_RESTRUCTURED', block_size=1000):
        try:
            for ip2 in ip2s:
                print(f"\nProcessing ip2 = {ip2}\n")
                query = f"""SELECT case_id, case_closed_date, ip2, case_subject, business 
                            from {self.schema}.RDR_MASTER 
                            where date(case_closed_date) >= '{from_date}'
                            and date(case_closed_date) <= '{to_date}'
                            and ip2 = '{ip2}' 
                            and {id_col} not in (select {id_col} from  {self.schema}.{processed_table})
                            order by ip2, case_closed_date 
                            ;    
                        """
                with self.get_connection('aws3') as aws_conn:
                    print(f'Fetching data from {self.schema}.RDR_MASTER')
                    df = pd.read_sql(query, aws_conn)

                if len(df) == 0:
                    print(
                        f'All these case_ids are already present in {self.schema}.{processed_table}.')
                    continue

                self.analyse_df(df)

                # Clean case_subject
                if target_col == 'case_subject':
                    cleaned_col = target_col + '_cleaned'
                    print('Cleaning case_subject column')
                    df[cleaned_col] = df[target_col].progress_apply(
                        self.clean_subject)
                    target_col = cleaned_col

                self.apply(df, target_col, id_col, processed_table, block_size)

                query = f"""select * from {self.schema}.{processed_table}
                            where term_1 is not null
                            and term_1 <> 'empty' 
                            and {id_col} not in  (select distinct {id_col} from {self.schema}.{restructured_table})
                            order by ip2, case_closed_date;"""

                with self.get_connection() as conn:
                    df = pd.read_sql(query, conn)

                if len(df) == 0:
                    return f'All these case_ids are already present in {self.schema}.{restructured_table}.'

                final_reshaped_df = self.restructure(df, id_col, target_col, cols_to_retain=[
                    id_col, target_col, 'case_closed_date', 'ip2', 'business'])

                self.load_to_vertica_database(
                    final_reshaped_df, self.schema, restructured_table)

                target_col = 'case_subject'

            return 'Successful'
        except Exception as e:
            print(f'Could not run apply_pyate due to {e}')
            return 'Unsuccessful'

    def remove_equals(self, s):

        if len(additional_lst) != 0:
            res_dct = {additional_lst[i]: '' for i in range(0, len(additional_lst))}
            remove_patterns.update(res_dct)
    
        if(isinstance(s, str)):
            s = s.lower()
            for pattern, sub_with in self.remove_patterns.items():
                s = re.sub(pattern, sub_with, s)
        return s

    def clean_subject(self, s):
        if s.split('/') is not None and len(s.split('/')) > 2:
            s = ' '.join(s.split('/')[-2:])
        s = self.remove_equals(s)
        s = self.remove_equals(s)
        s = self.remove_equals(s)
        return s
    def remover(self, terms, scores, counts):
        dict_scores, dict_counts  = dict(zip(terms, scores)), dict(zip(terms, counts))
        terms.sort(key=lambda x: x.count(' '), reverse=True)

        new_trm =[]
        for i in terms:
	    subset_flag=False
	    if len(new_trm)>0:
	        for j in new_trm:
		    if i in j:
		        matched_str=j
		        subset_flag=True
		        break
	        if subset_flag:
		    if dict_scores[i]>dict_scores[matched_str]:
		        dict_scores[matched_str]=dict_scores[i]
		    if dict_counts[i]>dict_counts[matched_str]:
		        dict_counts[matched_str]=dict_counts[i]            
	        else:
		    new_trm.append(i)
	    else:
	        new_trm.append(i)

        return new_trm, [dict_scores[i] for i in new_trm], [dict_counts[i] for i in new_trm]
    def get_terms_using_pyate_on_df(self, row, text_col, id_col, verbose):
        if not isinstance(row[text_col], str):
            return row

        # if nlp is None:
        #     nlp = self.nlp

        try:
            # doc = self.nlp(row[text_col])

            # topn_terms = doc._.combo_basic.sort_values(
            #     ascending=False).head(20)

            topn_terms = Modified_PYATE.modified_combo_basic(
                row[text_col])

            if len(topn_terms) == 0:
                return row

            topn_terms = topn_terms.sort_values(
                by=[0], ascending=False).head(20)

            if verbose:
                display(topn_terms)

            topn_terms = topn_terms.reset_index()

            if verbose:
                display(topn_terms)

            terms = topn_terms['index'].to_list()
            scores = topn_terms.loc[:, 0].to_list()
            counts = topn_terms.loc[:, 'counts'].to_list()
	    terms, scores, counts=remover(terms, scores, counts)
		
            i = 1
            for term, score, count in zip(terms, scores, counts):
                row[f'term_{i}'] = term
                row[f'score_{i}'] = score
                row[f'count_{i}'] = count
                i += 1

            # del nlp
            # del doc

            return row
        except Exception as e:
            print(f'Failed to process due to {e} \ntext: {row[id_col]}')
            return row
