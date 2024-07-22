import pandas as pd
import numpy as np
from fast_bert.prediction import BertClassificationPredictor
from tqdm import tqdm
tqdm.pandas()


class NM():
    def __init__(self):
        self.xyz = 'xyz'
        self.rdr_ps_model_dict = {
            'IP1': 'ps_ip1_categorizer',
            'IP2': 'ps_ip2_categorizer',
            'IP3': 'ps_ip3_categorizer'
        }

        self.rdr_print_model_dict = {
            'IP1': 'print_ip1_categorizer',
            'IP2': 'print_ip2_categorizer',
            'IP3': 'print_ip3_categorizer'
        }

    @staticmethod
    def load_model(model_name: str, LABEL_PATH: str = None, MODEL_PATH: str = None, verbose: bool = False) -> BertClassificationPredictor:
        """Loads a neural miner model which can be used for predictions using other functions of this module.

        Args:
            model_name (str): Name of the model you want to load. Find in /home/ubuntu/Notebooks/fast-bert/models/ on the 48 server.
            LABEL_PATH (str, optional): Provide path for labels.csv for the given model. Leave blank if you don't know. . Defaults to None.
            MODEL_PATH (str, optional): Provide path for model_name provided. Leave blank if you don't know. Defaults to None.
            verbose (bool, optional): Enable/Disable print statements. Defaults to False.
        Returns:
            BertClassificationPredictor: Predictor object which can be passed to other functions of this module.
        """
        if LABEL_PATH is None:
            LABEL_PATH = f'/home/ubuntu/Notebooks/Transformer_Models/Inputs/{model_name}'

        if MODEL_PATH is None:
            MODEL_PATH = f'/home/ubuntu/Notebooks/fast-bert/models/{model_name}/model_out/'

        if verbose:
            print(f'\nLoading {model_name} for classification...')

        predictor = BertClassificationPredictor(
            model_path=MODEL_PATH,
            label_path=LABEL_PATH,  # location for labels.csv file
            multi_label=False,
            model_type='bert',
            do_lower_case=True)

        if verbose:
            print(f"{model_name} loaded and ready for prediction.")
        return predictor

    @staticmethod
    def predict_single_level(df: pd.DataFrame, text_col: str, target_col: str, predictor: BertClassificationPredictor) -> pd.DataFrame:
        """Used for all single level predictions like business and RDR's IPs.

        Args:
            df (pd.DataFrame): Dataframe you want to predict on
            text_col (str): Text column the prediction should happen on.
            target_col (str): Name of the column where predictions would be added.
            predictor (BertClassificationPredictor): Object obtained from load_model function

        Returns:
            pd.DataFrame: Adds 2 columns. target_col and {target_col}_confidence to the dataframe provided.
        """
        texts = df[text_col].to_list()
        results = predictor.predict_batch(texts)

        targets = []
        scores = []

        for i, result in enumerate(results):
            # text = texts[i]
            targets.append(result[0][0])
            scores.append(result[0][1])

        df[target_col] = targets
        df[target_col + '_confidence'] = scores

        return df

    @staticmethod
    def get_tmip_predictions(df_orig: pd.DataFrame, text_col: str, predictor: BertClassificationPredictor) -> pd.DataFrame:
        """Predicts 3 levels of TMIP from a pandas dataframe for the provided column using the BertClassificationPredictor object provided.
            Note: You can get the BertClassificationPredictor using the load_model function.

        Args:
            df_orig (pd.DataFrame): Pandas dataframe one wants the result to be added to.
            text_col (str): Column name of the text one wants to predict on.
            predictor (BertClassificationPredictor): Object from the load_model function of neural miner

        Returns:
            pd.DataFrame: Adds 4 columns to the provided dataframe: tmip1, tmip2, tmip3 and confidence_score. Overwrites them if already present.
        """
        print(f'Predicting tmips for {text_col}')
        print(f'Total records: {len(df_orig)}')

        df = df_orig.copy()
        texts = df[text_col].to_list()
        results = predictor.predict_batch(texts)

        ip1s = []
        ip2s = []
        ip3s = []
        scores = []
        for i, result in enumerate(results):
            text = texts[i]

            ip1, ip2, ip3 = result[0][0].split('__')
            score = result[0][1]

            ip1s.append(ip1)
            ip2s.append(ip2)
            ip3s.append(ip3)
            scores.append(score)

        df['tmip1'] = ip1s
        df['tmip2'] = ip2s
        df['tmip3'] = ip3s
        df['confidence_score'] = scores

        return df

    def get_rdr_predictions(self, df, id_col, sub_col, text_col, target_col, predictor=None, model_name=None, rows=None, skip_prep=None, dont_drop=None, print_flag=0, verbose=False):

        df.reset_index(drop=True, inplace=True)
        df.drop_duplicates(subset=[text_col, id_col], inplace=True)

        if print_flag == 1:
            text_col, df = self.prep_for_print(
                df, sub_col, text_col)

        if int(target_col[-1]) in [2, 3]:
            if skip_prep:
                if verbose:
                    print(
                        f'Skipping prep. Assuming text_for_{target_col} exists.')
                text_col = f'text_for_{target_col}'
            else:
                df, text_col = self.prep_for_classification(
                    df, text_col, target_col, verbose=verbose)

        if predictor is None:
            if model_name:
                predictor = self.load_model(model_name, verbose=verbose)
            elif print_flag == 1:
                predictor = self.load_model(
                    self.rdr_print_model_dict[target_col], verbose=verbose)
            else:
                predictor = self.load_model(
                    self.rdr_ps_model_dict[target_col], verbose=verbose)

        df = self.predict_please(df, id_col=id_col, text_col=text_col,
                                 target_col=target_col, predictor=predictor, rows=rows, verbose=verbose)

        if int(target_col[-1]) in [2, 3]:
            if skip_prep or dont_drop:
                if verbose:
                    print(f'Not dropping {text_col}.')
            else:
                df = df.drop([f'text_for_{target_col}'], axis=1)

        # analyse_df(df)
        return df

    @staticmethod
    def prep_for_print(df, sub_col, text_col):
        sub_plus_notes = 'sub_plus_notes'
        df[sub_plus_notes] = df[[sub_col, text_col]].apply(
            lambda x: ' __subject__ '.join(x.map(str)), axis=1)
        return sub_plus_notes, df

    def prep_for_classification(self, df, text_col, target_col, verbose: bool = False):
        if verbose:
            print(f'\nPreparing data for {target_col} classification...')
        prev_level = None

        if int(target_col[-1]) in [2, 3]:
            prev_level = int(target_col[-1]) - 1
            df = df.apply(lambda row: self.add_col_to_text(
                row, text_col, f'IP{prev_level}', target_col), axis=1)
            text_col = f'text_for_{target_col}'

        df[text_col] = df[text_col].apply(self.remove_blanks)
        df.dropna(subset=[text_col], inplace=True)
        if verbose:
            print('Preparation complete.')
        return df, text_col

    @staticmethod
    def add_col_to_text(row, text_col, to_add_col, target_col):
        if isinstance(row[text_col], str) and isinstance(row[to_add_col], str):
            row[f'text_for_{target_col}'] = f'__{to_add_col}__ {row[to_add_col]} __{to_add_col}__ {row[text_col]}'
            return row
        else:
            row[f'text_for_{target_col}'] = np.nan
            return row

    @staticmethod
    def remove_blanks(s):
        if isinstance(s, str) and len(s) > 0:
            return s

    def predict_please(self, df, id_col, text_col, target_col, predictor, rows=None, verbose=False):

        if predictor:
            if rows:
                texts = list(df.loc[:rows-1, text_col])
            else:
                texts = list(df[text_col])

            predictions = {}

            if verbose:
                print(f'\nClassifying into {target_col}s...')
            results = predictor.predict_batch(texts)

            if len(results) == len(texts):
                if verbose:
                    print(f'All were predicted')
            k = 0

            for result in results:
                predictions[texts[k]] = result
    #             clear_output(wait=True)
    #             print(f'{k} / {len(texts)}')
                k += 1
            if verbose:
                print(f'k = {k}')
                print(f'Classified {len(predictions.keys())} / {len(texts)}')
                print('\nAppending IPs to Table...')

            df[target_col] = df.apply(lambda row: self.assign_ip(
                row, text_col, predictions), axis=1)
            df[f'{target_col}_confidence'] = df.apply(
                lambda row: self.assign_prob(row, text_col, predictions), axis=1)

            if verbose:
                print(f'{target_col} classification completed successfully!')

            return df
        else:
            print('Predictor not loaded.')
            return df

    @staticmethod
    def assign_ip(row, text_col, predictions):
        if row[text_col] in predictions.keys():

            return predictions[row[text_col]][0][0]
        else:
            return np.nan

    @staticmethod
    def assign_prob(row, text_col, predictions):
        if row[text_col] in predictions.keys():
            return predictions[row[text_col]][0][1]
        else:
            return np.nan
