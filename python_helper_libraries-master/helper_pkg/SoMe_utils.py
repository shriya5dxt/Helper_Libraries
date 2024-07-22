from .neural_miner import NM
from . import Helper

import pandas as pd
import re
import numpy as np
from tqdm import tqdm
tqdm.pandas()

pd.set_option('mode.chained_assignment', None)


class SoMe(Helper, NM):
    def __init__(self):
        Helper.__init__(self)
        NM.__init__(self)

        self.instant_ink_pattern = re.compile('instant ink(.*)')

    def run_tmip_prediction_pipeline(self, df: pd.DataFrame, text_col: str, queue_col: str = None, business_col: str = None, id_col: str = 'Conversation ID', master_table: str = 'SOME_TMIP_MASTER', stg_table: str = 'SOME_STG_TMIP') -> str:
        """Predicts the TMIPs from the given social media chats in a pandas dataframe, loads it into the provided staging table and merges it into the master table.

        Args:
            df (pd.DataFrame): DataFrame containing SoMe Assisted Chats
            text_col (str): Column containing the conversations.
            queue_col (str, optional): Used to scrape incoming_channel and source_language columns. Ignore if those are already present in your dataframe. Defaults to None.
            business_col (str, optional): Used to apply relevant models according to the business the conversation belong to. Business will be inferred if this arg is not provided. Defaults to None.
            id_col (str, optional): Id column. Defaults to 'Conversation ID'.
            master_table (str, optional): Final table the results need to be merged into. Defaults to 'SOME_TMIP_MASTER'.
            stg_table (str, optional): Staging table the results will be loaded into. Defaults to 'SOME_STG_TMIP'.

        Returns:
            str: 'Successful'/'Unsuccessful' and 'Partially Successful' if data loaded to staging table but merge failed.
        """
        # Cleaning conversations
        print('Cleaning SoMe Conversations')
        df = self.clean_SoMe_conversations(df, text_col, queue_col)
        print('\tCleaning complete!')

        # Predicting business
        if business_col is None:
            print(
                '\nPredicting business column from text since no business_col was provided.')
            self.business_predictor = self.load_model(
                'sm_business_categorizer')
            results = self.predict_single_level(
                df, text_col, target_col='business', predictor=self.business_predictor)
            print(results.business.value_counts())
        else:
            print('\nSkipping business prediction because business_col was provided')
            results = df

        # Processing PS records
        print('\nProcessing Personal Systems records')
        ps = results.loc[results.business == 'Personal Systems', :]

        if len(ps) > 0:
            self.analyse_df(ps)

            self.ps_tmip_predictor = self.load_model('ps_tmip_categorizer')
            self.ps_tmip_results = self.get_tmip_predictions(
                ps, text_col, self.ps_tmip_predictor)
            self.analyse_df(self.ps_tmip_results)
            print('\tCompleted Personal Systems Processing')
        else:
            self.ps_tmip_results = ps
            print('\tNo PS records to process')

        # Processing Print records
        print('\nProcessing Printing records')
        printing = results.loc[results.business.isin(['Print', 'Printing']), :]
        if len(printing) > 0:
            self.analyse_df(printing)

            self.print_tmip_predictor = self.load_model(
                'print_tmip_categorizer')
            self.print_tmip_results = self.get_tmip_predictions(
                printing, text_col, self.print_tmip_predictor)
            self.analyse_df(self.print_tmip_results)
            print('\tCompleted print Processing')
        else:
            self.print_tmip_results = printing
            print('\tNo Print records to process')

        self.others = results.loc[results.business == 'None', :]

        # Combining final results
        print('\nCombining final results')
        self.final = pd.concat([self.ps_tmip_results, self.print_tmip_results, self.others],
                               ignore_index=True, sort=False)

        # Processing final dataframe for loading to database
        print('\nProcessing final dataframe for loading to database')
        self.final[id_col] = self.final[id_col].astype(int)
        self.final[id_col] = self.final[id_col].astype(object)
        self.final.rename(columns={id_col: 'Case ID',
                                   text_col: 'P_Des',
                                   'Status': 'status',
                                   'Closed Date': 'closed_date',
                                   'Published Date': 'published_date'}, inplace=True)

        # Loading results to master table
        print('\nDropping current staging table')
        drop_stg_table = f"""DROP TABLE IF EXISTS {self.schema}.{stg_table};"""
        self.run_query(drop_stg_table)

        print('\nLoading results to staging table')
        status = self.load_to_vertica_database(
            self.final, self.schema, stg_table)

        print('\nMerging into the master table')
        try:
            self.merge_query = f"""
                        merge into {self.schema}.{master_table} m
                        using  {self.schema}.{stg_table} stg
                        on m."case id" = stg."case id"
                        when matched then update
                            set tmip1 = stg.tmip1,
                                tmip2 = stg.tmip2,
                                tmip3 = stg.tmip3,
                                confidence_score = stg.confidence_score
                        when not matched then insert(
                            "Case ID",
                            P_Des,
                            business,
                            business_confidence,
                            tmip1,
                            tmip2,
                            tmip3,
                            confidence_score,
                            incoming_channel,
                            source_language,
                            status,
                            closed_date,
                            published_date
                        ) values (
                            stg."Case ID",
                            stg.P_Des,
                            stg.business,
                            stg.business_confidence,
                            stg.tmip1,
                            stg.tmip2,
                            stg.tmip3,
                            stg.confidence_score,
                            stg.incoming_channel,
                            stg.source_language,
                            stg.status,
                            stg.closed_date,
                            stg.published_date
                        );
                    """
            self.run_query(self.merge_query)
        except Exception as e:
            print(
                f'Failed in merging the results into the master table due to {e}')
            status = 'Partially successful'

        return status

    def clean_SoMe_conversations(self, df, text_col, queue_col=None, id_col='Case ID'):
        if queue_col is not None:
            new_cols = df[queue_col].str.split(' - ', expand=True)
            df['country_actual'] = new_cols[0]

            df['incoming_channel'] = new_cols[1]
            df['incoming_channel'] = df['incoming_channel'].str.replace(
                "'", "")
            df['incoming_channel'] = df['incoming_channel'].str.replace(
                "}", "")
            df['incoming_channel'].value_counts(dropna=False)
            df['incoming_channel'].fillna(value='SoMe WhatsApp', inplace=True)

            df.country_actual = df.country_actual.str.replace("'", "")
            df.country_actual = df.country_actual.str.replace("{", "")
            df.country_actual = df.country_actual.str.replace("}", "")
            df.country_actual = df.country_actual.str.replace(
                " - SoMe WhatsApp", "")
            df.country_actual = df.country_actual.str.replace(
                " – SoMe WhatsApp", "")
            df.country_actual = df.country_actual.str.replace("_UK", "")
            df.country_actual = df.country_actual.str.replace("_EMEA", "")
            df.country_actual = df.country_actual.str.replace("_ES", "")

            df['country_actual'].value_counts(dropna=False)

            df.rename({'country_actual': 'source_language'},
                      inplace=True, axis=1)
        df[id_col] = df[id_col].astype(int)
        df[id_col] = df[id_col].astype(object)
        df = self.synonym(df, text_col)
        df[text_col] = df[text_col].apply(self.remove_starting_instant_ink)
        return df

    def remove_starting_instant_ink(self, text):
        if not isinstance(text, str):
            return np.nan

        res = re.match(self.instant_ink_pattern, text)
        if res:
            return res.group(1)
        else:
            return text

    def synonym(self, df, text_col):
        df[text_col] = df[text_col].str.lower()
        df[text_col] = df[text_col].str.replace('web cam', 'webcam')
        df[text_col] = df[text_col].str.replace('wi fi', 'wifi')
        df[text_col] = df[text_col].str.replace('wirless', 'wireless')
        df[text_col] = df[text_col].str.replace('wirelless', 'wireless')
        df[text_col] = df[text_col].str.replace('wex', 'warranty extension')
        df[text_col] = df[text_col].str.replace('warrenty', 'warranty')
        df[text_col] = df[text_col].str.replace('warranties', 'warranty')
        df[text_col] = df[text_col].str.replace('waranty', 'warranty')
        df[text_col] = df[text_col].str.replace('temperatures', 'temperature')
        df[text_col] = df[text_col].str.replace('printers', 'printer')
        df[text_col] = df[text_col].str.replace('cartridges', 'cartridge')
        df[text_col] = df[text_col].str.replace('Web services', 'webservices')
        df[text_col] = df[text_col].str.replace('leds', 'led')
        df[text_col] = df[text_col].str.replace('conection', 'connection')
        df[text_col] = df[text_col].str.replace('batteries', 'battery')
        df[text_col] = df[text_col].str.replace('drivers', 'driver')
        df[text_col] = df[text_col].str.replace(
            "https://support.hp.com/us-en/document/c", ' ')
        df[text_col] = df[text_col].str.replace("instant-ink", 'instant ink ')
        df[text_col] = df[text_col].str.replace("instantink", 'instant ink ')
        df[text_col] = df[text_col].str.replace("web services", 'instant ink ')
        df[text_col] = df[text_col].str.replace("web service", 'instant ink ')
        df[text_col] = df[text_col].str.replace("enrollment", 'enrol')
        df[text_col] = df[text_col].str.replace("enroll", 'enrol')
        df[text_col] = df[text_col].str.replace("sprocket", 'printer')
        df[text_col] = df[text_col].str.replace("laserjet", 'printer')
        df[text_col] = df[text_col].str.replace("deskjet", 'printer')
        df[text_col] = df[text_col].str.replace("officejet", 'printer')
        df[text_col] = df[text_col].str.replace("laser jet", 'printer')
        df[text_col] = df[text_col].str.replace("desk jet", 'printer')
        df[text_col] = df[text_col].str.replace("office jet", 'printer')
        df[text_col] = df[text_col].str.replace("envy", 'printer')
        df[text_col] = df[text_col].str.replace("officejet", 'printer')
        df[text_col] = df[text_col].str.replace("notebook", 'computer')
        df[text_col] = df[text_col].str.replace("pavilion", 'computer')
        df[text_col] = df[text_col].str.replace("spectre", 'computer')
        df[text_col] = df[text_col].str.replace("omen", 'computer')
        df[text_col] = df[text_col].str.replace("laptop", 'computer')
        df[text_col] = df[text_col].str.replace("pc", 'computer')
        df[text_col] = df[text_col].str.replace("stream", 'computer')
        df[text_col] = df[text_col].str.replace("new case", 'case')
        return df
