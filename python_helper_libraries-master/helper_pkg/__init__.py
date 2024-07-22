from datetime import datetime
from io import StringIO
import vertica_python
import pymssql
import cx_Oracle
import pandas as pd
import numpy as np
import re
import platform

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import requests
import json
from typing import Dict, List, Tuple, Any, Iterator, Union
from IPython.display import display

from helper_pkg.connections import Connections


class Helper:

    def __init__(self):
        super().__init__()
        self.time_read_string = "%Y-%m-%dT%H:%M:%S"

        self.schema = 'PPS_QUALITY'
        self.translation_tracking_table = 'TRACK_TEXT_TRANSLATION_CALLS'

        if platform.system() != 'Windows':
            from whatthelang import WhatTheLang
            self.wtl = WhatTheLang()

        self.conns = Connections()
        self.smtp_mail_server = "smtp3.hp.com"
        self.smtp_port = "25"

        self.translation_url = "https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to=en"
        self.headers = {
            'Content-Type': 'application/json',
            'Ocp-Apim-Subscription-Key': '4279442d1fbc4a8180f4c1c718995e9a'
        }

        self.fields_pattern = re.compile('select (.*) from')

        self.join_spill_enabling = """/* Make this query memory optimized (use more disk than memory) */
                        SELECT add_vertica_options('EE', 'ENABLE_JOIN_SPILL');
                      """

    def get_connection(self, to: str = 'aws3', total_tries: int = 5, verbose: bool = False,
                       recipients: List[str] = None, send_mail: bool = True):
        """Helps get the connection to a required database using our team's service accounts. To get all connection keys pass 'xyz' as your argument.

        Args:
            to (str, optional): Connection key to identify the connection. Defaults to 'aws3'.
            total_tries (int, optional): Maximum retries in case the connection fails. Defaults to 5.
            verbose (bool, optional): Set True for debugging. Defaults to False.
            recipients (List[str], optional): Recipients of mail in case connection fails after max_retries. Defaults to ['chirag.mehta@hp.com', 'shriya.dikshith1@hp.com'].
            send_mail (bool, optional): Set True to send mail when connection fails after maximum retries. Defaults to True

        Returns:
            connection(vertica/sql/oracle depending on connection key provided): Returns a connection to the database of the connection key.
        """
        if recipients is None:
            recipients = ['chirag.mehta@hp.com', 'shriya.dikshith1@hp.com']
        counter = 0
        while counter < total_tries:
            try:
                if to in self.conns.vertica_connections.keys():
                    return vertica_python.connect(**self.conns.vertica_connections[to])

                elif to in self.conns.sql_connections.keys():
                    print(self.conns.sql_connections.keys())
                    host = self.conns.sql_connections[to]['host']
                    port = self.conns.sql_connections[to]['port']
                    host += f':{port}'
                    username = self.conns.sql_connections[to]['username']
                    password = self.conns.sql_connections[to]['password']
                    database = self.conns.sql_connections[to]['database']

                    return pymssql.connect(host, username, password, database)

                elif to in self.conns.oracle_connections.keys():
                    host = self.conns.oracle_connections[to]['host']
                    port = self.conns.oracle_connections[to]['port']
                    username = self.conns.oracle_connections[to]['username']
                    password = self.conns.oracle_connections[to]['password']
                    database = self.conns.oracle_connections[to]['database']
                    dsn_tns = cx_Oracle.makedsn(host,
                                                port,
                                                service_name=database)

                    return cx_Oracle.connect(user=username, password=password, dsn=dsn_tns)
                else:
                    print(
                        'Invalid connection key passed. Please pass one of the following:', '\n')
                    print(
                        f'{list(self.conns.vertica_connections.keys()) + list(self.conns.sql_connections.keys()) + list(self.conns.oracle_connections.keys())}')
                    return None

            except Exception as e:
                if verbose:
                    print(f'Failed to connect to {to}: attempt {counter + 1}')
                counter += 1
                if counter < total_tries:
                    continue
                else:
                    if verbose:
                        print('Max retries reached. Breaking out of the loop.')
                    if send_mail:
                        self.send_mail(subject='Failed metrics_testfreak daily run',
                                       body=f'Failed due to {e}',
                                       recipients=recipients
                                       )
                    break

    @staticmethod
    def analyse_df(df: pd.DataFrame) -> None:
        """Helps get the DataFrame's info, head, tail and describe quickly. Does not return anything.

        Args:
            df (pandas.DataFrame): DataFrame you intend to analyse.
        """
        display(df.info(null_counts=True))
        display(df.head())
        display(df.tail())
        display(df.describe(include='all'))

    def convert_str_to_date(self, str_date):
        """Helps convert a string to a python datetime object. Can be helpful for datetime operations like finding difference in days between two dates.

        Args:
            str_date (str): String you wish to convert to a datetime object

        Returns:
            datetime.Date: Datetime object from the string provided.
        """
        return datetime.strptime(str_date, self.time_read_string)

    def convert_date_to_str(self, date_object):
        """Converts a datetime object to a string. Can be helpful for putting date filters in an sql query.

        Args:
            date_object (datetime.Date): Datetime object you wish to convert.

        Returns:
            str: Datetime object converted to iso format string.
        """
        return date_object.strftime(self.time_read_string)

    @staticmethod
    def get_dtypes_map(df, chunked):
        """Helps get a pandas datatype to sql datatype mapping of all columns in a given DataFrame.
        It is called inside create_table_builder function which in turn is called inside load_to_vertica_database function.

        Args:
            df (pandas.DataFrame): DataFrame you wish to get the mapping for.
        Returns:
            python dictionary: Python dictionary with column names as keys and their sql datatype as values.
        """
        dtype_map = {}
        for col, dtype in zip(df.columns, df.dtypes):
            dtype = str(dtype).lower()
            try:
                if 'object' in str(dtype) or 'string' in str(dtype):
                    if not chunked:
                        dtype_map.update(
                            {col: f'varchar({min(max([len(str(cell)) for cell in df[col]]) + 1000, 65000)})'})
                    else:
                        dtype_map.update(
                            {col: f'varchar(65000)'})
                elif 'float' in str(dtype):
                    dtype_map.update({col: 'float'})
                elif 'int' in str(dtype):
                    dtype_map.update({col: 'integer'})
                elif 'date' in str(dtype):
                    dtype_map.update({col: 'datetime'})
                elif 'bool' in str(dtype):
                    dtype_map.update({col: 'bool'})
                else:
                    print(f'datatype for {col} could not be resolved')
            except Exception as e:
                print(f'datatype for {col} could not be resolved due to {e}')

        return dtype_map

    def create_table_builder(self, df, table, chunked, schema=None):
        """Helps generate a create table query for the dataframe passed with the given table name inside Object's default schema ({self.schema}).
        It is called inside the load_to_vertica_database function if the relation was not found.

        Args:
            df (pandas.DataFrame): Pandas DataFrame you wish to create the table for.
            table (str): Name of the table
            schema (str, optional): Name of the schema. Defaults to '{self.schema}'.

        Returns:
            str: Create table query required to run for the given DataFrame.
        """
        if schema is None:
            schema = self.schema

        dtype_map = self.get_dtypes_map(df, chunked)

        query = f"""create table {schema}.{table} ("""
        counter = 0

        for column, dtype in dtype_map.items():
            if counter != 0:
                query += f""", "{column}" {dtype}"""
            else:
                query += f""" "{column}" {dtype}"""

            counter += 1
        # print(query)
        query += ');'
        return query

    @staticmethod
    def load_data_using_copy_command(df, schema, table, conn, output_stream):
        reject_table = table + '_REJECTED'
        cur = conn.cursor()
        cur.copy(
            f"COPY {schema}.{table} FROM STDIN parser fjsonparser()", output_stream)
        conn.commit()
        cur.execute('SELECT GET_NUM_REJECTED_ROWS ();')
        rows_rejected = cur.fetchall()
        rows_loaded = len(df.index) - rows_rejected[0][0]
        print(f'\tNumber of rows loaded: {rows_loaded}')
        print(f'\tNumber of rows rejected: {rows_rejected[0][0]}')
        return True if rows_loaded > 0 else False, rows_loaded, conn

    def load_to_vertica_database(self, df, schema, table, conn_key='aws3', conn=None, chunk_size=1000000):
        """Loads a pandas dataframe to a vertica database. By default loads it onto the aws3-node server. 
        If the table does not exist then its structure is created according to pandas dtypes. 
        If the table exists then rows are appended to the table. In case of a column mismatch between the dataframe and table, data is loaded only for the columns existing in the table already.

        Args:
            df (pandas.DataFrame): Dataframe that needs to be loaded onto the vertica database.
            schema (str): Schema in which the table needs to be loaded. Generally in {self.schema}
            table (str): Name of table the dataframe needs to be loaded on.
            conn_key (str, optional): A connection key supported by the get_connection function. If left blank, data is loaded onto the aws3-node server. Defaults to 'aws3'.
            conn(Connection object, optional): Vertica_python connection to the database. If provided, data is loaded using this connection. Defaults to None. 
            chunk_size (int, optional): If len(df) exceed chunk_size then it'll be loaded in batches of {chunk_size} rows. Defaults to 1000000

        Returns:
            str: 'successful' if data was loaded successfully else 'unsuccessful'.
        """

        dfs = []
        statuses = []
        chunked = False
        total_rows_loaded = 0

        if len(df.index) > chunk_size:
            chunked = True
            number_of_chunks = int(len(df.index) / chunk_size) + 1
            print(
                f'Dividing provided dataframe into chunks of {chunk_size} rows...')
            print(f'Number of chunks: {number_of_chunks}')
            dfs = np.array_split(df, number_of_chunks)

        else:
            dfs.append(df)

        if conn is None:
            conn = vertica_python.connect(
                **self.conns.vertica_connections[conn_key])

        for chunk in dfs:
            output_stream = StringIO()

            try:
                chunk.to_json(output_stream, orient='records',
                              date_format='iso', force_ascii=False, lines=True)

            except Exception as e:
                print(f"""Failed in loading chunk to memory: due to {e}
                        Try reducing the chunk size.""")
                continue
            output_stream.seek(0)

            try:
                status, rows_loaded, conn = self.load_data_using_copy_command(
                    chunk, schema, table, conn, output_stream)

                statuses.append(status)
                total_rows_loaded += rows_loaded
            except vertica_python.errors.Error as err:
                if (type(err) is vertica_python.errors.MissingRelation) or ('Sqlstate: 42V01' in err.args[0]):
                    print(f'Creating table since {schema}.{table} not found')
                    query = self.create_table_builder(chunk, table, chunked)
                    print(f'\nCreate table query: \n{query}')
                    cur = conn.cursor()
                    cur.execute(query)
                    conn.commit()
                    status, rows_loaded, conn = self.load_data_using_copy_command(
                        chunk, schema, table, conn, output_stream)

                    statuses.append(status)
                    total_rows_loaded += rows_loaded

            del output_stream

        print(f'\n\nTotal Rows loaded: {total_rows_loaded}')
        print(f'Total rejected rows: {len(df) - total_rows_loaded}')

        return 'Successful' if all(statuses) else 'Unsucessful'

    def detect_language_on_row(self, row, text_col, lang_col, dh, engine='spacy', nlp=None, verbose=False):
        try:
            self.counter += 1
            dh.update(f'{self.counter} / {self.num_records}')

            if not isinstance(row[text_col], str):
                row[lang_col] = 'Failed because not a string'
                row[lang_col + '_score'] = np.nan
                return row

            if engine == 'whatthelang':
                # from whatthelang import WhatTheLang
                # self.wtl = WhatTheLang()
                lang = self.wtl.pred_prob(row[text_col])
                row[lang_col] = lang[0][0][0]
                row[lang_col + '_score'] = lang[0][0][1]
                return row
            elif engine == 'spacy':
                try:
                    doc = nlp(row[text_col])
                    if doc._.languages:
                        row[lang_col] = str(doc._.languages[0])
                        row[lang_col + '_score'] = doc._.language_score[doc._.language[0]]
                    else:
                        row[lang_col] = 'unknown'
                        row[lang_col + '_score'] = np.nan
                except Exception as e:
                    if verbose:
                        print(f'Failed for {row[text_col]} due to {e}')
                    row[lang_col] = 'utf-8 error'
                    row[lang_col + '_score'] = np.nan

            return row
        except Exception as e:
            row[lang_col] = f'Failed due to {e}'
            row[lang_col + '_score'] = np.nan
            return row

    def detect_language_on_df(self, df, text_col, lang_col='language', engine='whatthelang', verbose=False):
        """Helps detect the language of all records in a given column of a pandas dataframe. 
        It adds 2 new columns: language and language_score(confidence).

        Args:
            df (pandas.DataFrame): DataFrame containing the text column whose language needs to be detected.
            text_col (str): Name of the text column whose language needs to be detected.
            lang_col (str, optional): Name of the language column you wish to add to the returning dataframe. Defaults to 'language'.
            engine (str, optional): Engine used for language_detection. Defaults to 'whatthelang'.

        Returns:
            pandas.DataFrame: Pandas DataFrame containing 2 additional columns containing language and language_score. 

        """
        self.counter = 0
        self.num_records = len(df)
        dh = display(f'{self.counter} / {self.num_records}', display_id=True)
        self.nlp = None
        if engine == 'spacy':
            import spacy
            from spacy_cld import LanguageDetector
            spacy.prefer_gpu()

            self.nlp = spacy.load('en_core_web_lg', disable=[
                'parser', 'tagger', 'ner'])
            self.nlp.add_pipe(self.nlp.create_pipe("sentencizer"))
            print('Loaded spacy model for language detection.')
            try:
                language_detector = LanguageDetector()
                self.nlp.add_pipe(language_detector)
            except Exception as e:
                print(f'Could not initialize language_detector due to {e}')
                print('Ignore if it says language detector is already set.')
        df = df.apply(lambda row: self.detect_language_on_row(
            row, text_col, lang_col, dh=dh, engine=engine, nlp=self.nlp, verbose=verbose), axis=1)

        display(pd.DataFrame(df[lang_col].value_counts()))
        return df

    def text_translate_on_row(self, row, text_col, translated_col, conn, dh):
        """Takes a row & text column to return english translation of the same in row's translated_col.

        Args:
            row (pandas.Series): Dataframe's row currently being processed
            text_col (str): Text column's name
            translated_col (str): Translated text column's name
            conn (vertica_python.Connection): Connection to vertica server with TextTranslate Function
            dh (display_id): Display id to update current status

        Returns:
            pandas.Series: Row with translation in translated_col
        """

        self.counter += 1
        dh.update(f'Status: {self.counter} / {self.num_records}')

        if isinstance(row[text_col], str):
            query = f"""select TextTranslate('{row[text_col].replace("'", "''")}') OVER();"""
            translated_row = pd.read_sql(query, conn)
            row[translated_col] = translated_row['Translated_Text'].values[0]
        else:
            row[translated_col] = 'Failed to translate because not a string'

        return row

    def text_translate_on_df(self, df, text_col, user, purpose='Not mentioned', translated_col='translated_text',
                             conn_key='aws3', ask_answer=True):
        """Translates the text in a given dataframe and stores it in translated_col provided using the Microsoft's Translator Service embedded on top of our Vertica Servers.

        Args:
            df (pandas.DataFrame): DataFrame containing text column to be translated
            text_col (str): Name of the text column to be translated
            user (str): Your hp email id. Ex: chirag.mehta@hp.com
            purpose (str, optional): Please mention the purpose of translation. Will ease tracking in case of issues. Defaults to 'Not mentioned'.
            translated_col (str, optional): Column to store translated text in. Defaults to 'translated_text'.
            conn_key (str, optional): Connection key to get connections using helper's get_connections function. Defaults to 'aws3'.
            ask_answer (bool, optional): When True it will prompt for an input to confirm total characters. Set to False for automated loads. Defaults to True.

        Returns:
            pandas.DataFrame: Pandas DataFrame with translated text added to translated_col
        """
        self.counter = 0
        self.num_records = len(df)

        total_characters = df[text_col].apply(len).sum()

        if ask_answer:
            answer = input(
                f'Total characters: {total_characters}.\nDo you want to continue?(yes/no)')
        else:
            print(
                f'Total characters: {total_characters}. Answer set to yes since ask_answer = False.')
            answer = 'yes'

        if answer != 'yes':
            print('Did not go through with the translations.')
            return df

        dh = display(f'{self.counter} / {self.num_records}', display_id=True)

        with self.get_connection(to=conn_key) as aws_conn:
            df = df.apply(lambda row: self.text_translate_on_row(
                row, text_col, translated_col, conn=aws_conn, dh=dh), axis=1)

        tracking_record = pd.DataFrame.from_dict({'id': [''],
                                                  'characters': [total_characters],
                                                  'user': [user],
                                                  'time': [datetime.now()],
                                                  'purpose': [purpose]
                                                  })

        status = self.load_to_vertica_database(
            tracking_record, self.schema, self.translation_tracking_table)
        print(f'Loading status to translation tracking table: {status}')
        return df

    def text_translate_api_on_row(self, row, text_col, translated_col, conn, dh, stg_table='TRANSLATION_STG_PYTHON'):
        """Takes a row & text column to return english translation of the same in row's translated_col.

        Args:
            row (pandas.Series): Dataframe's row currently being processed
            text_col (str): Text column's name
            translated_col (str): Translated text column's name
            conn (vertica_python.Connection): Connection to vertica server with TextTranslate Function
            dh (display_id): Display id to update current status

        Returns:
            pandas.Series: Row with translation in translated_col
        """

        self.counter += 1
        dh.update(f'Status: {self.counter} / {self.num_records}')

        if isinstance(row[text_col], str):
            try:
                request_dict = {'Text': row[text_col]}
                request_list = [request_dict]
                request_json = json.dumps(request_list)

                payload = request_json
                response = requests.request(
                    "POST", self.translation_url, headers=self.headers, data=payload)
                rsp = response.json()

                if response.ok:
                    row[translated_col] = rsp[0]['translations'][0]['text']
                    # _ = self.load_to_vertica_database(
                    #     row, self.schema, stg_table)
                else:
                    row[
                        translated_col] = f"""Translation failed due to Code: {rsp['error']['code']} Message: {rsp['error']['message']}"""
            except Exception as e:
                row[translated_col] = f'Translation failed due to {e}'

        else:
            row[translated_col] = np.nan

        return row

    def text_translate_api_on_df(self, df, text_col, user, purpose='Not mentioned', translated_col='translated_text',
                                 conn_key='aws3', ask_answer=True):
        """Translates the text in a given dataframe and stores it in translated_col provided using the Microsoft's Translator Service by making a POST call.

        Args:
            df (pandas.DataFrame): DataFrame containing text column to be translated
            text_col (str): Name of the text column to be translated
            user (str): Your hp email id. Ex: chirag.mehta@hp.com
            purpose (str, optional): Please mention the purpose of translation. Will ease tracking in case of issues. Defaults to 'Not mentioned'.
            translated_col (str, optional): Column to store translated text in. Defaults to 'translated_text'.
            conn_key (str, optional): Connection key to get connections using helper's get_connections function. Defaults to 'aws3'.
            ask_answer (bool, optional): When True it will prompt for an input to confirm total characters. Set to False for automated loads. Defaults to True.

        Returns:
            pandas.DataFrame: Pandas DataFrame with translated text added to translated_col
        """
        self.counter = 0
        self.num_records = len(df)

        total_characters = df[text_col].apply(len).sum()

        if ask_answer:
            answer = input(
                f'Total characters: {total_characters}.\nDo you want to continue? (yes/no) ')
        else:
            print(
                f'Total characters: {total_characters}. Answer set to yes since ask_answer = False.')
            answer = 'yes'

        if answer != 'yes':
            print('Did not go through with the translations.')
            return df

        dh = display(f'{self.counter} / {self.num_records}', display_id=True)

        with self.get_connection(to=conn_key) as aws_conn:
            df = df.apply(lambda row: self.text_translate_api_on_row(
                row, text_col, translated_col, conn=aws_conn, dh=dh), axis=1)

        total_characters = df.loc[~df[text_col].str.contains(
            'Translation Failed due to'), text_col].apply(len).sum()
        if total_characters > 0:
            tracking_record = pd.DataFrame.from_dict({'id': [''],
                                                      'characters': [total_characters],
                                                      'user': [user],
                                                      'time': [datetime.now()],
                                                      'purpose': [purpose]
                                                      })

            status = self.load_to_vertica_database(
                tracking_record, self.schema, self.translation_tracking_table)
            print(f'Loading status to translation tracking table: {status}')
        else:
            print('Not tracked since no translations happened.')

        return df

    def sentence_similarity_on_df(self, df, text_col, corpus, output_col='category', model=None,
                                  embedding='bert-base-nli-stsb-mean-tokens'):
        """Given a dataframe and text column get the most similar meaning sentence from a given corpus.
        Note: Results will be significantly faster on a GPU server. CPU servers like 116 may take a longer time to process. 

        Args:
            df (pandas.DataFrame): Pandas DataFrame containing the data
            text_col (str): Name of text column that needs to be processed.
            corpus (list(str)): List of target strings with which the comparisons need to be made.
            model (SentenceTransformer, optional): Object of SentenceTransformer Class from the sentence_transformers package. Defaults to None.
            embedding (str, optional): Name of a token set specified in sentence_transformers documentation. Defaults to 'bert-base-nli-stsb-mean-tokens'.

        Returns:
            pandas.DataFrame: Contains two added columns. output_col contains the top matching string between text_col and corpus and '{output_col}_distance' contains the distance between the two.
        """
        import scipy
        import sentence_transformers

        if model is None:
            model = sentence_transformers.SentenceTransformer(embedding)

        corpus_embeddings = model.encode(corpus)
        queries = df[text_col].to_list()
        # categories = df[target_col].to_list()
        query_embeddings = model.encode(queries)

        categories = []
        final_distances = []
        for query, query_embedding in zip(queries, query_embeddings):
            distances = scipy.spatial.distance.cdist(
                [query_embedding], corpus_embeddings, "cosine")[0]

            results = zip(range(len(distances)), distances)
            results = sorted(results, key=lambda x: x[1])

            category = corpus[results[0][0]].strip()
            final_distance = 1 - results[0][1]

            categories.append(category)
            final_distances.append(final_distance)

        df[output_col] = categories
        df[output_col + '_distance'] = final_distances

        return df

    @staticmethod
    def extract_matches(text, choices, threshold):
        if isinstance(text, str):
            from fuzzywuzzy import process
            closest_match = process.extractOne(text, choices)
            if closest_match[1] < threshold:
                print(text, ': ', closest_match)
                return np.nan, np.nan

            return closest_match
        else:
            return np.nan, np.nan

    def get_closest_match_on_df(self, df, text_col, choices, output_col='closest', threshold=80):
        """Helps get the closest match (based on character matching - Levenshtein distance for fuzzy string comparisons) between a string and a given list of choices.
        The threshold determines what a close match constitutes. 
        Note: If you don't want a cutoff then set threshold to 0. Will provide a result for every string in the text_col.

        Args:
            df (pandas.DataFrame): Pandas DataFrame which you wish to process.
            text_col (str): Name of the column.
            choices (list(str)): List of all the possible choices you wish to compare with.
            output_col (str, optional): Name of added column. Defaults to 'closest'.
            threshold (int, optional): Number between 0-100 to determine cutoff for closest. If no choice has a similarity score > threshold then np.nan is returned. Defaults to 80.

        Returns:
            pandas.DataFrame: Pandas Dataframe with 2 added columns. 'closest' for the closest matches string out of the choices and 'closest_score' with their respective scores.
        """
        texts = df[text_col]
        closest_matches = [self.extract_matches(
            text, choices, threshold) for text in texts]

        closest_strings = [match[0] for match in closest_matches]
        scores = [match[1] for match in closest_matches]

        df[output_col] = closest_strings
        df[output_col + '_score'] = scores

        return df

    def send_mail(self, subject, body, recipients, from_addr_list=None):
        """Helps send mails from a given list of emails to a given list of recipients using the MIMEMultipart class of email.mime package of python.

        Args:
            subject (str): Subject of the email.
            body (str): Content of the email.
            recipients (list(str)): List of email addresses you wish to mail to.
            from_addr_list (list, optional): List of email addresses from whom the mail will be triggered. Defaults to ['qms_tools@hp.com'].

        Returns:
            str: Success/Failure message.
        """
        if from_addr_list is None:
            from_addr_list = ['qms_tools@hp.com']
        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = ','.join(from_addr_list)
            message["To"] = ','.join(recipients)

            body = MIMEText(body, 'html')
            message.attach(body)
            mail_content = message.as_string()

            server = smtplib.SMTP(self.smtp_mail_server, self.smtp_port)

            server.connect(self.smtp_mail_server, self.smtp_port)
            server.sendmail(from_addr_list, recipients, mail_content)
            return 'Mail sent successfully!'

        except Exception as e:
            return f'Failed to send the mail due to {e}'

    def get_data(self, query, conn_key='aws3', chunksize=1000000):
        """Runs provided select query on the given connection key's server and returns data in a dataframe.
           Note: It returns a generator object if number of rows being fetched > chunksize. Loop through the returned object to access the data in chunks of specified chunksize. 
        Args:
            query (str): select query you want to run
            conn_key (str, optional): Valid connection key for helper library. Defaults to 'aws3'.
            chunksize (int, optional): In case you get memory issues, try reducing the chunksize. Defaults to 1000000.

        Returns:
            pandas.DataFrame: Data from the select query.
        """

        fields = self.fields_pattern.search(query.lower()).group(1)
        count_query = query.replace(fields, 'count(*) as count')
        # print(f'count_query: {count_query}')

        conn = self.get_connection(conn_key)
        record_count = pd.read_sql(count_query, conn)['count'].values[0]
        # print(f'record_count without limit clause: {record_count}')

        limit_count = re.compile('limit (\d*)').search(query)
        # print(f'limit_count: {limit_count}')
        if limit_count:
            record_count = min(int(limit_count.group(1)), record_count)
        print(f'Final record_count: {record_count}')

        if record_count > chunksize:
            print(f'Total records > {chunksize}')
            print(
                f'Returning a generator object with chunksize = {chunksize} so please loop through the returned object to get your dataframes and accept returned connection')
            df = pd.read_sql(query, conn, chunksize=chunksize)

        else:
            df = pd.read_sql(query, conn)
            conn.close()

        return df, conn

    def get_sample_data(self, query: str, verbose: bool = False, key: str = 'aws3', rows: int = 100) -> Union[
            str, Iterator[pd.DataFrame], pd.DataFrame]:
        """Helps get sample data based on the query provided.
        Note: The query should not already contain 'limit' and it only works for vertica servers.

        Args:
            query (str): Query you want to get the sample data for.
            verbose (bool, optional): Whether to print the limit query. Defaults to False
            key (str, optional): Valid connection key of the get_connection function. Defaults to 'aws3'.
            rows (int, optional): Number of rows you want to fetch for the sample. Defaults to 100.

        Returns:
            pandas.DataFrame: Dataframe containing the required sample data.

        Future Changes:
            1. Add support for sql server databases.
        """
        if 'limit' in query:
            print('Failed because limit already provided')
            return 'Failed'
        elif ';' in query:
            limit_query = query.replace(';', f' limit {rows};')
            # print(f'Running {limit_query}')
        else:
            limit_query = query + f' limit {rows};'

        if verbose:
            print(limit_query)

        with self.get_connection(key) as conn:
            df = pd.read_sql(limit_query, conn)
            return df

    def run_query(self, query, conn_key='aws3', join_spill=False) -> bool:
        """Runs a query on the given connection key's server

        Args:
            query (str): update, delete, merge, (basically anything except select) query to run
            conn_key (str, optional): Valid connection key from helper library. Defaults to 'aws3'.

        Returns:
            bool: True if it ran successfully, False if failed.
        """
        print(f'Running \n{query}\n on {conn_key}')

        try:
            with self.get_connection(conn_key) as conn:
                cur = conn.cursor()
                if join_spill:
                    cur.execute(self.join_spill_enabling)
                cur.execute(query)
                conn.commit()
            print('Ran successfully!')
            return True
        except Exception as e:
            print('Failed!')
            print(f'Due to {e}')
            return False

    def add_columns_to_table(self, table: str, cols: Dict[str, str], schema=None) -> None:
        """Add given columns with their sql datatypes to the table mentioned.

        Args:
            table (str): Name of the table
            cols (Dict[str, str]): Python dictionary with column names as keys and their data types as values.
            schema ([type], optional): Name of the schema your table exists in. Defaults to self.schema value if set to None.

        Returns:
            None: Returns None
        """

        if schema is None:
            schema = self.schema

        for col, dtype in cols.items():
            q = f"""alter table {schema}.{table}
                    add column {col} {dtype};"""
            self.run_query(q)
            print('\n')
        return None

    def update_from_another_table(self, tgt: str, src: str,
                                  tgt_key: str,
                                  tgt_cols: List[str],
                                  tgt_schema='pps_quality',
                                  src_key: str = None,
                                  src_cols: List[str] = None,
                                  src_schema='pps_quality', conn_key='aws3',
                                  update_only_nulls: bool = False) -> None:
        """
        Updates columns from a source table to a given target table based on the keys provided. 

        Args:
            tgt (str): Target table you want to update
            src (str): Source table from where the update will happen.
            tgt_key (str): Key column in target table
            tgt_cols (List[str]): Target columns that need to be updated 
            tgt_schema (str, optional): Schema of the target table. Defaults to 'pps_quality'.
            src_key (str, optional): Key column of source table. Defaults to tgt_key if set to None.
            src_cols (List[str], optional): Source columns for the provided target columns. Defaults to tgt_cols if set to None.
            src_schema (str, optional): Schema of the source table. Defaults to 'pps_quality'.
            conn_key (str, optional): Connection key that's supported by the helper library's get_connection function. Defaults to 'aws3'.
            update_only_nulls(bool, optional): Adds conditions to update only rows where at least one tgt_col is missing. Defaults to False

        Returns:
            None: None
        """

        if src_key is None:
            src_key = tgt_key

        if src_cols is None:
            src_cols = tgt_cols

        set_str = ''
        for i, cols in enumerate(zip(tgt_cols, src_cols)):
            tgt_col, src_col = cols
            if i == 0:
                set_str += f'{tgt_col} = src.{src_col}'
            else:
                set_str += f', {tgt_col} = src.{src_col}'

        q = f"""update {tgt_schema}.{tgt} tgt
                set {set_str}
                from {src_schema}.{src} src
                where tgt.{tgt_key} = src.{src_key}
            """

        if update_only_nulls:
            for i, tgt_col in enumerate(tgt_cols):
                if i == 0:
                    q += f' and (tgt.{tgt_col} is null'
                else:
                    q += f' or tgt.{tgt_col} is null'
            q += ')'

        self.run_query(q)
        return None

    def get_dtypes_from_ddl(self, table: str, schema: str = None, return_ddl: bool = False, conn_key: str = 'aws3') -> \
            Tuple[dict, Any]:
        """Gets sql columns and their datatypes as python dictionary using the export_objects function of vertica.
        If return_ddl is set to True then the create table query of the table will also be returned.
        Args:
            table (str): Table name
            schema (str, optional): Schema name. Defaults to self.schema if set to None.
            return_ddl (bool, optional): Whether to return the create_table_query or not. Defaults to False.
            conn_key (str, optional): Connection key supported by get_connection function of the helper library. Defaults to 'aws3'.

        Returns:
            if return_ddl = False:
                Dict[str, str]: Sql columns with column names as key and their data types as values
            else:
                Dict[str, str], str: Same dictionary as above along with a string containing the create_table_query
        """
        if schema is None:
            schema = self.schema

        query = f"""select export_objects('', '{schema}.{table}', FALSE);"""
        with self.get_connection(conn_key) as conn:
            objects = pd.read_sql(query, conn)

        ddl = objects.export_objects[0]
        create_table_query = ddl.split(';')[0]

        sql_cols = {}
        for line in create_table_query.split('\n')[4:]:
            words = line.split(' ')

            if len(words) > 4:
                col_name = words[4]
                col_type = words[5].replace(',', '')
                sql_cols[col_name] = col_type

        if return_ddl:
            return sql_cols, create_table_query
        else:
            return sql_cols

    @staticmethod
    def divide_into_chunks(df: pd.DataFrame, chunksize: int, verbose: bool = False):
        """Divides a given dataframe into chunks of size provided. Returns a list of dataframes.

        Args:
            df (pd.DataFrame): DataFrame one wants to reduce.
            chunksize (int): Size of every chunk (approx.)
            verbose (bool, optional): Enable/Disable print statements. Defaults to False.

        Returns:
            List[pd.DataFrame]: List of dataframes divided into chunks of size: chunksize (approx.)
        """

        num_blocks = (len(df) / chunksize) + 1
        dfs = np.array_split(df, num_blocks)
        if verbose:
            print(f'Divided into {len(dfs)} chunks each of size {len(dfs[0])}')
        return dfs
