from helper_pkg.neural_miner import NM
from tqdm import tqdm
import numpy as np
import re
from . import Helper
import pandas as pd
from pysbd.utils import PySBDFactory
import spacy
spacy.prefer_gpu()
# from tqdm import tqdm
tqdm.pandas()


class CDAX(Helper, NM):
    nlp = spacy.blank('en')
    nlp.add_pipe(PySBDFactory(nlp))

    date_pattern = '\d\d [a-z][a-z][a-z] 20\d\d'
    pattern_id = '@\w+'
    ip_pattern = '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
    date_pattern1 = '\d{1,2}-[a-z][a-z][a-z]-\d{2,4}'

    start_list = ['visible ext', r'issue', 'GN_TEMPLATE_START']
    stop_list = ['endts', date_pattern, 'GN_TEMPLATE_END']

    exclusion_list = [
        'name', 'phone',
        # 'email',
        r'na$', r'no$',
        'call direction', r'wise\s*doc', 'unit type',
        '-==',
        pattern_id, ip_pattern, date_pattern1,
        'start case summary', 'end case summary',
    ]

    info_list = ['\?', '\:',
                 '=>',
                 r'\*\* \d',
                 ]

    inclusion_list = [
        'problem description',
        'issue',
        'issue description',
        'failure detail',
    ]

    hard_exclude_list = [
        'failure category',
        'how to reproduce',
        'check emerging issues',
    ]

    def __init__(self):
        Helper.__init__(self)
        NM.__init__(self)

        remove_dict = {
            r'^=': '',
            r'^-': '',
            r'--+': ' ',
            r'===+': ' ',
            r'(\>\>|\=\>)': '.',
            r'\t+': ' ',
            r' +': ' ',
            r'\S+@\S+': ' ',
            r'\\+': ' ',
            r'/+': ' ',
            r'#+': '',
            r'\s•\s': ' ',
            r'[_\'\"-\(\)\^|]+': '',
            r';+': '.',
            r'[<\[\]]+': '',
            r'[\{\}]+': '',
            r'- -': '',
            r'\d:\d\d (a|p)m [a-z]*:': '',
            r'\d:\d\d (a|p)m': '',
            r'\r': ' ',
            r'\*+': '',
            r'[\d\-]{4,12}': ' '
        }

        self.remove_patterns = {}
        for remove_ele, sub_with in remove_dict.items():
            pattern = re.compile(remove_ele)
            self.remove_patterns.update({pattern: sub_with})

        self.alpha_pattern = re.compile('[a-zA-Z]')
        self.num_pattern = re.compile('\d')
        self.win_pattern = re.compile('win')

        self.non_alpha = re.compile("[^A-Za-z]+")
        self.exception_patterns = [re.compile(
            'win'), re.compile('==>'), re.compile('>[A-Za-z]+')]

        self.visible_externally_to_datetime = re.compile(
            'visible external[ly:.>< ]*(yes|no)*[:.>< ]*\W*(.+?)[a-z][a-z][a-z] \d\d \d\d\d\d \d\d:\d\d:\d\d')
        self.log_type = re.compile('log type')
        self.core_issue = re.compile('(.*)log type')
        self.start_to_datetime = re.compile(
            '(.+?)[a-z][a-z][a-z] \d\d \d\d\d\d \d\d:\d\d:\d\d')  # mar 26 2018 23:34:04
        self.start_to_datetime2 = re.compile(
            '(.+?)\d\d/\d\d/\d\d\d\d \d\d:\d\d:\d\d')  # 07/18/2016 07:46:54
        self.endts_to_end = re.compile('endts(.*)')
        self.visible_externally_to_end = re.compile(
            'visible externally[:.>< ]*(yes|no)*[:.>< ]*\W*(.*)')

    def clean_case_notes(self, df, notes_col, id_col='case_id', sub_col=None):
        """Applies case notes and case subject cleaning module to the provided dataframe and returns dataframe with cleaned columns. 
        Args:
            df (pandas.DataFrame): Dataframe containing cdax case notes
            notes_col (str): Case notes column name
            id_col (str, optional): case id column name. Defaults to 'case_id'.
            sub_col (str, optional): Case subject/title column name. Defaults to None.

        Returns:
            pandas.DataFrame: Original dataframe with added cleaned columns. 

        Note: It will only apply the case subject cleaning module if a sub_col is provided.

        Future upgrades:
            1. Check with RDR_MASTER if case note is already cleaned using case_id.
            2. Deal with new changes in the case notes template. Cases starting with 'Notes Type : Notes Log\rIs Visible Externa'
        """
        cleaned_col = notes_col + '_cleaned'

        if sub_col:
            cleaned_sub = sub_col + '_cleaned'
            print('Cleaning case subject since sub_col provided')
            print(f'\tSubject column: {sub_col}')
            df[cleaned_sub] = df[sub_col].progress_apply(self.clean_subject)

        print('Cleaning case_notes:')
        print('\tbasics and discarding junk... (1/4)')
        df[cleaned_col] = df[notes_col].apply(self.remove_equals)

        print('\tApplying main cleaning module...  (2/4)')
        df[cleaned_col] = df[cleaned_col].progress_apply(self.cleaner_main)

        # print('\tRemoving alpha-numerics...  (3/4)')
        # df[cleaned_col] = df[cleaned_col].apply(self.remove_alpha_numeric_v2)

        print('\tRemoving blanks...  (4/4)')
        df[cleaned_col] = df[cleaned_col].apply(self.remove_blanks)

        df.dropna(subset=[id_col, cleaned_col], inplace=True)
        print('Completed case notes cleaning')

        return df

    def remove_equals(self, s):
        """
        Replaces / Removes junk basics using remove_patterns dictionary.

        Parameters:
        s (string): Description of arg1

        Returns:
        str: Returns clean string
        """
        if(isinstance(s, str)):
            s = s.lower()
            for pattern, sub_with in self.remove_patterns.items():
                s = re.sub(pattern, sub_with, s)
        return s

    def remove_alpha_numeric_v2(self, text):
        """Removes any alpha-numeric words.

        Args:
            text (str): Text to be cleaned.

        Returns:
            str: Cleaned text

        Future upgrades:
            1. Use tokenizer instead of str.split()
            2. Add option to replace with <ALP_NUM> token as well.
        """
        if not isinstance(text, str):
            return np.nan

        chosen_words = []
        words = text.split()
        for word in words:
            if self.non_alpha.search(word):
                #             print(f'{word} contain non-alphabetic characters')
                for exception_pattern in self.exception_patterns:
                    if exception_pattern.search(word):
                        chosen_words.append(word)
                continue
            else:
                chosen_words.append(word)
        return ' '.join(chosen_words)

    def cleaner_main(self, s):
        """Main case notes cleaning function.

        Args:
            text (str): Text to be cleaned.

        Returns:
            str: Cleaned text
        """

        if not isinstance(s, str):
            return np.nan

        new_instances = []
        s_t_d = self.start_to_datetime.findall(s)
        s_t_d2 = self.start_to_datetime2.findall(s)

        if not s_t_d and not s_t_d2:
            new_instances.append(s)

        if s_t_d:
            instances = s_t_d
            for i, instance in enumerate(instances):
                new_instance = instance
                save_core = ''
                if i == 0:
                    c_i = self.core_issue.search(new_instance)
                    if c_i:
                        save_core = c_i.group(1)
                    v_e_t_e = self.visible_externally_to_end.search(
                        new_instance)
                    if v_e_t_e:
                        new_instance = v_e_t_e.group(2)
                else:
                    e_t_e = self.endts_to_end.search(instance)
                    if e_t_e:
                        new_instance = e_t_e.group(1)
                        v_e_t_e = self.visible_externally_to_end.search(
                            new_instance)
                        if v_e_t_e:
                            new_instance = v_e_t_e.group(2)
                    else:
                        new_instance = instance
                if save_core != '':
                    new_instance = save_core + " " + new_instance

                new_instances.append(new_instance)

        if s_t_d2:
            instances = s_t_d2
            for instance in instances:
                e_t_e = self.endts_to_end.search(instance)
                if e_t_e:
                    new_instance = e_t_e.group(1)
                else:
                    new_instance = instance
                new_instances.append(new_instance)

        result = "==> ".join(new_instances)
        return result

    def remove_blanks(self, s):
        if isinstance(s, str) and len(s) > 0:
            return s

    def clean_subject(self, s):
        sections = s.split('/')
        if sections is not None and len(sections) > 2:
            s = ' '.join(sections)
            # print(s)
        s = self.remove_equals(s)
        # print(s)
        s = self.remove_alpha_numeric_v2(s)
        s = self.remove_equals(s)
        s = self.remove_equals(s)
        s = self.remove_alpha_numeric_v2(s)

        if len(s) < 2:
            s = np.nan

        return s

    def clean_sentence_wise(self, s, verbose=False):
        if not isinstance(s, str):
            return np.nan

        doc = s.lower().replace('\r', '\n')

        relevant = []
        infos = []

        take_flag = False
        exclude = False
        include = False
        info = False
        pysbd_flag = False
        hard_exclude = False

        sentences = doc.split('\n')

        if verbose:
            print(f'Number of sentences: {len(sentences)}')
        if len(sentences) <= 35:
            sentences = CDAX.nlp(doc).sents
            pysbd_flag = True

        for sent in sentences:
            if pysbd_flag:
                sent = sent.text

            if take_flag:
                # Define info conditions. (':', '?', email: )
                for pattern in CDAX.info_list:
                    if re.search(pattern, sent):
                        for pattern in CDAX.inclusion_list:
                            if re.search(pattern, sent):
                                if pattern == 'issue':
                                    for p in CDAX.hard_exclude_list:
                                        if re.search(p, sent):
                                            hard_exclude = True
                                            if verbose:
                                                print(
                                                    f'\tHard Excluded: {sent.strip()}')
                                            break
                                    if not hard_exclude:
                                        include = True
                                        if verbose:
                                            print(
                                                f'\tIncluded: {sent.strip()}')
                                        break
                                else:
                                    include = True
                                    if verbose:
                                        print(f'\tIncluded: {sent.strip()}')
                                    break

                        if not include:
                            infos.append(sent.strip())
                            if verbose:
                                print(f'\tInfo: {sent.strip()}')
                            info = True
                            break

                if not info and not include:
                    # Define exclude conditions. (name:, phone:, email: )
                    for pattern in CDAX.exclusion_list:
                        if re.search(pattern, sent) or len(sent) < 3:
                            exclude = True
                            if verbose:
                                print(f'\tExcluded: {sent.strip()}')
                            break
    #             print(((not exclude and not info) or include) and not hard_exclude)
                if ((not exclude and not info) or include) and not hard_exclude:
                    text = sent.strip()
                    text = self.remove_equals(text)
                    relevant.append(text)

                exclude = False
                info = False
                include = False
                hard_exclude = False
                # Define stop sentence condition  ('endts', date_pattern)
                for pattern in CDAX.stop_list:
                    if re.search(pattern, sent):
                        if len(relevant) > 0 and relevant[-1] != '==>':
                            relevant.append('==>')
                        take_flag = False
                        break
            else:
                # Define start sentence condition  ('visible ext', 'issue')
                for pattern in CDAX.start_list:
                    if re.search(pattern, sent):
                        take_flag = True
                        if pattern == r'issue(.*)':
                            sent = re.search(pattern, sent).group(1)
                            text = sent.strip()
                            text = self.remove_equals(text)
                            relevant.append(text)
                            if verbose:
                                print(f'Started at: {sent.strip()}')
                        else:
                            xyz = 'xyz'
                            if verbose:
                                print(f'Started after: {sent.strip()}')
                        break
        if len(relevant) > 0:
            if relevant[-1] == '==>':  # Remove ending '==>'
                relevant.pop()

        final_result = ' . '.join(relevant) if len(relevant) > 0 else doc
        
        if verbose:
            print(f'Final result: {final_result}\n')
    #     print(f'Infos: {infos}')
        return final_result
