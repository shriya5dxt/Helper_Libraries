import pandas as pd
import treelib
from treelib import Tree, Node
import re
import stanza
import numpy as np
import pkg_resources
from tqdm import tqdm


class HPNLP():
    def __init__(self) -> None:
        # Properties
        x = 10
        self.dct = {'gr8': 'great', 'awsm': 'awesome', 'thnx': 'thanks'}
        self.records = []
        self.remove_dict = {
            r'^=': '',
            r'^-': '',
            r'--+': ' ',
            r'===+': ' ',
            r'\n+': ' ',
            r'(\>\>|\=\>)': '.',
            r'\t+': ' ',
            r' +': ' ',
            r'\S+@\S+': ' ',

            r'\*+': '',
            r'#+': '',
            r'\s•\s': ' ',
            r'[_\'\"-\(\)\^|]+': '',
            r';+': '.',
            r'[<\[\]]+': '',
            r'[\{\}]+': '',
            r'- -': '',
            r'\d:\d\d (a|p)m [a-z]*:': '',
            r'\d:\d\d (a|p)m': '',
        }

        self.remove_patterns = {}
        for remove_ele, sub_with in self.remove_dict.items():
            pattern = re.compile(remove_ele)  # Review IGNORECASE flag
            self.remove_patterns.update({pattern: sub_with})

        # initialize English neural pipeline
        self.nlp = stanza.Pipeline('en')
        self.non_alpha = re.compile("[^A-Za-z]+")
        self.exception_patterns = [re.compile('win'),
                                   re.compile('==>'),
                                   re.compile('>[A-Za-z]+'),
                                   re.compile('__neutral'),
                                   re.compile('[!\.\'",]')
                                   ]

    def apply(self, df, text_col, id_col):
        """Main function"""
        self.records = []
        for text, id_ in tqdm(zip(df[text_col], df[id_col]), desc='Loading...', ascii=False, total=len(df)):
            if not isinstance(text, str):
                # print(f'\n\n\n\n Skipping {text} \n\n\n\n')
                continue
            text = self.acronym(text)
            text = self.remove_equals(text.lower())
            # print(text.lower())
            self.getTags(self.nlp(text), id_, self.records)

        self.record_dicts = [record.__dict__ for record in self.records]
        self.result = pd.DataFrame(self.record_dicts)
        self.result = self.result.drop_duplicates(ignore_index=True)

        self.final = self.post_processing(self.result)
        return self.final

    def acronym(self, s, dct=None):
        if not isinstance(s, str):
            return s

        if not dct:
            dct = self.dct
        return ' '.join([dct.get(i, i) for i in s.split()])

    def remove_equals(self, s):
        """
        Replaces / Removes junk basics using remove_patterns dictionary.

        Parameters: 
        s (string): Description of arg1 

        Returns: 
        str: Returns clean string
        """
        if(isinstance(s, str)):
            for pattern, sub_with in self.remove_patterns.items():
                s = re.sub(pattern, sub_with, s)
        return s

    def getTags(self, doc, review_id, records):
        """Code to generate dependency tree from stanza documents

        Args:
            doc ([type]): [description]
            records ([type]): [description]
        """
        i = 0
        # print(*[f'Sentence: {1}\tid: {word.id}\tword: {word.text} \tlemma: {word.lemma}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}\tPOS: {word.xpos}' for sent in doc.sentences for word in sent.words], sep='\n')
        for sent_id, stanza in enumerate(doc.sentences):
            i = i + 1
            current = []
            sentiments = []
            nouns = []
            base = None
            tree = myTree()
            headid = 0
            sentence = ""
            phrase = ""
            for word in stanza.words:
                sentence = sentence + " " + word.lemma
                phrase = phrase + " " + word.text
                if "love" == word.lemma:
                    word.xpos = "JJ"
                if "no" == word.lemma:
                    word.xpos = "NN"
                if "NN" in word.xpos:
                    nouns.append(word.id)
                elif "JJ" in word.xpos:
                    sentiments.append(word.id)
                if word.head == 0:
                    tree.create_node(word, word.lemma, word.id)
                    current.append(word.id)
                    headid = word.id
            loc = 0

            # for word in stanza.words:
            # if ("VB" in word.xpos and len(word.lemma) > 3):
            #     word.xpos = 'JJ'

            for word in stanza.words:
                if word.id not in current:
                    if word.head == int(tree[headid].identifier):
                        loc = loc+1
                        tree.create_node(word, word.lemma,
                                         word.id, parent=headid)
                        current.append(word.id)

            for word in stanza.words:
                if word.id not in current:
                    for c in tree.is_branch(headid):
                        if word.head == int(c):
                            loc = loc+1
                            tree.create_node(word, word.lemma, word.id, c)
                            current.append(word.id)

            for word in stanza.words:
                if word.id not in current:
                    for c in tree.is_branch(headid):
                        for c1 in tree.is_branch(c):
                            if word.head == int(c1):
                                loc = loc+1
                                tree.create_node(word, word.lemma, word.id, c1)
                                current.append(word.id)

            for word in stanza.words:
                if word.id not in current:
                    for c in tree.is_branch(headid):
                        for c1 in tree.is_branch(c):
                            for c2 in tree.is_branch(c1):
                                if word.head == int(c2):
                                    loc = loc+1
                                    tree.create_node(
                                        word, word.lemma, word.id, c2)
                                    current.append(word.id)

            if len(sentiments) > 0:
                self.handle(tree, 1)
            else:
                self.handle(tree, 0)
            # tree.show(None, 0)
            self.ReduceTree(tree)
            self.NounParent(tree)
            self.NounGroup(tree)
            self.NounGroup(tree)
            self.NounGroupWChild(tree)
            self.SentimentGroup(tree, sentence)
            self.ease(tree)

            # tree.show()
            out = []
            self.MapSentiments(tree, out)
            tag = ""
            for c in out:
                tag = c
                try:
                    term, sentiment_term = tag.split('=>')
                    record = Record(review_id, doc.text, sent_id,
                                    phrase, term, sentiment_term)
                    records.append(record)

                except ValueError as e:
                    term = c
                    sentiment_term = np.nan
                    record = Record(review_id, doc.text, sent_id,
                                    phrase, term, sentiment_term)
                    records.append(record)
                except Exception as e:
                    continue

    def ReduceTree(self, tree):
        for node in tree.expand_tree():
            try:
                arr = ['IN', 'DT', 'PRP', 'TO', 'MD', '.']
                if any(c in tree[node].word.xpos for c in arr):
                    if int(tree[node].word.head) != 0 and len(tree.is_branch(node)) == 0:

                        tree.remove_node(node)
            except:
                print("node not found")

    def NounParent(self, tree):
        for node in tree.expand_tree():
            if len(tree.is_branch(node)) == 0:
                narr = ['NN', 'CC', 'CD']
                if any(n in tree[node].word.xpos for n in narr):
                    parentid = tree[node].predecessor(tree.identifier)
                    if parentid != None:
                        p = tree.get_node(parentid)
                        if ("NN" in p.word.xpos):
                            if p.word.id < tree[node].word.id:
                                p.tag = p.tag + " " + tree[node].tag
                            elif p.word.id > tree[node].word.id:
                                p.tag = tree[node].tag + " " + p.tag
            #                     p.word.id = tree[node].word.id
                            tree.remove_node(node)

    def NounGroup(self, tree):
        for node in tree.expand_tree(filter=lambda x: 'NN' in x.word.xpos or 'CC' in x.word.xpos or 'CD' in x.word.xpos or x.word.head == 0):
            parentid = tree[node].predecessor(tree.identifier)
            if parentid != None:
                p = tree.get_node(parentid)
                if ("NN" in p.word.xpos or "CC" in p.word.xpos or "CD" in p.word.xpos) and len(tree.is_branch(node)) == 0:
                    if p.word.id < tree[node].word.id:
                        p.tag = p.tag + " " + tree[node].tag
                    elif p.word.id > tree[node].word.id:
                        p.tag = tree[node].tag + " " + p.tag
                        # p.word.id = tree[node].word.id
                    tree.remove_node(node)

    def NounGroupWChild(self, tree):
        for node in tree.expand_tree(filter=lambda x: 'NN' in x.word.xpos or x.word.head == 0):
            try:
                if 'NN' in tree[node].word.xpos or tree[node].word.head == 0:
                    parentid = tree[node].predecessor(tree.identifier)
                    if parentid != None:
                        p = tree.get_node(parentid)
                        if ("NN" in p.word.xpos) and len(tree.is_branch(node)) > 0 and abs(int(p.word.id)-int(tree[node].word.id)) <= 2:
                            if int(p.word.id) < int(tree[node].word.id):
                                p.tag = p.tag + " " + tree[node].tag
                            elif int(p.word.id) > int(tree[node].word.id):
                                p.tag = tree[node].tag + " " + p.tag
            #                     p.word.id = tree[node].word.id
                            k = tree.subtree(node)
                            for child in k.expand_tree():
                                if "NN" in k[child].word.xpos or "JJ" in k[child].word.xpos:
                                    tree.move_node(child, parentid)
                            tree.remove_node(node)
                break
            except:
                print("skip this entry")

    def SentimentGroup(self, tree, sentence):
        for node in tree.expand_tree(filter=lambda x: 'JJ' in x.word.xpos or 'NN' in x.word.xpos or 'advmod' in x.word.deprel or x.word.head == 0):
            parentid = tree[node].predecessor(tree.identifier)
            if parentid != None:
                p = tree.get_node(parentid)
                if "JJ" in p.word.xpos and ("advmod" in tree[node].word.deprel or "JJ" in tree[node].word.xpos) and len(tree.is_branch(node)) == 0:
                    if p.word.id < tree[node].word.id:
                        if p.tag + " " + tree[node].tag in sentence or len(p.tag.split(' ')) == 1:
                            p.tag = p.tag + " " + tree[node].tag
                        else:
                            bifurcate = p.tag.split(' ')
                            if len(bifurcate) == 2:
                                if bifurcate[0] + " " + tree[node].tag + " " + bifurcate[1] in sentence:
                                    p.tag = bifurcate[0] + " " + \
                                        tree[node].tag + " " + bifurcate[1]
                    elif p.word.id > tree[node].word.id:
                        if tree[node].tag + " " + p.tag in sentence or len(p.tag.split(' ')) == 1:
                            p.tag = tree[node].tag + " " + p.tag
                        else:
                            bifurcate = p.tag.split(' ')
                            if len(bifurcate) == 2:
                                if bifurcate[0] + " " + tree[node].tag + " " + bifurcate[1] in sentence:
                                    p.tag = bifurcate[0] + " " + \
                                        tree[node].tag + " " + bifurcate[1]
                    tree.remove_node(node)

    def ease(self, tree):
        # tree.show()
        nodes_to_remove = []
        for node in list(tree.expand_tree(filter=lambda x: 'JJ' in x.word.xpos or 'advmod' in x.word.deprel or x.word.head == 0)):
            # print('Entered p', node)
            if 'JJ' in tree.get_node(node).word.xpos:
                # print('p is an adjective')
                siblings = tree.siblings(node)
                for sibling in siblings:
                    if "RB" in sibling.word.xpos:
                        # print('Found an advmod in its siblings')
                        tree[node].tag = f'{sibling.tag} {tree[node].tag}'
                        # print(sibling.word)
                        nodes_to_remove.append(sibling.word.id)

        for node in nodes_to_remove:
            if tree.get_node(node):
                tree.remove_node(node)
        # tree.show()

    def handle(self, tree, flag):
        for node in tree.expand_tree(filter=lambda x: 'VB' in x.word.xpos or x.word.head == 0):
            # print(tree[node])
            if ('VB' in tree[node].word.xpos and flag == 1):
                tree[node].word.xpos = 'NN'
            elif ('VB' in tree[node].word.xpos and flag == 0):
                tree[node].word.xpos = 'JJ'

    def MapSentiments(self, tree, lst):
        for node in tree.expand_tree():
            if "JJ" in tree[node].word.xpos:
                position = None
                subposition = None
                for c in tree.is_branch(tree[node].word.id):
                    if "NN" in tree[c].word.xpos:
                        if position == None:
                            position = tree[c].word.id
                        else:
                            if int(position) > int(tree[c].word.id):
                                subposition = position
                                position = tree[c].word.id
                            else:
                                subposition = tree[c].word.id
                if position != None:
                    if tree[node].tag not in str(lst):
                        lst.append(tree[position].tag+"=>"+tree[node].tag)
                if subposition != None:
                    self.MapSentiments(tree.subtree(subposition), lst)
                if position == None:
                    if tree[node].tag not in str(lst):
                        lst.append("GeneralFeedback"+"=>"+tree[node].tag)

            elif "NN" in tree[node].word.xpos or (tree[node].word.head == 0 and 'JJ' not in tree[node].word.xpos and len(tree[node].word.text) > 3):
                position = None
                subposition = None
                for c in tree.is_branch(tree[node].word.id):
                    if "JJ" in tree[c].word.xpos or "RB" in tree[c].word.xpos:
                        if position == None:
                            position = tree[c].word.id
                        else:
                            if int(position) > int(tree[c].word.id):
                                subposition = position
                                position = tree[c].word.id
                            else:
                                subposition = tree[c].word.id
                if position != None:
                    if tree[node].tag not in str(lst):
                        lst.append(tree[node].tag+"=>"+tree[position].tag)
                else:
                    if tree[node].tag not in str(lst):
                        lst.append(tree[node].tag)
                if subposition != None:
                    self.MapSentiments(tree.subtree(subposition), lst)

    def post_processing(self, records):
        # nlp_terms = pd.DataFrame(testdict.items(), columns=["text", "term"])
        # nlp_terms['term'].replace('', 'neutral', inplace=True)
        records.dropna(subset=['term'], inplace=True)
        records.reset_index(drop=True, inplace=True)
        # nlp_term_sentiment = pd.DataFrame(nlp_terms.term.str.split(
        #     ';').tolist(), index=nlp_terms.text).stack()
        # nlp_term_sentiment = nlp_term_sentiment.reset_index([0, 'text'])
        # nlp_term_sentiment.columns = ['text', 'term']
        # nlp_term_sentiment[['term', 'sentiment_term']
        #                    ] = nlp_term_sentiment.term.str.split("=>", expand=True)

        records['term'].replace('', np.nan, inplace=True)
        records.dropna(subset=['term'], inplace=True)
        records.reset_index(drop=True, inplace=True)
        # records = records.apply(lambda col: col.str.lower())

        records['sentiment_term'] = records['sentiment_term'].str.lower()
        records['term'] = records['term'].str.lower()

        # sdf = pd.DataFrame(nlp_term_sentiment)
        records['term'] = records['term'].apply(
            self.remove_prefix, args=['and '])
        records['sentiment_term'] = records['sentiment_term'].fillna(
            '__neutral')
        records = records[records['sentiment_term'].apply(len) > 1]

        resource_package = __name__
        resource_path = '/'.join(('hpnlp_files',
                                  'Ref_Term_Stoplist.csv'))
        self.stop_word_fs = pkg_resources.resource_stream(
            resource_package, resource_path)
        # with open("./hpnlp_files/stopword_list_2020_09_07.txt", 'r', encoding="latin-1'") as f:
        # with self.stop_word_fs as f:
        # self.stop = self.stop_word_fs.readlines()
        self.stopwords = pd.read_csv(self.stop_word_fs)
        self.stop = self.stopwords.Stopwords.to_list()

        records['cleaned_sent'] = records['sentiment_term'].str.split(' ').apply(
            lambda x: ' '.join(k for k in x if k not in self.stop))
        records = records[records['cleaned_sent'].apply(len) > 1]
        records["cleaned_sent"] = records["cleaned_sent"].apply(
            self.remove_alpha_numeric_v2)
        records['cleaned_sent'] = records['cleaned_sent'].replace("", np.nan)

        final_result = self.get_sentiment_scores(records)

        return final_result

    def get_sentiment_scores(self, df, sentiment_col='sentiment_score', sentiment_term_col='cleaned_sent'):
        import ast
        resource_package = __name__
        resource_path = '/'.join(('hpnlp_files',
                                  'Ref_Term_Score.csv'))
        self.scorer_fs = pkg_resources.resource_stream(
            resource_package, resource_path)

        # contents = self.scorer_fs.read()
        self.scores_df = pd.read_csv(self.scorer_fs)
        self.scores_df.set_index('Term', inplace=True)
        dictionary = self.scores_df.to_dict()['Sentiment_Score']
        # dictionary = ast.literal_eval(contents)
        df[sentiment_col] = df[sentiment_term_col].map(dictionary)
        return df

    def remove_prefix(self, text, prefix):
        if text.startswith(prefix):  # only modify the text if it starts with the prefix
            text = text.replace(prefix, "", 1)  # remove one instance of prefix
        return text
        # sdf.loc[sdf['term'].str.startswith("and "), 'sub'] = ""

    def remove_alpha_numeric_v2(self, text):
        chosen_words = []
        text = re.sub("^{P}+", "", text)
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


class myTree(Tree):
    def create_node(self, word, tag, identifier=None, parent=None):
        """
        Create a child node for the node indicated by the 'parent' parameter
        """
        node = myNode(tag, identifier)
        node.set_word(word)
        self.add_node(node, parent)
        return node


class myNode(Node):
    def set_word(self, word):
        if word is not None:
            self.word = word

    def get_word(self):
        return self.word


class Record():
    def __init__(self, review_id, review, sent_id, phrase, term, sentiment_term):
        self.review_id = review_id
        self.review = review
        self.sentence_id = sent_id
        self.sentence = phrase
        self.term = term
        self.sentiment_term = sentiment_term

    def __str__(self):
        return f'id: {self.review_id}\nphrase:{self.sentence}\nterm:{self.term}\nsentiment_term:{self.sentiment_term}'
