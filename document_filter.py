import MeCab
import pickle
import Stemmer
import numpy  as np
import pandas as pd
from sklearn.cross_validation        import ShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.grid_search             import GridSearchCV
from sklearn.metrics                 import f1_score
from sklearn.metrics                 import make_scorer
from sklearn.naive_bayes             import MultinomialNB
from sklearn.pipeline                import Pipeline

class JapaneseTfidfVectorizer(TfidfVectorizer):
    tagger               = MeCab.Tagger('-Owakati')

    def build_tokenizer(self):
        tokenizer        = super(JapaneseTfidfVectorizer, self).build_tokenizer()
        return lambda doc: self.tagger.parse(doc)

class EnglishTfidfVectorizer(TfidfVectorizer):
    english_stemmer      = Stemmer.Stemmer('en')

    def build_analyzer(self):
        analyzer         = super(EnglishTfidfVectorizer, self).build_analyzer()
        return lambda doc: self.english_stemmer.stemWords(analyzer(doc))

class DocumentFilter:
    def tune(self, lang, pos_label, tfile, efile):
        # Load
        columns          = ['labels', 'documents']
        data             = pd.read_table(tfile, header=None, names=columns)
        labels           = data['labels']
        documents        = data['documents']

        # Tune
        clf_factory      = self.__create_ngram_model(lang)
        estimator        = self.__grid_search_model(clf_factory, documents, labels, pos_label)

        # Serialize
        pickle.dump(estimator, open(efile, 'wb'))

    def predict(self, ifile, efile, ofile):
        # Load
        columns          = ['documents']
        data             = pd.read_table(ifile, header=None, names=columns)
        documents        = data['documents']

        # Deserialize
        estimator        = pickle.load(open(efile, 'rb'))

        # Predict
        probability      = estimator.predict_proba(documents)
        data['labels']   = estimator.predict(documents)
        data['C1_pr']    = probability[:, 0]
        data['C2_pr']    = probability[:, 1]

        # Save
        columns          = ['labels', 'C1_pr', 'C2_pr', 'documents']
        data.to_csv(
            ofile,
            sep          = '\t',
            columns      = columns,
            index        = False
        )

    def __create_ngram_model(self, lang):
        if   lang == 'en':
            tfidf_ngrams = EnglishTfidfVectorizer(decode_error='ignore')
        elif lang == 'ja':
            tfidf_ngrams = JapaneseTfidfVectorizer(decode_error='ignore')

        clf              = MultinomialNB()
        pipeline         = Pipeline([('vect', tfidf_ngrams), ('clf', clf)])

        return pipeline

    def __grid_search_model(self, clf_factory, documents, labels, pos_label):
        boolndarr        = labels.values == pos_label
        n                = documents.size
        n_pos            = labels[boolndarr].size
        n_neg            = n - n_pos

        param_grid       = {
            'vect__binary'      : [False, True],
            'vect__min_df'      : [1, 2],
            'vect__ngram_range' : [(1, 1), (1, 2), (1, 3)],
            'vect__smooth_idf'  : [False, True],
            'vect__stop_words'  : [None, 'english'],
            'vect__sublinear_tf': [False, True],
            'vect__use_idf'     : [False, True],
            'clf__alpha'        : [0, 0.01, 0.05, 0.1, 0.5, 1]
        }

        k                = 5
        cv               = ShuffleSplit(
            n,
            n_iter       = k,
            test_size    = 1 / k,
            random_state = 0
        )

        pos_weight       = n_neg / n_pos
        sample_weight    = np.ones(n)
        sample_weight[boolndarr] *= pos_weight
        fit_params       = {'clf__sample_weight': sample_weight}

        f1_scorer        = make_scorer(f1_score, pos_label=pos_label)

        grid_search      = GridSearchCV(
            clf_factory,
            param_grid,
            cv           = cv,
            fit_params   = fit_params,
            n_jobs       = -1,
            scoring      = f1_scorer
        )

        grid_search.fit(documents, labels)
        best_estimator   = grid_search.best_estimator_
        best_score       = grid_search.best_score_
        best_params      = grid_search.best_params_

        print("Best F1 score: {0:04.3f}".format(best_score))
        print("Parameters: {0}".format(best_params))

        return best_estimator
