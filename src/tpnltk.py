import gensim
from gensim import corpora, similarities, models
from gensim.matutils import cossim
import nltk, codecs
import numpy as np
import json
import sys, os, argparse
import glob
import pprint

nltk.download('punkt')
nltk.download('stopwords')
STOPWORDS = nltk.corpus.stopwords.words('spanish')

def get_corpus_filenames(path):
    import glob
    filenames=[]
    for f in glob.glob(os.path.join(path, '*.json')):
        filenames.append(f)
    return filenames

def remove_once_app_words(news):
    all_tokens = sum(news, [])
    #print all_tokens
    tokens_once = set(word for word in set(all_tokens)
                      if all_tokens.count(word) == 1)
    return [[word for word in new if word not in tokens_once] for
            new in news ]
    

def get_corpus(path):
    # get corpus (la nacion)
    # E.G: path = '../lanacion/'
    filenames = get_corpus_filenames(path)
    news = []

    for i in xrange(len(filenames)):
        jsonf = open(filenames[i], 'r')
        d = json.load(jsonf, strict=False)
        jsonf.close()

    for i in xrange(len(d)):
        wordss = []
        words = nltk.word_tokenize(d[i]['copete'])
        words = [word.lower()
                 for word in words
                 if word not in STOPWORDS]
        wordss.extend(words)

        cuerpo = ','.join(d[i]['cuerpo'])
        words = nltk.word_tokenize(cuerpo)
        words = [word.lower()
                 for word in words
                 if word not in STOPWORDS]
        wordss.extend(words)

        words = nltk.word_tokenize(d[i]['titulo'])
        words = [word.lower()
                 for word in words
                 if word not in STOPWORDS]
        wordss.extend(words)

        news.append(wordss)

    news = remove_once_app_words(news)
    dictionary = corpora.Dictionary(news)
    # print dictionary
    corpus = [dictionary.doc2bow(new) for new in news]
    return corpus, dictionary

def save_corpus(corpus):
    corpus.save('corpus.dict')

def load_corpus():
    return corpus.load('corpus.dict')

def lsi_model_from(corpus, id2word=None):
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=id2word, num_topics=300)
    # corpus_lsi = lsi[lsi]

    # test:
    # lsi.print_topics()
    # for i in lsi.print_topics():
    #     print i
    return lsi

def process_command_line_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',
                        action='store', dest='path',
                        help='Path for texts books', default='../textos/freud/')
    parser.add_argument('-b', '--book',
                        action='store', dest='book',
                        help='Name of text`s  Book',
                        default='Volumen_I.pdf.txt')
    parser.add_argument('-t', '--tokenizer',
                        action='store', dest='tokenizer',
                        help='Tokenizer: word, sent', default='sentence')
    opt = parser.parse_args(args[1:])
    return opt.path, opt.book, opt.tokenizer

def get_author_text(path, text, tokenizer):
    # E.g: freud -> Volument_I.pdf.txt
    # path = 'textos/freud/'
    # f = codecs.open(path+'Volumen_I.pdf.txt', 'r', 'utf-8-sig')
    f = codecs.open(path+text, 'r', 'utf-8-sig')
    t = f.read()

    # load tokenizer
    # E.g: sent_detector = nltk.data.load('tokenizers/punkt/spanish.pickle')
    return tokenizer.tokenize(t.strip())

    # test:
    # for i in range(0,4):
    #     print tokens[i] + '\n'
        # a = nltk.word_tokenize(text)
        #tokens = nltk.word_tokenize(t)
        #print tokens
        # text = nltk.Text(tokens)
        # print text.strip()
        #print lexical_diversity(text)

def doc_similarity(doc, dictionary, lsi_model, index):
    sim = []
    for i,s1 in enumerate(doc):
        try:
            s1 = doc[i]
            s2 = doc[i+1]
            # sentences = dictionary.doc2bow((s1+s2).lower().split())
            # sentences_vect = lsi_model[sentences]
            # sentences_sim = index[sentences_vect]
            # sentences_sim = np.mean(sentences_sim)
            # std = np.std(sentences_sim)
            
            s11 = dictionary.doc2bow(s1.lower().split())
            s22 = dictionary.doc2bow(s2.lower().split())
            s1_vec = [lsi_model[s11]]
            s2_vec = [lsi_model[s22]]
            s1_sim_mean = np.mean(index[s1_vec])
            s2_sim_mean = np.mean(index[s2_vec])
            print s1_sim_mean
            print s2_sim_mean
            sentences_sim = [gensim.matutils.cossim(s2_sim_mean, s2_sim_mean)]
            #print sentences_sim
            sim.append(sentences_sim)
        except:
            pass
    return sim

def main():
    path, text, tokenizer = process_command_line_args(sys.argv)
    # path = '../textos/freud/'
    # text = 'Volumen_I.pdf.txt'
    tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
    text = get_author_text(path, text, tokenizer)
    # try:
    #     load_corpus(corpus)
    # except:
    corpus, dictionary = get_corpus('../lanacion/')
    lsi_model = lsi_model_from(corpus, dictionary)
    index = similarities.MatrixSimilarity(lsi_model[corpus])

    sim_coeff = doc_similarity(text, dictionary, lsi_model, index)
    pprint.pprint( sim_coeff)
    # pprint.pprint( np.std(sim_coeff))

if __name__ == '__main__':
    main()
