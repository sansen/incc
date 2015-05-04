from gensim import corpora, similarities, models
import nltk, codecs
import json
import sys, os, argparse
import glob
import pprint

def get_corpus_filenames(path):
    import glob
    filenames=[]
    for f in glob.glob(os.path.join(path, '*.json')):
        filenames.append(f)
    return filenames

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
        words = [word.lower() for word in words]
        wordss.extend(words)

        cuerpo = ','.join(d[i]['cuerpo'])
        words = nltk.word_tokenize(cuerpo)
        words = [word.lower() for word in words]
        wordss.extend(words)

        words = nltk.word_tokenize(d[i]['titulo'])
        words = [word.lower() for word in words]
        wordss.extend(words)

        news.append(wordss)

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
            s2 = doc[i+1]
            sentences = dictionary.doc2bow((s1+s2).lower().split())
            sentences_vect = lsi_model[sentences]
            sentences_sim = index[sentences_vect]
            sim.append(sentences_sim)
        except:
            pass
    return sim

def main():
    nltk.download('punkt')
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
    pprint.pprint( sim_coeff[:3] )

if __name__ == '__main__':
    main()
