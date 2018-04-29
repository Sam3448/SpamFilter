
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from os import listdir
from os.path import isfile, join
import fileinput


nltk.data.path = ['../dataset/nltk_data']
stopwords = set(stopwords.words('english'))

def get_words(message):

    all_words = set(wordpunct_tokenize(message.replace('=\\n', '').lower()))
    msg_words = [word for word in all_words if word not in stopwords]
    return msg_words

    
def get_mail_from_file(file_name):
    message = ''
    
    with open(file_name, 'r') as mail_file:
        for line in mail_file:
            message += line
                    
    return message
    
    
def make_training_set(path):
    training_set = {}
    mails_in_dir = [mail_file for mail_file in listdir(path) if isfile(join(path, mail_file))]
    total_count = 0
    
    for mail_name in mails_in_dir:
        if mail_name.startswith('.DS'):
            continue

        for message in fileinput.input(path + mail_name):
            total_count += 1
            terms = get_words(message)

            for term in terms:
                if term in training_set:
                    training_set[term] = training_set[term] + 1
                else:
                    training_set[term] = 1

    for term in training_set.keys():
        training_set[term] = float(training_set[term]) / total_count
                            
    return training_set

def init(ham_path, spam_path):
    print('Loading training sets...')

    ham_training_set = make_training_set(ham_path)
    spam_training_set = make_training_set(spam_path)

    print('done.')
    return ham_training_set, spam_training_set