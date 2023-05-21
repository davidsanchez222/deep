import nltk
# nltk.download_shell()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import warnings
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
warnings.simplefilter(action='ignore', category=FutureWarning)
#%%
messages = [line.rstrip() for line in open(r'C:\Users\David\PycharmProjects\deep\nlp\smsspamcollection\SMSSpamCollection')]

for mess_no, message, in enumerate(messages[:10]):
    print(mess_no, messages)

messages = pd.read_csv(r'C:\Users\David\PycharmProjects\deep\nlp\smsspamcollection\SMSSpamCollection', sep='\t', names=["label", "message"])
messages.groupby('label').describe()

messages["length"] = messages["message"].apply(len)
messages["length"].plot.hist(bins=150)
plt.show()

messages["length"].describe()
messages[messages["length"] == 910]["message"].iloc[0]
messages.hist(column="length", by="label", bins=60, figsize=(12, 4))
plt.show()


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]


messages["message"].head(5).apply(text_process)

bow_transformer = CountVectorizer(analyzer=text_process).fit(messages["message"])
mess4 = messages["message"][3]
bow4 = bow_transformer.transform([mess4])

bow_transformer.get_feature_names_out()[4068]

messages_bow = bow_transformer.transform(messages["message"])

print("Shape of the Sparse Matrix: ", messages_bow.shape)

messages_bow.nnz

sparsity = (100 * messages_bow.nnz / (messages_bow.shape[0]) * messages_bow.shape[1])