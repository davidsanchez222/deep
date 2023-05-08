import nltk
# nltk.download_shell()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
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
