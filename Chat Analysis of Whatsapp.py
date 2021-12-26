#!/usr/bin/env python
# coding: utf-8

# In[1]:


import regex as re
import pandas as pd
import numpy as np
import emoji
import plotly.express as px
from collections import Counter
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def startsWithDateAndTime(s):
    pattern = '^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -' 
    result = re.match(pattern, s)
    if result:
        return True
    return False


# In[4]:


import regex as re
startsWithDateAndTime('10/10/2021, 13:03 - Mona: No placement')


# In[5]:


def FindAuthor(s):
  s=s.split(":")
  if len(s)==2:
    return True
  else:
    return False


# In[6]:


def getDataPoint(line):   
    splitLine = line.split(' - ') 
    dateTime = splitLine[0]
    date, time = dateTime.split(', ') 
    message = ' '.join(splitLine[1:])
    if FindAuthor(message): 
        splitMessage = message.split(': ') 
        author = splitMessage[0] 
        message = ' '.join(splitMessage[1:])
    else:
        author = None
    return date, time, author, message


# In[9]:


uploaded= pd.read_table('C:\\Users\\Hp\\Documents\\WhatsApp Chat with Current Affairs Discussions.txt')
data = []
parsedData=[]
conversation = 'WhatsApp Chat with Current Affairs Discussions.txt'
with open(conversation, encoding="utf-8") as fp:
    fp.readline() # Skipping first line
    messageBuffer = [] 
    date, time, author = None, None, None
    while True:
        line = fp.readline() 
        if not line: 
            break
        line = line.strip() 
        if startsWithDateAndTime(line): 
            if len(messageBuffer) > 0: 
                parsedData.append([date, time, author, ' '.join(messageBuffer)]) 
            messageBuffer.clear() 
            date, time, author, message = getDataPoint(line) 
            messageBuffer.append(message) 
        else:
            messageBuffer.append(line)


# In[10]:


df = pd.DataFrame(parsedData, columns=['Date', 'Time', 'Author', 'Message']) # Initialising a pandas Dataframe.
df["Date"] = pd.to_datetime(df["Date"])
df.tail(20)


# In[11]:


df.info()


# In[12]:


df.Author.unique()


# In[13]:


df = df.dropna()
df.info()


# In[14]:


total_messages = df.shape[0]
print(total_messages)


# In[15]:


media_messages = df[df['Message'] == '<Media omitted>'].shape[0]
print(media_messages)


# In[66]:


def split_count(text):
    emoji_counter=0

    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_counter +=1
            emoji_list.append(word)

    return emoji_counter, emoji_list

df["emoji"] = df["Message"].apply(split_count)


# In[80]:


emojis = sum(df['emoji'].str.len())
print(emojis)


# In[63]:


URLPATTERN = r'(https?://\S+)'
df['urlcount'] = df.Message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()


# In[64]:


links = np.sum(df.urlcount)


# In[79]:


print("Current Affairs Discussion")
print("Messages:",total_messages)
print("Media:",media_messages)
print("Emojis:",emojis)
print("Links:",links)


# In[68]:


media_messages_df = df[df['Message'] == '<Media omitted>']


# In[69]:


messages_df = df.drop(media_messages_df.index)


# In[70]:


messages_df.info()


# In[71]:


messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
messages_df["MessageCount"]=1


# In[72]:


messages_df.tail(20)


# In[73]:


l = ["David", "Mona", "Himanshu"]
for i in range(len(l)):
  
  req_df= messages_df[messages_df["Author"] == l[i]]

  print(f'Stats of {l[i]} -')

  # number of rows which indirectly means the number of messages
  print('Messages Sent', req_df.shape[0])
  #total words in one message
  words_per_message = (np.sum(req_df['Word_Count']))/req_df.shape[0]
  print('Words per message', words_per_message)
  #media messages
  media = media_messages_df[media_messages_df['Author'] == l[i]].shape[0]
  print('Media Messages Sent', media)
  #total no. of emojis
  emojis = sum(req_df['emoji'].str.len())
  print('Emojis Sent', emojis)
  #links
  links = sum(req_df["urlcount"])   
  print('Links Sent', links)   
  print()


# In[75]:


text = " ".join(review for review in messages_df.Message)
print ("There are {} words in all the messages.".format(len(text)))
stopwords = set(STOPWORDS)
# Generates a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
# Displays the generated image by matplotlib
plt.figure( figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[78]:


l = ["Rohan", "Ayana", "Himanshu"]
for i in range(len(l)):
  dummy_df = messages_df[messages_df['Author'] == l[i]]
  text = " ".join(review for review in dummy_df.Message)
  stopwords = set(STOPWORDS)
  #Generation of word cloud image
  print('Author name',l[i])
  wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
  #Displaying the generated image   
  plt.figure( figsize=(10,5))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.show()


# Finding frequent topics of discussion in the group

# In[164]:


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords


from nltk.tokenize import word_tokenize
with open ('C:\\Users\\Hp\\Documents\\HinglishStopwords.txt') as fin:
    irrelevantWords = word_tokenize(fin.read())



top_N = 13
txt = df['Message'].str.lower().str.replace(r'\|', ' ').str.cat(sep=' ')
words = nltk.tokenize.word_tokenize(txt)
word_dist = nltk.FreqDist(words)

words_except_stop_dist = nltk.FreqDist(w for w in words if w not in irrelevantWords)
print('All frequencies excluding irrelevant words but including emojis')

print('=' * 60)

rslt = pd.DataFrame(words_except_stop_dist.most_common(top_N),
                    columns=['Word', 'Frequency']).set_index('Word')
print(rslt)

matplotlib.style.use('ggplot')

rslt.plot.bar(rot=0)


# common political topics and messages about it

# In[156]:


bjp_df = df[df['Message'].str.contains(pat = 'bjp')]
congress_df = df[df['Message'].str.contains(pat = 'congress')]
election_df = df[df['Message'].str.contains(pat = 'election')]
minority_df = df[df['Message'].str.contains(pat = 'minority')]
majority_df = df[df['Message'].str.contains(pat = 'majority')]


# In[158]:


bjp_df.head(10)


# In[159]:


congress_df.head(10)


# In[161]:


election_df.head(10)


# In[162]:


minority_df.head(10)


# In[165]:


majority_df.head(10)


# In[169]:


plt.figure(figsize=(9,6))
mostly_active = df['Author'].value_counts()
#Top 5 contributer in the chat 
mostly_active.head(5)


# In[170]:


m_a = mostly_active.head(5)
bars = ['David','Himanshu','Mona','Tina', 'Ayana']
x_pos = np.arange(len(bars))
m_a.plot.bar()
plt.xlabel('Authors',fontdict={'fontsize': 14,'fontweight': 10})
plt.ylabel('No. of messages',fontdict={'fontsize': 14,'fontweight': 10})
plt.title('Mostly active member of Group',fontdict={'fontsize': 20,'fontweight': 8})
plt.xticks(x_pos, bars)
plt.show()


# In[172]:


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.figure(figsize=(8,5))
t = df['Time'].value_counts().head(20)
tx = t.plot.bar()
tx.yaxis.set_major_locator(MaxNLocator(integer=True))  #Converting y axis data to integer
plt.xlabel('Time',fontdict={'fontsize': 12,'fontweight': 10})
plt.ylabel('No. of messages',fontdict={'fontsize': 12,'fontweight': 10})
plt.title('Time when Group was highly active.',fontdict={'fontsize': 18,'fontweight': 8})
plt.show()


# In[ ]:




