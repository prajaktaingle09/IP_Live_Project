#TO GENERATE  TRAINING DATASET FOR EVENT TYPE 
import numpy as np
import pandas as pd

df=pd.read_csv('event_type_training.csv',encoding='latin1')
df.head()

event_list= pd.read_csv("event_type_training.csv", usecols = ['Event'],encoding='latin1')
event_list.to_csv("intermediate_csv\\event.csv",index=True) 
event_l=pd.read_csv("intermediate_csv\\event.csv",encoding='latin1')
event_l.head()
event_l.rename( columns={'Unnamed: 0':'Index'}, inplace=True )
event_l.to_csv("intermediate_csv\\p1.csv",index=False) 



types_list = df.Type.apply(lambda x: list(x.split(",")))
types_df =pd.DataFrame({"Type":types_list})
types_df.head()

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
types_encoded = pd.DataFrame(mlb.fit_transform(types_df["Type"]),columns=mlb.classes_)
types_encoded.head()

types_encoded.to_csv("intermediate_csv\\ty.csv",index=True) 

t_l=pd.read_csv("intermediate_csv\\ty.csv",encoding='latin1')
t_l.rename( columns={'Unnamed: 0':'Index'}, inplace=True )
t_l.to_csv("intermediate_csv\\p2.csv",index=False)


a = pd.read_csv("intermediate_csv\\p1.csv",encoding='latin1')
b = pd.read_csv("intermediate_csv\\p2.csv",encoding='latin1')


merged = a.merge(b, on='Index')
merged.to_csv("intermediate_csv\\p3.csv", index=False)

f=pd.read_csv("intermediate_csv\\p3.csv",encoding='latin1')
keep_col = ['Event','Certifications','Competitions','Courses','Expos','Fests','Hackathons','Internships','Jobs','Talks','Trainings','Webinars','Workshops']
new_f = f[keep_col]
new_f.to_csv("intermediate_csv\\ffile.csv", index=False)

#----------------------------------------------------------------------------------------------------------------
#TO GENERATE  TRAINING DATASET FOR EVENT DOMAIN
import pandas as pd

df=pd.read_csv("domain_training.csv",encoding='latin1')
df.head()

event_list= pd.read_csv("domain_training.csv", usecols = ['Event'],encoding='latin1')
event_list.to_csv("intermediate_csv\\event_doma.csv",index=True) 
event_l=pd.read_csv("intermediate_csv\\event_doma.csv",encoding='latin1')
event_l.head()
event_l.rename( columns={'Unnamed: 0':'Index'}, inplace=True )
event_l.to_csv("intermediate_csv\\p_event.csv",index=False) 

#df.dropna(thresh=2)

types_list1 = df.Domain.apply(lambda x: list(x.split(",")))
types_df1 =pd.DataFrame({"Domain":types_list1})
types_df1.head()

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
types_encoded1 = pd.DataFrame(mlb.fit_transform(types_df1["Domain"]),columns=mlb.classes_)
types_encoded1.head()
types_encoded1.to_csv("intermediate_csv\\doma.csv",index=True) 

t_l=pd.read_csv("intermediate_csv\\doma.csv",encoding='latin1')
t_l.rename( columns={'Unnamed: 0':'Index'}, inplace=True )
t_l.to_csv("intermediate_csv\\p_doma.csv",index=False)


a = pd.read_csv("intermediate_csv\\p_event.csv",encoding='latin1')
b = pd.read_csv("intermediate_csv\\p_doma.csv",encoding='latin1')


merged = a.merge(b, on='Index')
merged.to_csv("intermediate_csv\\p_4d.csv", index=False)

f=pd.read_csv("intermediate_csv\\p_4d.csv",encoding='latin1')
keep_col = ['Event','Artificial Intelligence','Blockchain','C','C++','Cloud Computing','Coding','Data Science','Development Process','Finance','Hardware','Higher Education','IoT','Java','Javascript','Machine Learning','Management','Mobile Applications','Networking','Other','Python','Security','Software Architecture','Web Development']
new_f = f[keep_col]
new_f.to_csv("intermediate_csv\\ffile1.csv", index=False)

#--------------------------------------------------------------------------------------------------------------------------

#FOR TYPE PREDICTION
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

 
TRAIN_DATA = "intermediate_csv\\ffile.csv"
GLOVE_EMBEDDING = "intermediate_csv\\glove.6B.100d.txt"

train = pd.read_csv(TRAIN_DATA,encoding='latin1')

ans=["Certifications", "Competitions", "Courses", "Expos", "Fests", "Hackathons","Internships", "Jobs","Talks", "Trainings", "Webinars", "Workshops"]
x_t= train["Event"].str.lower()
y_t= train[["Certifications", "Competitions", "Courses", "Expos", "Fests", "Hackathons","Internships", "Jobs","Talks", "Trainings", "Webinars", "Workshops"]].values
max_words = 2152
max_len = 150


x_train, x_test, y_train, y_test1 = train_test_split(x_t, y_t, test_size=0.0010, random_state=42)

embed_size = 100
 
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, lower=True)

tokenizer.fit_on_texts(x_train)
 
x_train = tokenizer.texts_to_sequences(x_train)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)

vocab_size = len(tokenizer.word_index) + 1
embeddings_index = {}
 
with open(GLOVE_EMBEDDING, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        embed = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embed
 
word_index = tokenizer.word_index
 
num_words = min(max_words, len(word_index) + 1)
 
embedding_matrix = np.zeros((num_words, embed_size), dtype='float32')
 
for word, i in word_index.items():
 
    if i >= max_words:
        continue
 
    embedding_vector = embeddings_index.get(word)
 
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

input = tf.keras.layers.Input(shape=(max_len,))
 
x = tf.keras.layers.Embedding(vocab_size, embed_size, weights=[embedding_matrix], trainable=False)(input)

x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1,
                                                      recurrent_dropout=0.1))(x)
 
x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x)
 
avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
 
x = tf.keras.layers.concatenate([avg_pool, max_pool])
 
preds = tf.keras.layers.Dense(12, activation="sigmoid")(x)
 
model1 = tf.keras.Model(input, preds)
 
model1.summary()
 
model1.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])

batch_size = 128
model1.fit(x_train, y_train, validation_split=0.01, batch_size=batch_size,epochs=25,  verbose=1)

#-----------------------------------------------------------------------------------------------------
#FOR DOMAIN PREDICTION

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TRAIN_DATA1 = "intermediate_csv\\ffile1.csv"
GLOVE_EMBEDDING = "intermediate_csv\\glove.6B.100d.txt"

train1 = pd.read_csv(TRAIN_DATA1,encoding='latin1')

ans1=['Artificial Intelligence','Blockchain','C','C++','Cloud Computing','Coding','Data Science','Development Process','Finance','Hardware','Higher Education','IoT','Java','Javascript','Machine Learning','Management','Mobile Applications','Networking','Other','Python','Security','Software Architecture','Web Development']
x_t1 = train1["Event"].str.lower()
y_t1 = train1[['Artificial Intelligence','Blockchain','C','C++','Cloud Computing','Coding','Data Science','Development Process','Finance','Hardware','Higher Education','IoT','Java','Javascript','Machine Learning','Management','Mobile Applications','Networking','Other','Python','Security','Software Architecture','Web Development']].values
max_words = 2152
max_len = 150
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_t1, y_t1, test_size=0.010, random_state=42)
 
embed_size = 100
 
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words, lower=True)

tokenizer.fit_on_texts(x_train1)
 
x_train1 = tokenizer.texts_to_sequences(x_train1)

x_train1 = tf.keras.preprocessing.sequence.pad_sequences(x_train1, maxlen=max_len)

vocab_size = len(tokenizer.word_index) + 1

embeddings_index = {}
 
with open(GLOVE_EMBEDDING, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        embed = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embed
 
word_index = tokenizer.word_index
 
num_words = min(max_words, len(word_index) + 1)
 
embedding_matrix = np.zeros((num_words, embed_size), dtype='float32')
 
for word, i in word_index.items():
 
    if i >= max_words:
        continue
 
    embedding_vector = embeddings_index.get(word)
 
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

input = tf.keras.layers.Input(shape=(max_len,))
 
x1 = tf.keras.layers.Embedding(vocab_size, embed_size, weights=[embedding_matrix], trainable=False)(input)


x1= tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, return_sequences=True, dropout=0.1,
                                                      recurrent_dropout=0.1))(x1)
 
x1 = tf.keras.layers.Conv1D(64, kernel_size=3, padding="valid", kernel_initializer="glorot_uniform")(x1)
 
avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x1)
max_pool = tf.keras.layers.GlobalMaxPooling1D()(x1)
 
x1 = tf.keras.layers.concatenate([avg_pool, max_pool])
 
preds = tf.keras.layers.Dense(23, activation="sigmoid")(x1)
 
model = tf.keras.Model(input, preds)
 
model.summary()
 
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=1e-3), metrics=['accuracy'])


batch_size = 128
model.fit(x_train1, y_train1, validation_split=0.01, batch_size=batch_size,epochs=25,  verbose=1)

#--------------------------------------------------------------------------------------------------------
#PREDICTION

import csv
import numpy
filename = "test_events.csv"
raw_data = open(filename, 'rt')
reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
x = list(reader)


final_type=[]
final_domain=[]
i=0;
for xx_test in x:
    xx_test=x[i]
    xx_test = tokenizer.texts_to_sequences(xx_test)
    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 200
    xx_test = tf.keras.preprocessing.sequence.pad_sequences(xx_test, maxlen=max_len)
    predictions = model1.predict(xx_test)
    predictions1 = model.predict(xx_test)

    
    result = int(numpy.argmax(predictions, axis=1))
    final_type.append(ans[result])
    
    result1 = int(numpy.argmax(predictions1, axis=1))
    final_domain.append(ans1[result1])
    
    i=i+1
    
print(final_type)
print(final_domain)

#----------------------------------------------------------------------------
#SEARCHING

emp_list=[]
emp=pd.read_csv("CCMLEmployeeData.csv",encoding='latin1')

for t in range(5):
    z=[]
    for index,row in emp.iterrows():
        #row=row.tolist()
        if((row['Event1']==final_type[t] or row['Event2']==final_type[t]) and row['Domain']==final_domain[t]):
            z.append(row['Name'])
        elif((row['Event1']==final_type[t] or row['Event2']==final_type[t]) and "Other"==final_domain[t]):
            z.append(row['Name'])
            
    emp_list.append(z)
    
print(emp_list)

#--------------------------------------------------------------------------------------------------
#GENRATING OUTPUT
test=pd.read_csv("test_events.csv",encoding='latin1',names=['Event'])

test.to_csv("intermediate_csv\\tt2.csv",index=False) 

c=[]
 
for i in range(len(emp_list)):
     
    c.append(",".join(emp_list[i]))
 
    test['Name']=pd.Series(c)


test.to_excel("output.xlsx",index=False)