import pandas as pd
import re
import numpy as np

df = pd.read_json('/content/drive/MyDrive/Dataset NER/goldstandard-0811.json')

def unique(list1):
    x = np.array(list1)
    print(np.unique(x))

dict_fix2 = []
for datadf in df['paragraphs']:
  text = []
  labels_fix = []
  for data in datadf:
    sentence = []
    labels = []
    for data1 in data['sentences']:
      words = []
      label = []
      
      for data2 in data1['tokens']:
        words.append(data2['orth'])
        label.append(data2['ner'])
        
      sentence.append(words)
      label = [x.replace('U-rson', 'Person').replace('U-ganisation', 'Organization').replace('U-ace', 'Location') for x in label] #Replace based on you case
      labels.append(label)
    sentences = [' '.join(x) for x in sentence]
    
    text.extend(sentences)
    labels_fix.append(labels)
  
  position = []
  for sentence in text:
    end = [ ele.end() - 1 for ele in re.finditer(r'\S+', sentence)]
    start = [ele.start() for ele in re.finditer(r'\S+', sentence)]
    position.append([[x,y] for x,y in zip(start,end)])
  
  json2 = []
  for label, pos, sent in zip(labels_fix[0], position, text):
    ent = []
    for tag, a in zip(label, pos):
      if tag!='O':
        ent.append(tuple(a+ [tag]))
      else:
        pass  
    json2.append((sent, {'entities':ent}))
  dict_fix2.append(json2[0])