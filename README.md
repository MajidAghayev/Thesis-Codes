# Thesis-Codes
Python NLP libraries' functionalities benchmark

For codes to run succesfully , all necessary libraries should be downloaded
Additionally for CoreNLP stanza, stanford corenlp should be downloaded and server should be opened. I recommend opening in localhost 9000


After redirecting to you downloaded and extracted stanfordcorenlp file path in command line, use something like this : java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 3600000 -annotators "tokenize,ssplit,pos,lemma,ner,parse,sentiment" -preload "tokenize,ssplit,pos,lemma,ner,parse,sentiment" -quiet true -maxCharLength 100000
