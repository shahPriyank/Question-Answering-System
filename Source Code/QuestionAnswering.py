import os
import nltk
import string
import pysolr
import json
import subprocess
import collections
import spacy

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.wsd import lesk

from stanfordcorenlp import StanfordCoreNLP
from nltk.parse import CoreNLPDependencyParser

nltk.download("wordnet")
nltk.download("stopwords")

def startSolr():
    path_for_solr = os.path.join(os.getcwd(),'solr-8.0.0\\bin')
    cmd_to_start_solr = "solr start -p 8983"
    p = subprocess.Popen(cmd_to_start_solr, cwd=path_for_solr, shell=True)
    
    cmd_to_create_core = "solr create -c NLProject"
    p2 = subprocess.Popen(cmd_to_create_core, cwd=path_for_solr, shell=True)
    
def startCoreNlp():
    cmd_to_start_CoreNLP = "java -mx4g -cp \"*\" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000"
    p=subprocess.Popen(cmd_to_start_CoreNLP, cwd='CoreNLP/stanford-corenlp-full-2018-10-05')
    
count = 0

startSolr()
startCoreNlp()

solr = pysolr.Solr('http://localhost:8983/solr/NLProject', timeout=10)
nlp = StanfordCoreNLP('http://localhost', port=9000)
nlp2 = spacy.load('en_core_web_sm')

wnl = nltk.WordNetLemmatizer()
props={'annotators': 'tokenize,ssplit,pos,depparse,ner,natlog,openie', 'openie.triple.strict':'true', 'pipelineLanguage':'en','outputFormat':'json',
       'applyNumericClassifiers':'false','useSUTime':'false','applyFineGrained':'false','buildEntityMentions':'false'}

stop_words = set(stopwords.words('english'))

ps = nltk.PorterStemmer()
ls = nltk.LancasterStemmer()
sb = nltk.stem.snowball.SnowballStemmer("english")

ner = {}
ner["WHO"] = ["PERSON", "ORGANIZATION"]
ner["WHERE"] = ["STATE_OR_PROVINCE", "CITY", "COUNTRY", "LOCATION"]
ner["WHEN"] = ["DATE", "TIME"]

def findLemma(posTag):
    lemma = set()
    for word,tag in posTag:
        if word in stop_words or word in string.punctuation or len(word) <= 2:
            continue
        if tag.startswith('N'):
            lemma.add(wnl.lemmatize(word,'n'))
        elif tag.startswith('V'):
            lemma.add(wnl.lemmatize(word,'v'))
        elif tag.startswith('J'):
            lemma.add(wnl.lemmatize(word,'a'))
        else:
            lemma.add(wnl.lemmatize(word))
    
    return list(lemma)

def depParser(sent):
    dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')
    result = dep_parser.raw_parse(sent)
    newResult = result.__next__()
    dep=list(newResult.triples())

    return dep

def findSynonymWord(word):
    synonymSet = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonymSet.add(l.name())
    return synonymSet           

def findSynonym(wordTokens):
    synonymSet = set()
    for word in wordTokens:
        synonymSet.update(findSynonymWord(word))
    
    return list(synonymSet)

def extractFeatures(wordTokens):
    hypernyms = set()
    hyponyms = set()
    meronyms = set()
    holonyms = set()
    
    for word in wordTokens:
        bestSense = lesk(wordTokens, word)
        if bestSense is not None:
            for hypernym in bestSense.hypernyms()[:2]:
                for h in hypernym.lemmas():
                    hypernyms.add(h.name())
            for hyponym in bestSense.hyponyms()[:2]:
                for h in hyponym.lemmas():
                    hyponyms.add(h.name())
            for meronym in bestSense.part_meronyms()[:2]:
                for h in meronym.lemmas():
                    meronyms.add(h.name())
            for holonym in bestSense.part_holonyms()[:2]:
                for h in holonym.lemmas():
                    holonyms.add(h.name())
                
    return list(hypernyms), list(hyponyms), list(meronyms), list(holonyms)

def findRelation(sent):
    svo = []
    output = nlp.annotate(sent, properties=props)
    result = json.loads(output)["sentences"][0]["openie"]
    for re in result:
        curr = (re['subject'],re['relation'],re['object'])
        svo.append(curr)
    return svo

def findType(sent, most_frequent):
    retType = set()
    ner = nlp.ner(sent)
    for entity in ner:
        if entity[1] == "PERSON" or entity[1] == "ORGANIZATION":
            if entity[0].lower() not in most_frequent:
                retType.add("WHO")
                continue
                
        if entity[1] == "STATE_OR_PROVINCE" or entity[1] == "CITY" or entity[1] == "COUNTRY" or entity[1] == "LOCATION":
            if entity[0].lower() not in most_frequent:
                retType.add("WHERE")
                continue
                
        if entity[1] == "DATE" or entity[1] == "TIME":
            if entity[0].lower() not in most_frequent:
                retType.add("WHEN")
    return list(retType)

def findStemWord(word):
    q = set()
    q.add(ps.stem(word))
    q.add(ls.stem(word))
    q.add(sb.stem(word))
    
    return q

def findStem(wordTokens):
    stemSet = set()
    for w in wordTokens:
        stemSet.update(findStemWord(w))
    return stemSet

def addToSolr(sent, filename, most_frequent):
    global count
    wp = "Task1\\"+filename[:-4]+".json"

    sent = sent.replace("&","")
    wordTokens = nltk.word_tokenize(sent)
    wT = [w for w in wordTokens if w not in stop_words and w not in string.punctuation and len(w)>2]
    
    posTag = nltk.pos_tag(wordTokens)
    
    lemma = findLemma(posTag)
    depParse = depParser(sent)
    synonyms = findSynonym(wT)
    hypernyms, hyponyms, meronyms, holonyms = extractFeatures(wT)
    type_of_question = findType(sent, most_frequent)
    stem = findStem(wT)
    
    x = {
        "SENTENCE_TOKENS":sent,
        "WORD_TOKENS":wordTokens,
        "POS_TAG":posTag,
        "LEMMA":lemma,
        "DEPENDENCY_PARSE":depParse,
        "SYNONYMS:":synonyms,
        "HYPERNYMS:":hypernyms,
        "HYPONYMS:":hyponyms,
        "MERONYMS:":meronyms,
        "HOLONYMS:":holonyms,
    }
    
# For task 1

#     with open(wp, 'a', encoding="utf-8") as outfile:
#         json.dump(x, outfile, indent=2)
#         outfile.write('\n')
    
    id = count
    
    solr.add([{"ID":id, "TITLE":filename, "SENTENCE":sent, "CONTENT":wT, "POS_TAG":posTag, "LEMMA":lemma, "DEPENDENCYPARSE":depParse, "SYNONYM":synonyms, "HYPERNYM":hypernyms, "HYPONYM":hyponyms,"MERONYMS":meronyms, "HOLONYMS":holonyms, "TYPE":type_of_question, "STEM": stem}])
    
    count = count + 1
    
# For task 1
# wp = "Questions.txt"
# with open(wp, 'r') as fc:
#    qts = fc.read()
#    qts = nltk.sent_tokenize(qts)
#    
#    for qt in qts:
#        addToSolr(qt, "QuestionTask1.txt")
    
def getMostFrequentWords(doc):
    mostFreq = set()
    wordCount = {}
    for sent in nltk.sent_tokenize(doc):
        for word in nltk.word_tokenize(sent):
            if word.lower() in stop_words or word in string.punctuation or len(word) <= 2:
                continue
            if word not in wordCount:
                wordCount[word] = 1
            else:
                wordCount[word] += 1
    word_counter = collections.Counter(wordCount)
    for word, count in word_counter.most_common(3):
        mostFreq.add(word)
    
    return list(mostFreq)

def createSolrSchema(path):
    
    if not os.path.exists("Task1"):
        os.mkdir("Task1")

    cmd_for_solr = "java -Dc=NLProject -jar post.jar *.xml"
    p=subprocess.Popen(cmd_for_solr, cwd='solr-8.0.0\\example\\exampledocs',shell=True)
        
    
    solr.delete(q='*:*')
    
    for filename in os.listdir(path):
        fp = path + "\\" + filename
        with open(fp, 'r',encoding="utf-8-sig") as filecontent:
            doc = filecontent.read()
            most_frequent = getMostFrequentWords(doc)
            doc = nltk.sent_tokenize(doc)
        
        for sentences in doc:
            for sent in sentences.split('\n'):
                if len(nltk.word_tokenize(sent)) >= 6:
                    addToSolr(sent, filename, most_frequent)

    p=subprocess.Popen(cmd_for_solr,cwd='solr-8.0.0\\example\\exampledocs',shell=True)
    
folder_name = input("Enter the path of articles")
question_path = input("Enter the path of question file")
createSolrSchema(folder_name)

def readQuestions(fileName):
    with open(fileName, 'r') as filecontent:
        doc = nltk.sent_tokenize(filecontent.read())
        return doc
    
def findTypeOfQuestion(question):
    if question.lower().find("who") != -1:
        return "WHO"
    elif question.lower().find("where") != -1:
        return "WHERE"
    elif question.lower().find("when") != -1:
        return "WHEN"
    

def findRoot(question):
    parsedQues = nlp2(question)
    
    for q in parsedQues:
        if q.dep_ == "ROOT":
            return (q.orth_)

def findRootInRelation(root, relation, boolToAddSynonym):
    relation = nltk.word_tokenize(relation)
    relation = [w for w in relation if w not in stop_words]
    
    rootSet = set()
    rootSet.add(root)
    rootSet.update(findStem(root))
    
    if boolToAddSynonym:
        rootSet.update(findSynonymWord(root))
    
    relationSet = set()
    
    for r in relation:
        relationSet.add(r)
        relationSet.update(findStem(r))
        
        if boolToAddSynonym:
            rootSet.update(findSynonymWord(root))
        
    if len(rootSet.intersection(relationSet)) >= 1:
        return True
    
    return False

def isNER(obj, question):
    ner = ["PERSON", "ORGANIZATION", "STATE_OR_PROVINCE", "CITY", "COUNTRY", "LOCATION", "DATE", "TIME"]
    nt = nlp.ner(question)
    for w in nt:
        if w[1] in ner:
            if w[0].lower() == obj:
                return True
    
    return False

def check(sub, candidate_words):
    for cw in candidate_words:
        cwt = nltk.word_tokenize(cw)
        for c in cwt:
            if c == sub:
                return cw
    return ""

def ansInCandWords(subject, candidate_words, obj, wT, question):
    subject = nltk.word_tokenize(subject)
    subject = [w.lower() for w in subject if w not in stop_words]
    
    obj = nltk.word_tokenize(obj)
    obj = [w.lower() for w in obj if w not in stop_words]
    
    wT = [w.lower() for w in wT]

    for sub in subject:
        ch = check(sub, candidate_words)
        if ch != "" :
            if len(obj) > 0:
                for o in obj:
                    if o not in wT:
                        return False, ""
                else:
                    return True, ch
    
    return False, ""

def findBestSentence(candidate_sents, question, wT, type_of_question):
    
    currQt = question
    
    root = findRoot(question)
    while root in ["is","was","did", "where", "when", "who"]:
        question = nltk.word_tokenize(question)
        question = [w for w in question if w != root]
        question = " ".join(question)
        root = findRoot(question)
    
    for sent in candidate_sents:
        candidate_words = set()
        
        sentStr = sent['SENTENCE'][0]
        entity = nlp.ner(sentStr)
        
        for i in range(len(entity)):
            st = ""
            while entity[i][1] in ner[type_of_question] and entity[i][0] not in wT:
                st += entity[i][0] + " "
                i = i + 1
                if i >= len(entity):
                    break
            
            if st != "":
                candidate_words.add(st.lower())
                
        
        
        candidate_words = list(candidate_words)
        output = nlp.annotate(sentStr, properties=props)
        svo = json.loads(output)["sentences"][0]["openie"]
        
        for triple in svo:
            if findRootInRelation(root, triple['relation'], False):
                b,a = ansInCandWords(triple['subject'], candidate_words, triple['object'], wT, currQt)
                if b:
                    return a,sentStr,sent['TITLE'][0]
                
                b,a = ansInCandWords(triple['object'], candidate_words, triple['subject'], wT, currQt)
                if b:    
                    return a,sentStr,sent['TITLE'][0]
    
    
    for sent in candidate_sents:
        candidate_words = set()
        
        sentStr = sent['SENTENCE'][0]
        entity = nlp.ner(sentStr)
        
        for i in range(len(entity)):
            st = ""
            while entity[i][1] in ner[type_of_question] and entity[i][0] not in wT:
                st += entity[i][0] + " "
                i = i + 1
                if i >= len(entity):
                    break
            
            if st != "":
                candidate_words.add(st.lower())
        
        candidate_words = list(candidate_words)
        output = nlp.annotate(sentStr, properties=props)
        svo = json.loads(output)["sentences"][0]["openie"]
        
        for triple in svo:
            if findRootInRelation(root, triple['relation'], True):
                b,a = ansInCandWords(triple['subject'], candidate_words, triple['object'], wT, currQt)
                if b:
                    return a,sentStr,sent['TITLE'][0]
                
                b,a = ansInCandWords(triple['object'], candidate_words, triple['subject'], wT, currQt)
                if b:    
                    return a,sentStr,sent['TITLE'][0]
    
    maxVal = 0
    maxSent = ""
    for sent in candidate_sents:
        sentStr = sent['SENTENCE'][0]
        setSent = nltk.word_tokenize(sentStr)
        setSent = [w for w in setSent if w not in stop_words]
        
        setSent = set(setSent)
        wT = set(wT)
        lenCommon = len(setSent.intersection(wT))
        if lenCommon > maxVal:
            maxVal = lenCommon
            maxSent = sent
        
    
    return "",maxSent['SENTENCE'][0], maxSent['TITLE'][0]

def getBestAnswer(sent, wT, type_of_question):
    ans = []
    a = nlp.ner(sent)
    for w in a:
        if w[1] in ner[type_of_question] and w[0] not in wT:
            ans.append(w[0])
        
    return ans

fileData = readQuestions(question_path)
finalAns = []
for question in fileData:
    question = question.translate({ord(i):None for i in '&?'})
    question = question.translate({ord(i):" " for i in '.'})
    searchData = ""
    wordTokens = nltk.word_tokenize(question)
    wT = [w for w in wordTokens if w not in stop_words and w.lower() not in ["who","where","when"]]
    
    posTag = nltk.pos_tag(wordTokens)
    lemma = findLemma(posTag)
    
    depParse = depParser(question)
    synonyms = findSynonym(wT)
    hypernyms, hyponyms, meronyms, holonyms = extractFeatures(wT)
    stem = list(findStem(wT))
    
    type_of_question = findTypeOfQuestion(question)
    
    for k in range(len(wordTokens)):
        if wordTokens[k] in stop_words or wordTokens[k] in string.punctuation or wordTokens[k] == " ":
            continue
        searchData += "CONTENT:" + wordTokens[k] + " & "
    
    for k in range(len(posTag)):
        if posTag[k][0] in stop_words or posTag[k][0] in string.punctuation or posTag[k][0] == " ":
            continue
        searchData += "POS_TAG:" + "(" + "".join(posTag[k][0]) + ", " + "".join(posTag[k][1]) + ") & "
        
    for k in range(len(lemma)):
        if lemma[k] in stop_words or lemma[k] in string.punctuation or lemma[k] == " ":
            continue
        searchData += "LEMMA:" + "".join(lemma[k]) + " & "
    
    for k in range(len(hypernyms)):
        if hypernyms[k] in stop_words or hypernyms[k] in string.punctuation or hypernyms[k] == " ":
            continue
        searchData += "HYPERNYM:" + "".join(hypernyms[k]) + " & "
        
    for k in range(len(depParse)):
        if ':' not in str(depParse[k]):
            searchData += "DEPENDENCYPARSE:" + str(depParse[k]) + " & "

    for k in range(len(synonyms)):
        searchData += "SYNONYM:" + "".join(synonyms[k]) + " & "
        
        
    for k in range(len(stem)):
        searchData += "STEM:" + "".join(stem[k]) + " & "

    searchData += "TYPE:" + type_of_question + " & "
        
    searchData = searchData.rstrip("& ")
    results = solr.search(q=searchData)
    candidate_sents = []
    for result in results:
        candidate_sents.append(result)
    ans,best,title = findBestSentence(candidate_sents, question, wT, type_of_question)
    

    ans = getBestAnswer(best, wT, type_of_question)
    x = {
        "Question":question + '?',
        "Answers:":ans,
        "Sentences:":best,
        "Document:":title
    }
    
    finalAns.append(x)

wp = "Output.json"
for ans in finalAns:
    with open(wp, 'w', encoding="utf-8") as outfile:
        json.dump(ans, outfile, indent=2)
        outfile.write('\n')
    
    
        
