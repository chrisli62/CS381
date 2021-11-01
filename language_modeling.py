import math

# Calculate frequency for each word using word as a key
def recordWords(lines,words):
	for word in lines.split():
		if word in words:
			words[word] += 1
		else:
			words[word] = 1

def calculateNonOccuringTypeTokens(nf,trainWords,testWords,tokens,types):
    check = {}
    for line in nf:
        recordWords(line,testWords)
        for word in line.split():
            if word not in trainWords:
                if word not in tokens:
                    tokens[word] = 1
                else:
                    tokens[word] += 1
                if word not in types:
                    types.append(word)
            else:
                if word in check:
                    if check[word] == testWords[word]:
                        if word in tokens:
                            tokens[word] += 1
                        else:
                            tokens[word] = 1
                    else:
                        check[word] += 1
                else:
                    check[word] = 1
    check.clear()
    nf.close()

def calculateNonOccuringTypeBigramTokens(nf,trainBigram,testBigram,tokenBigram,typeBigram):
	check = {}
	for line in nf:
		createBigram(line,testBigram)
		line = line.split()
		length = len(line) - 1
		for i in range(length):
			temp = (line[i],line[i+1])
			if not temp in trainBigram:
				if not temp in tokenBigram:
					tokenBigram[temp] = 1
				else:
					tokenBigram[temp] += 1
				if not temp in typeBigram:
					typeBigram.append(temp)
			else:
				if temp in check:
					if check[temp] == testBigram[temp]:
						if temp in tokenBigram:
							tokenBigram[temp] += 1
						else:
							tokenBigram[temp] = 1
					else:
						check[temp] += 1
				else:
					check[temp] = 1
	check.clear()
	nf.close()

def replaceUnk(fp,nf,list):
	index = 0
	for line in fp:
		while list[index] in line.split():
			line = line.replace(list[index],"<unk>")
			index += 1
		nf.write(line + "\n")
	nf.close()

def sentenceReplaceUnk(sentence,words):
    for index in range(len(sentence)):
        if sentence[index] not in words:
            sentence[index] = "<unk>"
    return " ".join(sentence)

# Obtain the unigram
def unigram(log,sentenceWords,words):
	for word in sentenceWords:
		if word != "<unk>":
			if word in words:
				temp = round(math.log(words[word]/sum(words.values()),2),2)
				log.append(temp)

# Obtain the bigram
def createBigram(line,bigram):
	line = line.split()
	length = len(line) - 1
	for i in range(length):
		temp = (line[i],line[i+1])
		if not temp in bigram:
			bigram[temp] = 1
		else:
			bigram[temp] += 1

def logBigram(log,bigram,sentenceBigram):
    for temp in sentenceBigram:
      temp = round(math.log(sentenceBigram[temp]/sum(bigram.values())),2)
      log.append(temp)

def addOneSmoothing(bigram):
	for temp in bigram:
		bigram[temp] += 1

def processing(files):

    pre = "<s> "
    post = " </s>"

    # Initialize list and dictionaries
    words = {}
    bigram = {}
    testWords = {}
    testBigram = {}
    noTestTokens = {}
    noTestTypes = []
    noTestBigramTokens = {}
    noTestBigramTypes = []
    testLog = []
    testBigramLog = []
    testSmoothLog = []


    for f in files:
        # Pad each sentence in the training and test corpora with start <s> and end symbol </s>
        with open(f) as fp:
            nf = open(f"new_{f}", "w")
            for line in fp:
                nf.write(pre + line.lower().strip() + post + '\n')
            nf.close()

        if f == "train.txt":
            nf = open(f"new_{f}", "r")
            for line in nf:
                recordWords(line,words)
            count = sum(words.values())
            onceOnly = [word for word in words.keys() if words.get(word) == 1]
            typeWords = len(words) - len(onceOnly) + 1
            nf.close()

        if f == "test.txt":
            nf = open(f"new_{f}", "r")
            calculateNonOccuringTypeTokens(nf,words,testWords,noTestTokens,noTestTypes)
            testCount = sum(testWords.values())
            onceTest = [word for word in testWords.keys() if testWords.get(word) == 1]

    for f in files:
        with open(f"new_{f}") as fp:
            if f == "train.txt":
                nf = open(f"unk_{f}", "w")
                replaceUnk(fp,nf,onceOnly)
                nf = open(f"unk_{f}", "r")
                for line in nf:
                    createBigram(line,bigram)
                nf.close()

            if f == "test.txt":
                nf = open(f"unk_{f}", "w")
                replaceUnk(fp,nf,onceTest)
                nf = open(f"unk_{f}", "r")
                calculateNonOccuringTypeBigramTokens(nf,bigram,testBigram,noTestBigramTokens,noTestBigramTypes)
    
    # Section 1.2 Training Models
    unigram(testLog,testWords,words)
    logBigram(testBigramLog,bigram,testBigram)
    testSmooth = testBigram.copy()
    addOneSmoothing(testSmooth)
    logBigram(testSmoothLog,bigram,testSmooth)

    #Perplexity calculation for each model for test.txt
    testPerplexity = round(2**(-1*(sum(testLog)/testCount)),4)
    testBigramPerplexity = round(2**(-1*(sum(testBigramLog)/testCount)),4)
    testSmoothPerplexity = round(2**(-1*(sum(testSmoothLog)/testCount)),2)

    # Percentage calculations for Section 1.3 Question 2
    try:
        testTokenPercentage = sum(noTestTokens.values())/testCount * 100
        testTypePercentage = len(noTestTypes)/len(testWords) * 100

        testBigramTokenPercentage = sum(noTestBigramTokens.values())/sum(testBigram.values()) * 100
        testBigramTypePercentage = len(noTestBigramTypes)/len(testBigram) * 100
    
    except ZeroDivisionError as e:
        print(e)

    # Section 1.3 Question 5 Model
    modelLog = []
    modelBigramLog = []
    modelSmoothLog = []
    modelWords = {}
    modelBigram = {}
    modelSmooth = {}

    model = "<s> i look forward to hearing your reply . </s>"

    model = model.split()
    model = sentenceReplaceUnk(model,words)
    recordWords(model,modelWords)
    unigram(modelLog,modelWords,words)
    createBigram(model,modelBigram)
    logBigram(modelBigramLog,bigram,modelBigram)

    # Smoothing
    modelSmooth = modelBigram.copy()
    addOneSmoothing(modelSmooth)
    logBigram(modelSmoothLog,bigram,modelSmooth)

    # Perplexity calculation for model
    modelPerplexity = round(2**(-1*(sum(modelLog)/len(model))),2)
    modelBigramPerplexity = round(2**(-1*(sum(modelBigramLog)/len(model))),2)

    # Prints answers to Section 1.3 in order of questions
    print('Total word types in the training corpus: ' + str(typeWords) + '\n')
    print('Total word tokens in the training corpus: ' + str(count) + '\n')
    print('\n')
    print('Percentage of word tokens in test.txt that did not occur is: ' + str(round(testTokenPercentage,4)) + '%' + '\n')
    print('Percentage of word types in test.txt that did not occur is: ' + str(round(testTypePercentage,4)) + '%' + '\n')
    print('\n')
    print('Percentage of bigram tokens in test.txt that did not occur is: ' + str(round(testBigramTokenPercentage,4)) + "%" + '\n')
    print('Percentage of bigram types in test.txt that did not occur is: ' + str(round(testBigramTypePercentage,4)) + '%' + '\n')
    print('\n')
    print(model)
    print('Unigram:')
    print('log2(p(<i>))+log2(p(<look>))+log2(p(<forward>))+log2(p(<to>))+log2(p(<hearing>))+log2(p(<your>))+log2(p(<reply>))+log2(p(<.>))+log2(p(</s>))')
    print(str(modelLog[0]) + str(modelLog[1]) + str(modelLog[2]) + str(modelLog[3]) + str(modelLog[4]) + str(modelLog[5]) + str(modelLog[6]) + str(modelLog[7]) + str(modelLog[8]) + str(modelLog[9]) + " = " + str(round(sum(modelLog),2)))
    print('Bigram:')
    print('log2(p(<i>|<s>))+log2(p(<look>|<i>))+log2(p(<forward>|<look>))+log2(p(<to>|<forward>))+log2(p(<hearing>|<to>))+log2(p(<your>|<hearing>))+log2(p(<reply>|<your>))+log2(p(<.>|<reply>))+log2(p(</s>|<.>))')
    print(str(modelBigramLog[0]) + str(modelBigramLog[1]) + str(modelBigramLog[2]) + str(modelBigramLog[3]) + str(modelBigramLog[4]) +str(modelBigramLog[5]) + str(modelBigramLog[6]) + str(modelBigramLog[7]) + str(modelBigramLog[8]) + " = " + str(round(sum(modelBigramLog),2)))
    print('Add one smoothing:')
    print('log2(p(<i>|<s>))+log2(p(<look>|<i>))+log2(p(<forward>|<look>))+log2(p(<to>|<forward>))+log2(p(<hearing>|<to>))+log2(p(<your>|<hearing>))+log2(p(<reply>|<your>))+log2(p(<.>|<reply>))+log2(p(</s>|<.>))')
    print(str(modelSmoothLog[0]) + str(modelSmoothLog[1]) + str(modelSmoothLog[2]) + str(modelSmoothLog[3]) + str(modelSmoothLog[4]) +str(modelSmoothLog[5]) + str(modelSmoothLog[6]) + str(modelSmoothLog[7]) + str(modelSmoothLog[8]) + " = " + str(round(sum(modelSmoothLog),2)))
    print('\n')
    print('Unigram perplexity for the model is: ' + str(modelPerplexity) + '\n')
    print('Bigram perplexity for the model is: ' + str(modelBigramPerplexity) + '\n')
    print('\n')
    print('Unigram perplexity for test.txt is: ' + str(testPerplexity) + '\n')
    print('Bigram perplexity for test.txt is: ' + str(testBigramPerplexity) + '\n')
    print('Smoothing perplexity for test.txt is: ' + str(testSmoothPerplexity) + '\n')


def main():
	
    train = "train.txt"
    test = "test.txt"
    files = (train, test)
    processing(files)


if __name__ == "__main__":
    main()