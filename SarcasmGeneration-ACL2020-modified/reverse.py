from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from loadconfig import loadConfig
import sys
# CAMBIO: Se importa librería para corregir frases en inglés.
from language_tool_python import LanguageTool, utils


def getWordNetAntonyms():
	m= {}
	for line in open('./data/antonyms.txt'):
		m[line.strip().split()[0]] = line.strip().split()[1]
		# CAMBIO: Grabamos los antónimos en doble dirección.
		m[line.strip().split()[1]] = line.strip().split()[0]
	return m


def findIfnegationPresent(utterance):
	words = utterance.split()
	for w in words:
		if w=='not' or w=='never' or  w=='Not' or w=='Never':
			return w,True
	return '',False


def findIfendingwithnt(utterance):
	# CAMBIO: Añidamos las formulas "isn't", "aren't", "won't", "couldn't", "mustn't", "shan't", "needn't", "haven't", "hasn't", "wasn't" y "weren't".
	d = {"didn't": "did", "don't": "do", "doesn't": "does", "can't": "can",
		"cannot": "can", "wouldn't": "would", "shouldn't": "should",
		"isn't": "is", "aren't": "are", "won't": "will", "couldn't": "could",
		"mustn't": "must", "shan't": "shall", "needn't": "need",
		"haven't": "have", "hasn't": "has", "wasn't": "was", "weren't": "were"}
	words = utterance.split()
	for w in words:
		if w in d:
			return w,d[w],True
		if w.lower() in d:
			return w,d[w.lower()].capitalize(),True
	return '','',False


# CAMBIO: Incluimos 'antonyms' como parámetro para evitar cargar la lista de antónimos cada vez que llamamos a la función.
def getAntonym(word, antonyms):
	if word.lower() not in antonyms:
		synonymsset = []
		antonymsset = []
		for syn in wn.synsets(word.lower()):
			for l in syn.lemmas():
				synonymsset.append(l.name())
				if l.antonyms():
					antonymsset.append(l.antonyms()[0].name())
		if len(antonymsset)==0:
			for w in synonymsset:
				if w in antonyms:
					return antonyms[w.lower()]
			return "not "+word
		else:
			# CAMBIO: Formamos la palabra en pasado simple si es necesario.
			if word.endswith('ed') and word_exists(antonymsset[0]+'ed'):
				antonymsset[0] += "ed"
			return antonymsset[0]
	else:
		return antonyms[word.lower()]


def ifTwoNegation(utterance):
	exception_vadarneg_words, missing_vadarneg_words= loadConfig('ROV')
	utterance = utterance.replace(',','')
	sid = SentimentIntensityAnalyzer()
	arr = []
	sent = word_tokenize(utterance)
	for i in range(len(sent)):
		w = sent[i]
		if w == 'no':
			continue
		ss = sid.polarity_scores(w)
		if (ss['neg']==1.0 or w in missing_vadarneg_words) and (w not in exception_vadarneg_words):
			arr.append((w,i,abs(ss['compound'])))
	if len(arr)==2:
		if abs(arr[0][1]-arr[1][1])==2:
			return [arr[0][0],arr[1][0]],True
		else:
			return [arr[1][0]],True
	else:
		return [],False


def isThereOnlyOneNegation(utterance):
	exception_vadarneg_words, missing_vadarneg_words= loadConfig('ROV')
	sid = SentimentIntensityAnalyzer()
	count = 0
	word = ''
	arr = []
	for w in word_tokenize(utterance):
		if w=='no':
			continue
		ss = sid.polarity_scores(w)
		if ss['neg']==1.0 and w not in exception_vadarneg_words:
			count = count+1
			if count<=1:
				word = w
			arr.append(word)
		elif w in missing_vadarneg_words and count==0:
			count = count+1
			if count<=1:
				word = w
	if count==1:
		return word,True
	return 'cant_change',False


# CAMBIO: Función para comprobar que una palabra existe en WordNet (manejo de formas no lematizadas).
def word_exists(word):    
	lemma = WordNetLemmatizer().lemmatize(word.lower())
	synsets = wn.synsets(lemma)
	return bool(synsets)


# CAMBIO: Función para corregir frases de entrada.
def correct_sentence(sentence, tool):
	pending_close = tool is None
	if pending_close:
		tool = LanguageTool('en-US')
	correction = tool.check(sentence)
	sentence_corrected = utils.correct(sentence, correction)
	if sentence_corrected[-1] != '.':
		sentence_corrected += '.'
	return sentence_corrected


# CAMBIO: Función para negar verbos modales.
def modal_negation(word):
	d = {"can": "can't", "could": "couldn't", "must": "mustn't", "ought": "ought not",
		"should": "shouldn't", "shall": "shan't", "will": "won't", "would": "wouldn't",
		"need": "needn't", "may": "may not", "might": "might not"}
	if word in d:
		return d[word], True
	return word, False


# CAMBIO: Se incluye 'antonyms' como parámetro opcional para evitar cargar la lista de antónimos en cada ejecución.
# CAMBIO: Se incluye 'correction_tool' como parámetro opcional para evitar cargar la herramienta de corrección en cada ejecución.
def reverse_valence(utterance, antonyms=None, correction_tool=None):
	# CAMBIO: En caso de no recibir los antónimos por parámetro, se cargan llamando a la función correspodiente una sola vez.
	if antonyms is None:
		antonyms = getWordNetAntonyms()
	# CAMBIO: Corregimos posibles errores formulación de la frase de entrada.
	utterance = correct_sentence(utterance, correction_tool)
	#directly handle these without going for complicated logic
	utterance = utterance.lower()
	utterance = utterance.replace(' i ',' I ')
	if 'should be' in utterance or 'would be' in utterance:
		return utterance.replace(' be ',' not be ').capitalize()
	if ' need to ' in utterance:
		return utterance.replace(' need to ',' need not ').capitalize()
	if 'hate' in utterance:
		return utterance.replace('hate','love').capitalize()
	# CAMBIO: Se formula el caso anterior hate-love de forma inversa.
	if 'love' in utterance:
		return utterance.replace('love','hate').capitalize()
	if 'least' in utterance:
		return utterance.replace('least','most').capitalize()
	# CAMBIO: Se formula el caso anterior least-most de forma inversa.
	if 'most' in utterance:
		return utterance.replace('most','least').capitalize()	
	if utterance.endswith('lies.'):
		return utterance.replace('lies','truth').capitalize()
	# CAMBIO: Afirmamos las oraciones negativas en presente simple.
	if "don't" in utterance:
		return utterance.replace("don't ",'').capitalize()

	#check if negation present , in terms of single or double words or not/n't words
	word,verdict = findIfnegationPresent(utterance)
	negword,replneg,verdict1 = findIfendingwithnt(utterance)
	words,verdict3 = ifTwoNegation(utterance)
	negative, verdict2 = isThereOnlyOneNegation(utterance)

	#handle case by case , give priority to remove not first
	if verdict == True:
		return utterance.replace(word+' ','').capitalize()
	elif verdict1==True and verdict2==False:
		# CAMBIO: Se indica que únicamente se reemplace una de las palabras negativas.
		return utterance.replace(negword,replneg,1).capitalize()
	elif verdict3==True:		
		for w in words:
			if getAntonym(w, antonyms).startswith('not'):
				continue
			utterance = utterance.replace(w,getAntonym(w, antonyms))
		return utterance.capitalize()
	else:
		prev_utterance = utterance		
		# CAMBIO: Se limita la sustitución del antónimo de la palabra negativa a cuando el antónimo no comienza por "not" o no es un sustantivo.
		if verdict2 == True:
			for a,b in pos_tag(word_tokenize(utterance)):
				if (negative == a) and ((b != "NN") or not getAntonym(negative, antonyms).startswith('not')):
					utterance = utterance.replace(negative,getAntonym(negative, antonyms))
					break
		#incase algorithm could not handle still try to negate
		#cases replace present tense verbs by appending a don't
		#cases replace unique words prefixing with un 
		if utterance == prev_utterance:
			text = word_tokenize(utterance)
			pos_text = pos_tag(text)			
			for a,b in pos_text:
				# CAMBIO: Se añade la forma verbal base para incluir formas como el imperativo.
				if b in ['VBP', 'VB']:
					# CAMBIO: Se corrige la negación del verbo "to be" para la primera persona del singular y las formas en plural.
					if a in ["am", "are"]:
						utterance = utterance.replace(a,a+" not",1)
						break
					utterance = utterance.replace(a,"don't "+a,1)
					break
				# CAMBIO: Se añade la negación de verbos en presente de tercera persona del singular.
				if b == 'VBZ':
					# CAMBIO: Se corrige la negación del verbo "to be" en tercera persona del singular.
					if a=='is':
						utterance = utterance.replace(" is "," is not ",1)
						break
					if a=="'s":
						utterance = utterance.replace(a," is no",1)
						break
					# CAMBIO: Se trata como excepción el verbo "have got" en tercera persona del singular.
					if a=='has':
						utterance = utterance.replace(a," has not",1)
						break
					utterance = utterance.replace(a,"doesn't "+a[:-1],1)
					break
				# CAMBIO: Se corrige la negación del verbo "to be" en pasado simple.
				if b == 'VBD':
					if a=='was':
						utterance = utterance.replace(" was "," was not ",1)
						break
					if a=='were':
						utterance = utterance.replace(" were "," were not ",1)
						break					
				# CAMBIO: Se niegan los verbos modales.
				if b == "MD":
					modal, present = modal_negation(a)
					if present:
						utterance = utterance.replace(a,modal,1)
						break	
				# CAMBIO: Se comprueba que exista la palabra sin el prefijo 'un'.
				if a.startswith('un') and word_exists(a[2:]):
					utterance = utterance.replace(a,a[2:],1)
					break
				# CAMBIO: Se comprueban otros prefijos de negación que contienen 2 letras.
				if b == "JJ" and a.startswith(('in', 'im', 'il', 'ir', 'de')) and word_exists(a[2:]):
					utterance = utterance.replace(a,a[2:],1)
					break
				# CAMBIO: Se comprueban otros prefijos de negación que contienen 3 letras.
				if b == "JJ" and a.startswith('dis') and word_exists(a[3:]):
					utterance = utterance.replace(a,a[3:],1)
					break
				# CAMBIO: Forzamos la búsqueda de antónimos.
				if (a in antonyms) and (b in ["JJ", "JJS", "NN", "VB"]):
					utterance = utterance.replace(a,antonyms[a],1)
					break
		if utterance == prev_utterance:
			# CAMBIO: Sustituimos preposiciones por términos opuestos.
			# CAMBIO: Preposiciones a sustituir de prioridad alta.
			for a,b in pos_text: 
				if b == 'IN':
					if a == 'with':
						return utterance.replace(' with ',' without ').capitalize()
					if a == 'without':
						return utterance.replace(' without ',' with ').capitalize()
					if a == 'above':
						return utterance.replace(' above ', ' below ').capitalize()
					if a == 'below':
						return utterance.replace(' below ', ' above ').capitalize()
			# CAMBIO: Preposiciones a sustituir de prioridad baja.
			for a,b in pos_text:
				if b == 'IN':					
					if a == 'in':
						return utterance.replace(' in ',' out ').capitalize()
					if a == 'out':
						return utterance.replace(' out ',' in ').capitalize()					
					if a == 'on':
						return utterance.replace(' on ',' off ').capitalize()
					if a == 'off':
						return utterance.replace(' off ',' on ').capitalize()
			# CAMBIO: Si no se han aplicado cambios todavía, negamos el gerundio de las frases nominales.
			for a,b in pos_text:
				if b == "VBG":
					singular = True
					for c,d in pos_text:
						if (a == c) and (b == d):
							break	
						singular = True if d == "NN" else False if d == "NNS" else singular
					verb = "is" if singular else "are"
					utterance = utterance.replace(a,verb+" not "+a,1)
					break
		utterance = utterance.split()
		for i in range(len(utterance)):
			# CAMBIO: Se mantiene el artículo "an" delante de palabras comenzadas por "h".
			if utterance[i] == 'an' and utterance[i+1][0] not in ['a','e','i','o','u','h']:
				utterance[i] = 'a'
		utterance = ' '.join(utterance)
		# CAMBIO: Se añade un punto a final de la frase.
		if utterance[-1] != '.':
			utterance += '.'
		return utterance.capitalize()


# CAMBIO: Descomentamos la línea y la condicionamos a que se esté ejecutando el módulo como programa principal y a que se reciba el argumento.
if __name__ == "__main__" and len(sys.argv) > 1:
	print(reverse_valence(sys.argv[1]))