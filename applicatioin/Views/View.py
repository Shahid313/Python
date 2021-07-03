from flask_classful import FlaskView, route
from applicatioin import db
from flask import render_template, request
from flask import redirect, url_for
from applicatioin.forms.forms import InputForm
from fuzzywuzzy import fuzz
import re
from applicatioin.Builders.ModelBuilder import *

class InputView(FlaskView):
	def check_plagiarism(self, a, b):
		a= re.sub("[^a-zA-Z]", "", a)
		b= re.sub("[^a-zA-Z]", "", b)
		print(fuzz.token_sort_ratio(a,b))
		return fuzz.token_sort_ratio(a,b)

	def count_pos(essay):
    
	    tokenized_sentences = essay_to_sentences(essay, remove_stopwords=True)
	    
	    noun_count = 0
	    adj_count = 0
	    verb_count = 0
	    adv_count = 0
	    response = {}
	    
	    for sentence in tokenized_sentences:
	        tagged_tokens = nltk.pos_tag(sentence)
	        
	        for token_tuple in tagged_tokens:
	            pos_tag = token_tuple[1]
	        
	            if pos_tag.startswith('N'): 
	                noun_count += 1
	            elif pos_tag.startswith('J'):
	                adj_count += 1
	            elif pos_tag.startswith('V'):
	                verb_count += 1
	            elif pos_tag.startswith('R'):
	                adv_count += 1
	            
	    response.update({"noun":noun_count,'adj': adj_count,'verbs': verb_count,
	    	'adverbs': adv_count})
	    return response

	def spell_check(essay,suggest):
	    import enchant
	    d=enchant.Dict("en_US")
	    c=0
	    response = {}
	    for i in range(len(essay.split(" "))):
	        
	        a=essay.split(" ")
	        b=a[i]
	        e=""
	        e=e.join(b)
	        e= re.sub("[^a-zA-Z]", "", e)
	        b=""
	        if(len(e)):
	            if (d.check(e) == False):
	                c=c+1 
	                if suggest:
	                	response.update({str(e): d.suggest(e)})
	        else:
	            pass
	    return response

	def word_count(essay):
	    words=essay_to_wordlist(essay, remove_stopwords=False)
	    return len(words)

	def most_frequent_words(essay):
	    words=essay_to_wordlist(essay, remove_stopwords=True)
	    allWordDist = nltk.FreqDist(w for w in words)
	    t_list=[]
	    for i in range(10):
	        t_list.append(allWordDist.most_common(10)[i][0])
	    return t_list

	@route('/',methods=['POST','GET'])
	def input_text(self):
		form = InputForm()
		if request.method == 'POST':
			if form.validate_on_submit():
				if form.generate_report.data:
					essay = form.input_field_one.data
					mfw_list = most_frequent_words(essay)
					wordCount = word_count(essay)
					spellCheck =spell_check(essay,suggest=True)
					part_of_speech = count_pos(essay)
					return render_template('index.html', form=form, plg=None, most_frequent_words=str(mfw_list),
						word_count=wordCount, spell_check=str(spell_check), part_of_speech=str(part_of_speech))
				else:
					text_one = form.input_field_one.data
					text_two = form.input_field_two.data
					print(text_one)
					return render_template("index.html", form=form,
						plg=str(self.check_plagiarism(text_one, text_two)))
		return render_template('index.html', form=form, plg=None)


