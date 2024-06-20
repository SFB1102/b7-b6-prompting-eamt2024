#! /usr/bin/python3
# coding: utf-8

'''
this sctipt contains only the lang-independent functions
to analyse conllu and to be imported into dev and production scripts
'''

import numpy as np
from igraph import *
import itertools
import re
from collections import OrderedDict
from operator import itemgetter
import warnings
warnings.simplefilter("ignore")


def get_meta(input):
	#/home/masha/02targets_wlv/data/croco/pro/de
	# prepare for writing metadata
	lang_folder = len(os.path.abspath(input).split('/')) - 1
	status_folder = len(os.path.abspath(input).split('/')) - 2
	korp_folder = len(os.path.abspath(input).split('/')) - 3
	
	status0 = os.path.abspath(input).split('/')[status_folder]
	korp0 = os.path.abspath(input).split('/')[korp_folder]
	# register0 = os.path.abspath(input).split('/')[reg_folder]
	lang0 = os.path.abspath(input).split('/')[lang_folder]

	return lang0,korp0,status0 ##register0,


def get_HTQmeta(input):
	# /home/masha/02targets_wlv/HTQ/htq40sets/bad/ru
	# prepare for writing metadata
	lang_folder = len(os.path.abspath(input).split('/')) - 1
	group_folder = len(os.path.abspath(input).split('/')) - 2
	
	group0 = os.path.abspath(input).split('/')[group_folder]
	lang0 = os.path.abspath(input).split('/')[lang_folder]
	
	return lang0, group0 ##register0,

# string = get_meta(arg)
# print(string)

def get_trees(data): # data is one object: a text or all of corpus as one file
	sentences = []
	only_punct = []
	current_sentence = []  # определяем пустой список
	for line in data:  # fileinput.input():  # итерируем строки из обрабатываемого файла
		if line.strip() == '':  # что делать есть строка пустая (это граница предложения!):
			if current_sentence:  # и при этом в списке уже что-то записано
				sentences.append(current_sentence)
			# if only_punct:
			# 	# if set(only_punct) == 2:
			# 		print('GOTCHA', only_punct)
			current_sentence = []  # обнуляем список
			only_punct = []
			# if the number of sents can by divided by 1K without a remainder.
			# В этом случае, т.е. после каждого 1000-ного предложения печатай месседж. Удобно!
			#         if len(sentences) % 1000 == 0:
			#             print('I have already read %s sentences' % len(sentences))
			continue
		if line.strip().startswith('#'):
			continue
		res = line.strip().split('\t')
		(identifier, token, lemma, upos, xpos, feats, head, rel, misc1, misc2) = res
		if '.' in identifier or '-' in identifier:  # ignore empty nodes possible in the enhanced representations
			continue
		# во всех остальных случаях имеем дело со строкой по отдельному слову
		# in ref_RU data there are sentences that consist of 4 PUNCTs only!
		for i in res:
			# print(res)
			only_punct.append(res[3])
		var = list(set(only_punct))
		# throw away sentences that consist of just PUNCT, particularly rare 4+ PUNCT
		if len(var) == 1 and var[0] == 'PUNCT':
			# print('GOTCHA', set(only_punct))
			continue
		else:
			current_sentence.append((int(identifier), token, lemma, upos, xpos, feats, int(head), rel))
	## это записывает последнее предложение, не заканчиающееся пустой строкой? НЕЕЕТ, там есть пустая строка
	if current_sentence:
		# if len(current_sentence) < 3:
		# 	print('+++', current_sentence)
		sentences.append(current_sentence)  # получаем список предложений из файла
	
	sentences = [s for s in sentences if len(s) >= 4] ### here is where I filter out short sents
	# print('==', len(sentences))
	return sentences


## functions to traverse the trees
def get_headwd(node, sentence): # when calling, test whether head exists --- if head:
	head_word = None
	head_id = node[5]
	# print(own_id)
	for word in sentence:
		if head_id == word[0]:
			head_word = word
	return head_word


def get_kids(node, sentence):
	kids = []
	own_id = node[0]
	# print(own_id)
	for word in sentence:
		if own_id == word[5]:
			kids.append(word)
	return kids  # requires iteration of children to get info on individual properties


def choose_kid_by_featrel(node, sentence, feat, rel):
	targetedkid_ind = None
	kids = get_kids(node, sentence)
	for kid in kids:
		# specify kids features
		if kid[6] == rel and feat in kid[4]:
			targetedkid_ind = kid[0]

	return targetedkid_ind


def choose_kid_by_posfeat(node, sentence, pos, feat):
	targetedkid_ind = None
	kids = get_kids(node, sentence)
	for kid in kids:
		# specify kids features
		if kid[3] == pos and feat in kid[4]:
			targetedkid_ind = kid[0]

	return targetedkid_ind


def choose_kid_by_posrel(node, sentence, pos, rel):
	targetedkid_ind = None
	kids = get_kids(node, sentence)
	for kid in kids:
		# specify kids features
		if kid[3] == pos and rel in kid[6]:
			targetedkid_ind = kid[0]

	return targetedkid_ind


def choose_kid_by_lempos(node, sentence, lemma, pos):
	targetedkid_ind = None
	kids = get_kids(node, sentence)
	for kid in kids:
		# specify kids features
		if kid[3] == pos and kid[2] == lemma:
			targetedkid_ind = kid[0]
	return targetedkid_ind


def has_auxkid_by_lem(node, sentence, lemma):
	res = False
	kids = get_kids(node, sentence)
	for kid in kids:
		# specify kids features
		if kid[3] == 'AUX' and kid[2] == lemma:
			res = True


def has_kid_by_lemlist(node, sentence, lemmas):
	res = False
	kids = get_kids(node, sentence)
	for kid in kids:
		# specify kids features
		if kid[2] in lemmas:
			res = True
	return res


def has_auxkid_by_tok(node, sentence, token):
	res = False
	kids = get_kids(node, sentence)
	for kid in kids:
		# specify kids features
		if kid[3] == 'AUX' and kid[1] == token:
			res = True
	
	return res  # test if True or False


# list of dependents pos
def get_kids_pos(node, sentence):  # there are no native tags (XPOS) in Russian corpora
	kids_pos = []
	own_id = node[0]
	for word in sentence:
		if own_id == word[5]:
			kids_pos.append(word[3])
	return kids_pos

#
# # list of dependents pos
# def get_kids_xpos(node, sentence):  # there are no native tags (XPOS) in Russian corpora
# 	kids_xpos = []
# 	own_id = node[0]
# 	for word in sentence:
# 		if own_id == word[5]:
# 			kids_xpos.append(word[4])
# 	return kids_xpos


# list of dependents dependency relations to the node
def get_kids_rel(node, sentence):
	kids_rel = []
	own_id = node[0]
	for word in sentence:
		if own_id == word[5]:
			kids_rel.append(word[4])
	return kids_rel


# flattened list of grammatical values; use with care in cases where there are many dependents
# -- gr feature can be on some other dependent
def get_kids_feats(node, sentence):
	deps_feats0 = []
	deps_feats1 = []
	deps_feats = []
	own_id = node[0]
	for word in sentence:
		if own_id == word[5]:
			deps_feats0.append(word[4])
		# split the string of several features and flatten the list
		for el in deps_feats0:
			try:
				el_lst = el.split('|')
				deps_feats1.append(el_lst)
			except:
				deps_feats1.append(el)
		deps_feats = [y for x in deps_feats1 for y in x]  # flatten the list
	return deps_feats


# list of dependents lemmas
def get_kids_lem(node, sentence):
	kids_lem = []
	own_id = node[0]
	for word in sentence:
		if own_id == word[5]:
			kids_lem.append(word[2])
	return kids_lem


def get_prev(node, sentence):
	prev = None
	for i,w in enumerate(sentence):
		if w == node:
			node_id = i
			prev = sentence[node_id-1]
	
	return prev

###########################################################################
###########################################################################
## file-level counts for normalisation

def wordcount(trees):
	words = 0
	for tree in trees:
		words += len(tree)
	
	return words

## this function adjusts counts for number of sentences, accounting for ca. 4% of errors in EN/GE
# where sentence ends with colon or semi-colon; in RU this error makes only 0.4%
def sents_num(trees, lang):
	sentnum = 0
	if lang == 'en':
		for tree in trees:
			lastwd = tree[-1]
			if not lastwd[2] in [':', ';', 'Mr.', 'Dr.']:
				sentnum += 1
	if lang == 'de':
		for tree in trees:
			lastwd = tree[-1]
			if not lastwd[2] in [':', ';', 'z.B.', 'Dr.']:
				sentnum += 1
	if lang == 'ru':
		for tree in trees:
			lastwd = tree[-1]
			if not lastwd[2] in [':', ';', 'Дж.']:
				sentnum += 1  # this is a fair num-of-sents count for a file
	
	return sentnum

def verbs_num(trees, lang):
	verbs = 0
	for tree in trees:
		for w in tree:
			if w[3] == 'VERB':
				verbs += 1
	return verbs

###############################################################
def freqs_dic(trees, func, lang):
	dic = {}
	tot = 0
	for tree in trees:
		intree, lst = func(tree, lang)
		tot += intree
		for i in set(lst):
			freq = lst.count(i)
			if i in dic:
				dic[i] += freq
			else:
				dic[i] = freq
	
	dic_sort = OrderedDict(sorted(dic.items(), key=itemgetter(1), reverse=True))
	# print(list(dic_sort.items())[:100])
	tuples = list(dic_sort.items())[:100]
	for tu in tuples:
		print(':'.join(i for i in [tu[0],str(tu[1])]), end ="; ")
	print("Dict size", len(list(dic_sort.items())))
	
	return tot

def support_all_lang(lang):

	if lang == 'en':
		
		file0 = "/home/masha/02targets_wlv/code/extract/get_features/searchlists/en_deverbals_stop.lst"
		stoplist = open(file0, 'r').readlines()
		pseudo_deverbs = []
		for wd in stoplist:
			wd = wd.strip()
			pseudo_deverbs.append(wd)
		
		file1 = open("/home/masha/02targets_wlv/code/extract/get_features/searchlists/en_adv_quantifiers.lst", 'r').readlines()
		quantifiers = []
		for q in file1:
			q = q.strip()
			quantifiers.append(q)

		file2 = open("/home/masha/02targets_wlv/code/extract/get_features/searchlists/en_modal-adj_predicates.lst", 'r').readlines()
		adj_pred = []
		for adj in file2:
			adj = adj.strip()
			adj_pred.append(adj)

		file3 = open("/home/masha/02targets_wlv/code/extract/get_features/searchlists/en_converts.lst", 'r').readlines()
		# file3 = open("/home/masha/02targets_wlv/code/extract/get_features/searchlists/en_converts.lst", 'r').readlines()
		converts = []
		for conv in file3:
			conv = conv.strip()
			converts.append(conv)

	if lang == 'de':
		file0 = "/home/masha/02targets_wlv/code/extract/get_features/searchlists/de_deverbals_stop.lst"
		stoplist = open(file0, 'r').readlines()
		pseudo_deverbs = []
		for wd in stoplist:
			wd = wd.strip()
			pseudo_deverbs.append(wd)
		
		file1 = open("/home/masha/02targets_wlv/code/extract/get_features/searchlists/de_adv_quantifiers.lst", 'r').readlines()
		quantifiers = []
		for q in file1:
			q = q.strip()
			quantifiers.append(q)
		
		# DONE --- TODO I have the list including möglich, bestimmt, sicher: Es ist unmöglich abzulehnen.
		file3 = open("/home/masha/02targets_wlv/code/extract/get_features/searchlists/de_modal-adj_predicates.lst", 'r').readlines()
		adj_pred = []
		for adj in file3:
			adj = adj.strip()
			adj_pred.append(adj)

		file4 = open("/home/masha/02targets_wlv/code/extract/get_features/searchlists/de_converts.lst", 'r').readlines()
		converts = []
		for conv in file4:
			conv = conv.strip()
			converts.append(conv)
		
	if lang == 'ru':
		
		file0 = "/home/masha/02targets_wlv/code/extract/get_features/searchlists/ru_deverbals_stop.lst"
		stoplist = open(file0, 'r').readlines()
		pseudo_deverbs = []
		for wd in stoplist:
			wd = wd.strip()
			pseudo_deverbs.append(wd)

		file1 = open("/home/masha/02targets_wlv/code/extract/get_features/searchlists/ru_adv_quantifiers.lst", 'r').readlines()
		quantifiers = []
		for q in file1:
			q = q.strip()
			quantifiers.append(q)

		file2 = open("/home/masha/02targets_wlv/code/extract/get_features/searchlists/ru_modal-adj_predicates.lst", 'r').readlines()
		adj_pred = []
		for adj in file2:
			adj = adj.strip()
			adj_pred.append(adj)
		converts = []

		
		
	return quantifiers, adj_pred, pseudo_deverbs, converts

def dms_support_all_langs(lang):
	
	add = "/home/masha/02targets_wlv/code/extract/get_features/searchlists/dms/" + lang + "_additive.lst"
	add_list = [i.strip() for i in open(add, 'r').readlines()]

	adv = "/home/masha/02targets_wlv/code/extract/get_features/searchlists/dms/" + lang + "_adversative.lst"
	adv_list = [i.strip() for i in open(adv, 'r').readlines()]

	caus = "/home/masha/02targets_wlv/code/extract/get_features/searchlists/dms/" + lang + "_causal.lst"
	caus_list = [i.strip() for i in open(caus, 'r').readlines()]

	temp_sequen = "/home/masha/02targets_wlv/code/extract/get_features/searchlists/dms/" + lang + "_temp_sequen.lst"
	temp_sequen_list = [i.strip() for i in open(temp_sequen, 'r').readlines()]

	epistem = "/home/masha/02targets_wlv/code/extract/get_features/searchlists/dms/" + lang + "_epistemic.lst"
	epist_list = [i.strip() for i in open(epistem, 'r').readlines()]

	
	return add_list, adv_list, caus_list, temp_sequen_list, epist_list