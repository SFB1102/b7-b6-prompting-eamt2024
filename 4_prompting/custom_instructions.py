"""
9 March 2024
a massive update to reflect reconceptualised types of prompts: we have two approaches by two modes now

prints stats on the most_first and most_second instructions

17 Nov 2023
re-working after building a new resource that has a truly raw version
this splits the process of prompt generation into 2 subprocesses:
(1) retrieve individual stats from earlier table and thres on all feats (including targeted De14, En12)
(2) for each of the 10 (sic!) possible setup from the combination of 3 args.mode options and 3 args.content option + lazy
outputs a table with formated prompts to be fed to the model (after chunking)

5 October 2023
The script calculates the size of input for GPT3.5-turbo and GPT-4 to estimate costs
but the real expenses incurred also include the output which we cannot predict.

In the end, in March 2024 we ended up paying around 400 euros for all experiments since October 2023, including GPT3.5 and GPT-4


python3 4_prompting/custom_instructions.py --tables 4_prompting/input/ --level seg --mode min --vratio 2.5 --approach feature-based --thresholds 3_feats_analysis/res/ --lang en

"""
import ast
import os
import re
import sys
import time
from collections import defaultdict, Counter, OrderedDict
from datetime import datetime
from operator import itemgetter

import numpy as np
from tqdm import tqdm
import pandas as pd
import stanza
from stanza.pipeline.core import DownloadMethod
import argparse

from utils import get_features_thresholds, get_tree
from utils import ud_freqs, speakdiff, count_dms, sents_complexity, finites, pasttense, content_ttr_and_density
from utils import nouns_to_all_args, readerdiff, mean_sent_length, sconj, possdet, advmod_no_negation


def generate_context(src=None, tgt=None, instructions=None, approach=None, slang=None, tlang=None, mode=None):
    context = None
    if instructions:
        if approach == 'feature-based':  # approach feature_based
            if mode == 'detailed':
                context = f'Your task is to reduce translationese in a human translation by re-writing it in a more natural, less translated way.' \
                          f'Translationese refers to any properties of translations that make them statistically distinct from texts originally produced in the target language. ' \
                          f'\nHere is an original {slang} text: ```{src}``` \nThis is its human translation into {tlang}: ```{tgt}``` ' \
                          f'\nRevise this translation following the instructions which reflect deviations of this segment from the expected target language norm: ' \
                          f'\n{instructions} ' \
                          f'\nDo not add any meta-phrases or quotation marks. Do not copy the original text.'
            else:  # if mode == 'min':
                context = f'Your task is to re-write a human translation in a more natural way.' \
                          f'\nHere is an original {slang} text: ```{src}``` \nThis is its human translation into {tlang}: ```{tgt}``` ' \
                          f'\nRevise this translation following the instructions: \n{instructions} ' \
                          f'\nDo not add any meta-phrases or quotation marks. Do not copy the original text.'
            # elif components == 'srclos':
            #     context = f'This is a human {tlang} translation of a {slang} text: ```{tgt}```  Revise this translation following the instructions: {instructions}. Do not add meta-phrases or any quotation marks. Do not copy the original text.'
            # elif components == 'tgtlos':
            #     context = f'This is an original {slang} text: ```{src}``` Translate this text into {tlang} following the instructions: {instructions}. Do not add meta-phrases or any quotation marks. Do not copy the original text.'
            # else:  # lazy, expert, but this should not happen - I pass None for these args.content=>components
            #     context = f'This is an original {slang} text: ```{src}``` This is its human {tlang} translation: ```{tgt}```  Revise this translation to make it sound more like a text originally produced in the target language instead of being a translation. Do not add meta-phrases or any quotation marks. Do not copy the original text.'
    elif approach == 'translated':
        context = f'Your task is to re-translate a human translation to make it more natural in the target language if necessary.' \
                          f'\nHere is an original {slang} text: ```{src}```\nThis is its human translation into {tlang}: ```{tgt}```  ' \
                          f'\nIf this translation can be re-translated to sound more like a text originally produced in the target language, return a re-translated version. If this translation sounds natural enough, return the input translation.' \
                          f'\nDo not add any meta-phrases or quotation marks. Do not copy the original text.'
    else:  #
        if approach == "self-guided":
            if mode == 'detailed':
                context = f'Your task is to reduce translationese in a human translation by re-writing it in a more natural way where possible. ' \
                          f'Translationese refers to any regular linguistic features in the translated texts that make them distinct from texts originally produced in the target language, outside the communicative situation of translation. These features are typically detected by statistical analysis and are explained by the specificity of the translation process. ' \
                          f'Human translators are known to simplify the source language content and to make it more explicit. ' \
                          f'Translations can exhibit a tendency to conform to patterns which are typical of the target language, ' \
                          f'making the output less varied than in comparable non-translations in the target language. ' \
                          f'The more obvious sign of translationese is interference, which can be defined as over-reliance on the intersection of patterns found in source and target languages. ' \
                          f'Translationese is manifested in the inflated frequencies of specific linguistic items such as function words ' \
                          f'(especially connectives and pronouns), unusual frequencies of some parts of speech (especially nouns and adverbs) or grammatical forms (especially forms of verbs), in reduced lexical variety and unexpected lexical sequences, ' \
                          f'in less natural word order, in longer and more complex sentences as well as lack of target language specific items and structures.' \
                          f'\nHere is an original {slang} text: ```{src}```\nThis is its human translation into {tlang}: ```{tgt}```' \
                          f'\nIf you can detect any translationese deviations in this translation, revise this translation to make it sound less translated and return the revised version. If no translationese is detected, return the input translation.' \
                          f'\nDo not add any meta-phrases or quotation marks. Do not copy the original text.'
            # Comment your variant. If you cannot not detect any translationese, copy the existing translation as is.
            else:  # mode == 'min':
                context = f'Your task is to re-write a human translation in a more natural way if necessary.' \
                          f'\nHere is an original {slang} text: ```{src}```\nThis is its human translation into {tlang}: ```{tgt}```  ' \
                          f'\nIf this translation can be revised to sound more like a text originally produced in the target language, return a revised version. If this translation sounds natural enough, return the input translation.' \
                          f'\nDo not add any meta-phrases or quotation marks. Do not copy the original text.'
        else:
            # if the passed list of instruction is empty for some reason, fall back to expert mode, but this should not happen
            context = f'Your task is to reduce translationese in a human translation by re-writing it in a more natural way where possible. ' \
                      f'Translationese refers to any regular linguistic features in the translated texts that make them distinct from texts originally produced in the target language, outside the communicative situation of translation. These features are typically detected by statistical analysis and are explained by the specificity of the translation process. ' \
                      f'Human translators are known to simplify the source language content and to make it more explicit. ' \
                      f'Translations can exhibit a tendency to conform to patterns which are typical of the target language, ' \
                      f'making the output less varied than in comparable non-translations in the target language. ' \
                      f'The more obvious sign of translationese is interference, which can be defined as over-reliance on the intersection of patterns found in source and target languages. ' \
                      f'Translationese is manifested in the inflated frequencies of specific linguistic items such as function words ' \
                      f'(especially connectives and pronouns), unusual frequencies of some parts of speech (especially nouns and adverbs) or grammatical forms (especially forms of verbs), in reduced lexical variety and unexpected lexical sequences, ' \
                      f'in less natural word order, in longer and more complex sentences as well as lack of target language specific items and structures.' \
                      f'\nHere is an original {slang} text: ```{src}```\nThis is its human translation into {tlang}: ```{tgt}```' \
                      f'\nIf you can detect any translationese deviations in this translation, revise this translation to make it sound less translated and return the revised version. If no translationese is detected, return the input translation.' \
                      f'\nDo not add any meta-phrases or quotation marks. Do not copy the original text.'
            # Do not add any comments or any quotation marks. Do not copy the original text.
            print('This should not happen!')
            exit()
    if not context:
        print('Context is not built')
        exit()

    prompt_wc = len(context.split())

    return prompt_wc, context


def split_list_by_last_index(lst, last_indices):
    sublists = []
    start_index = 0

    for last_index in last_indices:
        sublists.append(lst[start_index:last_index + 1])
        start_index = last_index + 1

    # Add the remaining elements if any
    if start_index < len(lst):
        sublists.append(lst[start_index:])

    return sublists


def ratio_based_tuples_generator(tuples_lst_in=None, my_thres=None):
    """
    Args:
        ratio_thres:
        tuples_lst_in: all valid sanity tuples: [(0, ('advmod', 1.0, 2.885, 'add')), (1, ('ttr', 1.0, 0.962, 'remove')),
         that contain
        (index_of_worded_instruction_lst, (feat_name, observed_val, expected_val, thres_type))

    Returns: the input list of tuples (1) filtered based on the > 3 ratio
    between the anticipated bigger and anticipated smaller values, if observed value is over 0
    and (2) sorted by the size of the ratio
    """
    lst_out = []
    if my_thres == 'std2':
        ratio_thres_num = int(my_thres[-1])

        for itm in tuples_lst_in:

            if itm[1][1] > 0:
                if itm[1][3] == 'remove':
                    # observed - expected
                    big_diff = itm[1][1] - itm[1][2]
                    # print(big_diff)
                    # observed - expected = + 0.75 else pass
                    if abs(big_diff) > (itm[1][4] * ratio_thres_num):
                        if big_diff > 0:  # more than one std
                            lst_out.append((itm, big_diff))
                else:
                    # print('Am I here?')
                    big_diff = itm[1][1] - itm[1][2]
                    if abs(big_diff) > (itm[1][4] * ratio_thres_num) and big_diff < 0:  # more than one std
                        lst_out.append((itm, big_diff))
            else:
                print('I should not get here, all tuples with observed_zero are filtered out already')
                exit()
    else:  # this is the old routine for ratio2.5
        # print(f'\n*** Using {ratio_thres} ratio, not STD as cut-off ***\n')
        ratio_thres_num = float(my_thres.replace('ratio', ''))

        for itm in tuples_lst_in:
            # (11, ('nn', 0.1392405063291139, 0.199, 'add', 0.0783081982107748))
            if itm[1][1] > 0 and itm[1][2] > 0:
                if itm[1][3] == 'remove':
                    # observed / expected
                    if itm[1][1] / itm[1][2] >= ratio_thres_num:
                        big_diff = itm[1][1] / itm[1][2]
                        lst_out.append((itm, big_diff))
                else:
                    # print('Am I here?')
                    # print(itm[1][2] / itm[1][1], ratio_thres)
                    if itm[1][2] / itm[1][1] >= ratio_thres_num:
                        big_diff = itm[1][2] / itm[1][1]
                        lst_out.append((itm, big_diff))
            else:
                print(itm)
                print('I should not get here, all tuples with observed_zero are filtered out already')
                exit()
    # print(lst_out)
    # input()

    lst_sorted = sorted(lst_out, key=lambda x: x[1], reverse=False)
    lst_sorted = [i[0] for i in lst_sorted]
    # print(lst_sorted)

    return lst_sorted


def filter_n_sort(my_instr=None, my_tuples=None, my_ratio=None):
    # [('advmod', 1.0, 2.885, 'add', 0.45567), ('ttr', 1.0, 0.962, 'remove', 0.45567)] -> # [(0, ('advmod', 1.0, 2.885, 'add')), (1, ('ttr', 1.0, 0.962, 'remove')),
    tuple_list_with_indices = list(enumerate(my_tuples))
    # print(tuple_list_with_indices)
    filtered_lst = []
    selected_tuple_list_with_indices = ratio_based_tuples_generator(tuples_lst_in=tuple_list_with_indices,
                                                                    my_thres=my_ratio)

    sorted_indices = [ind for ind, _ in selected_tuple_list_with_indices]
    # print(sorted_indices)
    apply_these = [my_instr[i] for i in sorted_indices]

    for i in selected_tuple_list_with_indices:
        filtered_lst.append(i[1][0])

    this_mode_worded_inst_list = apply_these
    # print(apply_these)
    # lose indices used to filter the parallel list of worded instructions
    this_mode_tuples = [i[1] for i in selected_tuple_list_with_indices]

    return filtered_lst, this_mode_worded_inst_list, this_mode_tuples


def generate_instuction_string(instructions=None):
    if instructions:
        if len(instructions) == 0:
            string_of_instructions = 'copy over the translation intact'
            print('Do I ever get here?')
            exit()
        else:
            # convert the list of instructions into a string
            string_of_instructions = '\n'.join(instructions)
    else:
        print('--- the diff between target and norm is not large enough')
        string_of_instructions = 'copy over the translation intact'
    # print(string_of_instructions)

    return string_of_instructions


class Logger(object):
    def __init__(self, logfile=None):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")  # overwrite, don't "a" append

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def rewriting_oracle(feat_name=None, bins=None, observed_val=None, expected_val=None, thres_type=None, org_std=None):
    res_tuple = (feat_name, observed_val, expected_val, thres_type, org_std)
    if observed_val == 0 or expected_val == 0:
        return None, res_tuple
    else:
        if thres_type == 'add':
            if observed_val <= expected_val:
                return bins[0], res_tuple
            else:
                return None, res_tuple
        elif thres_type == 'remove':
            if observed_val > expected_val:
                # constructions -> phrases
                try:
                    return bins[1], res_tuple
                except IndexError:
                    print(feat_name)
                    exit()
            else:
                return None, res_tuple
        else:
            print(f'{feat_name} ERROR!')
            exit()


def typical_diviations(thres_lst=None, order=None):
    filtered_thres_llst = [lst for lst in thres_lst if lst]
    ol_first = []
    for lst in filtered_thres_llst:
        try:
            first = lst[order][0]
            ol_first.append(first)
        except IndexError:  # not all segs have two feats!
            continue
            # print(lst)

    fdict = Counter(ol_first)

    dic_sort = OrderedDict(sorted(fdict.items(), key=itemgetter(1), reverse=True))
    tuples = list(dic_sort.items())
    my_strings = []
    for tu in tuples:
        string = f'{tu[0]}: {tu[1] / len(filtered_thres_llst) * 100:.1f};'

        my_strings.append(string)

    stringed_dict = " ".join(my_strings)
    tot_rewritten_segs = len(filtered_thres_llst)

    return stringed_dict, tot_rewritten_segs


# a dict with feat_names as keys and a list of two instructions for add and remove
my_map = {'advers': 'adversative', 'addit': 'additive', 'caus': 'causative', 'tempseq': 'temporal-sequencial'}
# 9 Mar: features with stat signif diffs between src and HT DE:44, EN:45
# excluded flat, 'inf', fixed, discourse, nummod

feat_dict = {'de': {'seg': ['acl', 'addit', 'advcl', 'advmod', 'advmod_verb', 'advqua', 'amod', 'appos', 'aux',
                            'aux:pass', 'caus', 'ccomp', 'cconj', 'conj', 'cop', 'demdets', 'dens', 'deverb', 'epist',
                            'fin', 'iobj', 'mdd', 'mhd', 'mpred', 'negs', 'nmod', 'nn', 'nnargs', 'nsubj',
                            'obj', 'obl', 'parataxis', 'pastv', 'poss', 'ppron', 'prep', 'self', 'simple', 'tempseq',
                            'ttr', 'vs_noun', 'wdlen']},
             'en': {'seg': ['acl', 'addit', 'advers', 'advmod', 'advmod_verb', 'advqua', 'amod', 'appos', 'aux',
                            'aux:pass',
                            'caus', 'ccomp', 'cconj', 'compound', 'cop', 'dens', 'deverb', 'epist', 'fin',
                            'iobj', 'mark', 'mdd', 'mean_sent_wc', 'mpred', 'negs', 'nmod', 'nn', 'nnargs', 'nsubj',
                            'obl', 'parataxis',
                            'pastv', 'ppron', 'prep', 'sconj', 'self', 'tempseq', 'vorfeld', 'wdlen', 'xcomp']}
             }

# detailed description for the top 15 segment level predictors
# examples are given in the descending order of frequency in translations
# [add instruction, remove instruction]

detailed_seg_instructions = {
    'de': {
        'acl': [
            'Use attributive clauses if necessary. Attributive clauses are finite and non-finite clauses that modify a nominal, '
            'including relative clauses that are introduced with a relative pronoun that refers back to the nominal. Such relative pronouns can include daß, der, auf dem, die, etc.',
            'Remove Use attributive clauses if necessary. Attributive clauses are finite and non-finite clauses that modify a nominal, '
            'including relative clauses that are introduced with a relative pronoun that refers back to the nominal. Such relative pronouns can include daß, der, auf dem, die, etc.'],
        'addit': [
            'Add connectives that introduce additive information. Additive connectives include, but not are limited to, items like: '
            'auch, dafür, sowie, nicht nur, außerdem, in Bezug auf, weiterhin, sowohl, ebenfalls, nämlich.'
            'Remove connectives that introduce additive information. Additive connectives include, but not are limited to, items like: '
            'auch, dafür, sowie, nicht nur, außerdem, in Bezug auf, weiterhin, sowohl, ebenfalls, nämlich.'],
        'advcl': [
            'Prefer adverbial clauses. An adverbial clause is a clause which modifies a verb or other predicate (adjective, etc.). '
            'This includes things such as a temporal clause, consequence, conditional clause, purpose clause, etc.',
            'Avoid adverbial clauses. An adverbial clause is a clause which modifies a verb or other predicate (adjective, etc.). '
            'This includes things such as a temporal clause, consequence, conditional clause, purpose clause, etc.'],
        'advmod': [
            'Use more adverbial modifiers. An adverbial modifier of a word is a (non-clausal) adverb or adverbial phrase '
            'that serves to modify a predicate or a modifier word.',
            'Avoid adverbial modifiers. An adverbial modifier of a word is a (non-clausal) adverb or adverbial phrase '
            'that serves to modify a predicate or a modifier word.'],
        'advmod_verb': [
            'If necessary, use adverbial modifiers in the beginning of the sentence, before the predicate. An adverbial modifier of a word is a (non-clausal) adverb or adverbial phrase '
            'that serves to modify a predicate or a modifier word.',
            'Avoid adverbial modifiers in the beginning of the sentence. An adverbial modifier of a word is a (non-clausal) adverb or adverbial phrase '
            'that serves to modify a predicate or a modifier word.'],
        'advqua': [
            'If necessary, use adverbial quantifiers. An adverbial quantifier is a word that is used to characterise the action named by the predicate. '
            'Adverbial quantifiers include words like absolut, einzig, furchtbar, gerade, intensiv, ungewöhnlich, größtenteils.',
            'Avoid adverbial quantifiers. An adverbial quantifier is a word that is used to characterise the action named by the predicate. '
            'Adverbial quantifiers include words like absolut, einzig, furchtbar, gerade, intensiv, ungewöhnlich, größtenteils.'],
        'appos': [
            'If necessary, use appositional modifier. An appositional modifier is a nominal immediately following '
            'the first noun that serves to define, modify, name, or describe that noun. For example, Der Begriff "Wende" steht für..., Die Senatorin, Traute Müller,...',
            'Avoid appositional modifier. An appositional modifier is a nominal immediately following '
            'the first noun that serves to define, modify, name, or describe that noun. For example, Der Begriff "Wende" steht für..., Die Senatorin, Traute Müller,...'],
        'amod': [
            'Use adjectival modifier. An adjectival modifier of a noun (or pronoun) is any adjectival phrase that serves to modify the noun (or pronoun). For example, rohes Fleisch.',
            'Avoid adjectival modifier. An adjectival modifier of a noun (or pronoun) is any adjectival phrase that serves to modify the noun (or pronoun). For example, rohes Fleisch.'],
        'aux': [
            'Increase the frequency of auxiliary verbs. An auxiliary of a clause is a function word associated with a verbal predicate that expresses categories such as tense, mood, aspect, voice or evidentiality.',
            'Reduce the frequency of auxiliary verbs. An auxiliary of a clause is a function word associated with a verbal predicate that expresses categories such as tense, mood, aspect, voice or evidentiality.'],
        'caus': [
            'Make causative relations between parts of the sentence more explicit. This can be done by using connectives like: '
            'damit, da, weil, denn, deshalb, darum, bedeutet, auf diese Weise, als Folge, für diesen Zweck.',
            'Avoid causative connectives. Causative connectives include, but are not limited to, the following: '
            'damit, da, weil, denn, deshalb, darum, bedeutet, auf diese Weise, als Folge, für diesen Zweck.'],
        'fin': [
            'Use more verbs in their finite forms instead of infinitives and participles. Finite verb forms function as main predicates, '
            'they usually agree with subjects in person and have different forms for various tenses.',
            'Use non-finite constructions instead of full clauses. Finite verb forms function as main predicates, '
            'they usually agree with subjects in person and have different forms for various tenses.'],
        'iobj': [
            'Prefer constructions with indirect objects. An indirect object of a verb is any nominal phrase that is a obligatory argument of the verb '
            'but is not its subject or direct object. The prototypical example is the recipient (dem Kind) with verbs of exchange: Die Frau gibt dem Kind einen Apfel.',
            'Avoid constructions with indirect objects. An indirect object of a verb is any nominal phrase that is a obligatory argument of the verb '
            'but is not its subject or direct object. The prototypical example is the recipient (dem Kind) with verbs of exchange: Die Frau gibt dem Kind einen Apfel.'],
        'mhd': ['Produce sentences with the more complex hierarchy of syntactic relations.',
                'Avoid complex syntactic structures with the deep hierarchy of dependency relations between the elements.'],
        'mean_sent_wc': ['Make the sentences a bit longer.', 'Make sentences shorter.'],
        'mdd': ['Allow more intervening tokens between words and their syntactic heads.',
                'Keep words closer to their syntactic heads.'],
        'nmod': [
            'Use nominal modifiers of nouns such as prepositional and genitive case phrases.',
            'Avoid nominal modifiers of nouns such as prepositional and genitive case phrases.'],
        'nnargs': [
            'Make sure that verbs only have nouns or proper names as dependents.',
            'Use pronouns instead of nouns or proper names as verbal arguments where possible.'],
        'parataxis': [
            'Avoid explicit connectives. Instead rely on simple sequence of parts of the sentence or sequence of sentences.',
            'Use explicit markers of discourse relations to indicate relations between parts of sentences or separate sentences.'],
        'pastv': [
            'Prefer Präteritum to other ways of indicating the past tense.',
            'Avoid Präteritum in favour of other ways of indicating the past tense.'],
        'poss': [
            'Where possible explicitly mark possessive relations using such pronouns as mein, dein, sein, ihr, Ihr, unser, eurer.',
            'Avoid possessive pronouns such as mein, dein, sein, ihr, Ihr, unser, eurer.'],
        'ttr': [
            'Use more varied vocabulary to make sure that in your version the ratio of unique content words to all content words is higher than in the input translation.',
            # remove, use fewer types
            'Make sure that in your version the ratio of unique content words to all content words is lower than in the input translation.'],
        'ccomp': [
            'Increase the frequency of clausal complements. A clausal complement of a verb or adjective is a dependent clause which is an obligatory argument. '
            'That is, it functions like an object of the verb, or adjective.',
            'Avoid clausal complements. A clausal complement of a verb or adjective is a dependent clause which is an obligatory argument. '
            'That is, it functions like an object of the verb, or adjective.'
        ],
        'simple': [
            'Prefer simple sentences. A simple sentence is a sentence which consists of one clause and has one predicate.',
            'Avoid simple sentences. A simple sentence is a sentence which consists of one clause and has one predicate.'
        ],
        'self': ['Use reflexive pronouns where appropriate.',
                 'Do not overuse reflexive pronouns.'],
        'aux:pass': ['Use passive voice where appropriate.',
                     'Avoid passive constructions.'],
        'dens': ['Make your vocabulary more varied.',
                 'Repeat words if it is necessary for better cohesion.'],
        'mpred': [
            'Prefer modal predicates, i.e. clauses with such verbs as dürfen, können, mögen, müssen, sollen, wollen.',
            'Avoid modal predicates, i.e. clauses with such verbs as dürfen, können, mögen, müssen, sollen, wollen.'],
        'wdlen': [
            'Use longer words.',
            'Prefer shorter words.'],
        'vorfeld': [
            'Where possible, prefer the word order where the sentence starts with a topicalised element, which is not the subject of the sentence.',
            'Avoid sentences starting with non-subject members of the sentence.'],
        'cop': ['Use constructions with copula verbs where appropriate. '
                'A copula verb is a function word used to link a subject to a nonverbal predicate, which in German would be the forms of the verb "sein").',
                'Avoid constructions with copula verbs. '
                'A copula verb is a function word used to link a subject to a nonverbal predicate, which in German would be the forms of the verb "sein").'],
        'mark': ['Use subordinate clauses if necessary.',
                 'Avoid sentences with several subordinate clauses.'],
        'nn': ['Prefer constructions with nouns to express ideas.',
               'Reduce the number of nouns per sentence.'],
        'nsubj': ['Increase the number of subject relations per sentence. '
                  'A subject is a nominal which is the syntactic subject and the proto-agent of a clause.',
                  'Reduce the number of subject relations per sentence. '
                  'A subject is a nominal which is the syntactic subject and the proto-agent of a clause.'],
        'cconj': [
            'Increase the number of coordinating conjunctions per sentence. Coordinating conjunctions include function words like und, oder, aber, sondern, sowie, als, wie, doch.',
            'Reduce the number of coordinating conjunctions per sentence. Coordinating conjunctions include function words like und, oder, aber, sondern, sowie, als, wie, doch.'],
        'epist': ['Where appropriate use epistemic constructions. Epistemic constructions are introductory phrases, '
                  'which reflect the attitude of the speaker to the truth-value of the statement. They include: wahrscheinlich, Ich bein sicher, es ist klar, offensichtlich, bestimmt.',
                  'Avoid epistemic constructions. Epistemic constructions are introductory phrases, '
                  'which reflect the attitude of the speaker to the truth-value of the statement. They include: wahrscheinlich, Ich bein sicher, es ist klar, offensichtlich, bestimmt.'],
        'prep': ['Where necessary use prepositional phrases.',
                 'Avoid prepositional phrases.'],
        'deverb': ['Use nouns derived from verbs where possible.',
                   'Avoid nouns derived from verbs, prefer verbs or other ways of expression instead.'],
        'tempseq': [
            'Use temporal and sequential discourse markers (such as anschließend, bevor, bis dann, damals, Dritte, endlich, Weiter folgt) to increase text cohesion.',
            'Avoid temporal and sequential discourse markers (such as anschließend, bevor, bis dann, damals, Dritte, endlich, Weiter folgt).'],
        'demdets': [
            'Use demonstrative pronouns where necessary. Demonstrative pronouns are words like dies, alle, jed, einige, solch, viel, irgendwelch, dieselbe. They increase text cohesion.',
            'Avoid demonstrative pronouns where necessary. Demonstrative pronouns are words like dies, alle, jed, einige, solch, viel, irgendwelch, dieselbe.'],
        'conj': [
            'Use coordinated members of the sentence where possible.',
            'Avoid coordinated members of the sentence.'],
        'negs': [
            'Use more sentences with negation.',
            'Rephrase without using negative particles.'],
        'obj': ['Prefer structures with direct objects.',
                'Avoid structures with direct objects.'],
        'obl': [
            'Where possible use nominals (noun, pronoun, noun phrase) functioning as optional arguments dependent on verb, adjective or another adjective. They are usually introduced by a preposition.',
            'Avoid prepositional nominals in adverbial function. They function as optional arguments dependent on verb, adjective or another adjective.'],
        'ppron': [
            'Use more of personal pronouns such as ich, ihr, du, er, sie, es, wir, mich, mir, dich, dir, ihm.',
            'Avoid personal pronouns such as ich, ihr, du, er, sie, es, wir, mich, mir, dich, dir, ihm.'],
        'vs_noun': [
            'Allow inversion in main clause in affirmative sentences. Inversion is a form of sentence structure, which reverses the traditional order of subject/verb.',
            'Avoid inversion in main clause in affirmative sentences if possible. Inversion is a form of sentence structure, which reverses the traditional order of subject/verb..']
    },
    'en': {
        'acl': [
            'Use attributive clauses if necessary. Attributive clauses are finite and non-finite clauses that modify a nominal, '
            'including relative clauses that are introduced with a relative pronoun that refers back to the nominal. Such relative pronouns include who, that, which.'
            'Remove Use attributive clauses if necessary. Attributive clauses are finite and non-finite clauses that modify a nominal, '
            'including relative clauses that are introduced with a relative pronoun that refers back to the nominal. Such relative pronouns include who, that, which.'],
        'addit': [
            'Add connectives that introduce additive information. They include, but not limited to, items like: '
            'also, for example, not only, in particular, such as, in this regard, too, with regard to, namely, in other words, in addition.',
            'Remove connectives that introduce additive information. They include, but not limited to, items like: '
            'also, for example, not only, in particular, such as, in this regard, too, with regard to, namely, in other words, in addition.'],
        'advcl': [
            'Prefer adverbial clauses. An adverbial clause is a clause which modifies a verb or other predicate (adjective, etc.). '
            'This includes things such as a temporal clause, consequence, conditional clause, purpose clause, etc.',
            'Avoid adverbial clauses. An adverbial clause is a clause which modifies a verb or other predicate (adjective, etc.). '
            'This includes things such as a temporal clause, consequence, conditional clause, purpose clause, etc.'],
        'advmod': [
            'Use more adverbial modifiers. An adverbial modifier of a word is a (non-clausal) adverb or adverbial phrase '
            'that serves to modify a predicate or a modifier word.',
            'Avoid adverbial modifiers. An adverbial modifier of a word is a (non-clausal) adverb or adverbial phrase '
            'that serves to modify a predicate or a modifier word.'],
        'advmod_verb': [
            'If necessary, use adverbial modifiers in the beginning of the sentence, before the predicate. An adverbial modifier of a word is a (non-clausal) adverb or adverbial phrase '
            'that serves to modify a predicate or a modifier word.',
            'Avoid adverbial modifiers in the beginning of the sentence. An adverbial modifier of a word is a (non-clausal) adverb or adverbial phrase '
            'that serves to modify a predicate or a modifier word.'],
        'advqua': [
            'If necessary, use adverbial quantifiers. An adverbial quantifier is a word that is used to characterise the action named by the predicate. '
            'Adverbial quantifiers include words like absolutely, completely, highly, mildly, partially, scarcely, thoroughly, rather, perfectly.',
            'Avoid adverbial quantifiers. An adverbial quantifier is a word that is used to characterise the action named by the predicate. '
            'Adverbial quantifiers include words like absolutely, completely, highly, mildly, partially, scarcely, thoroughly, rather, perfectly.'],
        'appos': [
            'If necessary, use appositional modifier. An appositional modifier is a nominal immediately following '
            'the first noun that serves to define, modify, name, or describe that noun. For example, Sam, my brother, ..., The Australian Broadcasting Corporation (ABC)...',
            'Avoid appositional modifier. An appositional modifier is a nominal immediately following '
            'the first noun that serves to define, modify, name, or describe that noun. For example, Sam, my brother, ..., The Australian Broadcasting Corporation (ABC)...'],
        'amod': [
            'Use adjectival modifier. An adjectival modifier of a noun (or pronoun) is any adjectival phrase that serves to modify the noun (or pronoun). For example, large house.',
            'Avoid adjectival modifier. An adjectival modifier of a noun (or pronoun) is any adjectival phrase that serves to modify the noun (or pronoun). For example, large house.'],
        'aux': [
            'Increase the frequency of auxiliary verbs. An auxiliary of a clause is a function word associated with a verbal predicate that expresses categories such as tense, mood, aspect, voice or evidentiality.',
            'Reduce the frequency of auxiliary verbs. An auxiliary of a clause is a function word associated with a verbal predicate that expresses categories such as tense, mood, aspect, voice or evidentiality.'],
        'caus': [
            'Make causative relations between parts of the sentence more explicit. This can be done by using connectives like: '
            'because, therefore, so that, for this reason, as a result, after all, for that reason, hence, consequently, to this end.',
            'Avoid causative connectives. Causative connectives include, but are not limited to, the following: '
            'because, therefore, so that, for this reason, as a result, after all, for that reason, hence, consequently, to this end.'],
        'fin': [
            'Use verbs in their finite forms instead of infinitives and participles. Finite verb forms function as main predicates, '
            'they usually agree with subjects in person and have different forms for various tenses.',
            'Use non-finite constructions instead of full clauses. Finite verb forms function as main predicates, '
            'they usually agree with subjects in person and have different forms for various tenses.'],
        'iobj': [
            'Prefer constructions with indirect objects. An indirect object of a verb is any nominal phrase that is a obligatory argument of the verb '
            'but is not its subject or direct object. The prototypical example is the recipient of ditransitive verbs of exchange: She gave me a raise.',
            'Avoid constructions with indirect objects. An indirect object of a verb is any nominal phrase that is a obligatory argument of the verb '
            'but is not its subject or direct object. The prototypical example is the recipient of ditransitive verbs of exchange: She gave me a raise.'],
        'mhd': ['Produce sentences with the more complex hierarchy of syntactic relations.',
                'Avoid complex syntactic structures with the deep hierarchy of dependency relations between the elements.'],
        'mean_sent_wc': [
            'Make the sentences a bit longer.',
            'Make sentences shorter.'],
        'mdd': [
            'Allow more intervening tokens between words and their syntactic heads.',
            'Keep words closer to their syntactic heads.'],
        'nmod': [
            'Use nominal modifiers of nouns such as prepositional and genitive case phrases.',
            'Avoid nominal modifiers of nouns such as prepositional and genitive case phrases.'],
        'nnargs': [
            'Make sure that verbs only have nouns or proper names as dependents.',
            'Use pronouns instead of nouns or proper names as verbal arguments where possible.'],
        'parataxis': [
            'Avoid explicit connectives. Instead rely on simple sequence of parts of the sentence or sequence of sentences.',
            'Use explicit markers of discourse relations to indicate relations between parts of sentences or separate sentences.'],
        'pastv': [
            'Prefer Simple Past to other ways of indicating the past tense.',
            'Avoid Simple Past in favour of other ways of indicating the past tense.'],
        'poss': [
            'Where possible explicitly mark possessive relations using such pronouns as my, their, his, her.',
            'Avoid possessive pronouns such as my, their, his, her.'],
        'ttr': [
            'Use more varied vocabulary to make sure that in your version the ratio of unique content words to all content words is higher than in the input translation.',
            # remove, use fewer types
            'Make sure that in your version the ratio of unique content words to all content words is lower than in the input translation.'],
        # + uniq for en
        'advers': [
            'Use more of connectives that introduce adversative information such as '
            'however, still, actually, yet, instead, rather than, in fact, nevertheless, otherwise, on the contrary.',
            'Remove connectives that introduce adversative information. '
            'Adversative connectives inclide, but are not limited to, the following items: however, still, actually, yet, instead, rather than, in fact, nevertheless, otherwise, on the contrary\n'],
        'compound': [
            'Make use of compounds. Compounds are combinations of words that morphosyntactically behave as single words, '
            'usually nouns modifying other nouns that can be written as separate words, such as ice cream flavours or apple juice.',
            'Avoid compounds. Compounds are combinations of words that morphosyntactically behave as single words, '
            'usually nouns modifying other nouns that can be written as separate words, such as ice cream flavours or apple juice.'],
        'conj': [
            'Use coordinated members of the sentence where possible.',
            'Avoid coordinated members of the sentence.'],
        'numcls': [
            'Use more dependent clauses per sentence.',
            'Use fewer clauses per sentence.'],
        'obl': [
            'Where possible use nominals (noun, pronoun, noun phrase) functioning as optional arguments dependent on verb, adjective or another adjective. They are usually introduced by a preposition.',
            'Avoid prepositional nominals in adverbial function. They function as optional arguments dependent on verb, adjective or another adjective.'],
        'ppron': [
            'Use more of personal pronouns such as i, you, he, she, it, we, they, me, him, her, us, them.',
            'Avoid personal pronouns such as i, you, he, she, it, we, they, me, him, her, us, them.'],
        'negs': [
            'Use more sentences with negation.',
            'Rephrase without using negative particles.'],
        # + uniq at doc level
        'demdets': [
            'Increase the frequency of demonstrative pronouns such as this, some, these, that, any, all, every, another, each, those.',
            'Reduce the number of demonstrative pronouns such as this, some, these, that, any, all, every, another, each, those.'
        ],
        'sconj': [
            'Use more subordinating conjunctions. Subordinating conjunctions are function words which express relations between two clauses, where one is dependent on the other. '
            'Typical subordinating conjunctions include if, as, while, because, despite, unless, whereas, although, though, since.',
            'Avoid overusing subordinating conjunctions. Subordinating conjunctions are function words which express relations between two clauses, where one is dependent on the other. '
            'Typical subordinating conjunctions include if, as, while, because, despite, unless, whereas, although, though, since.'
        ],
        'self': ['Use reflexive pronouns where appropriate.',
                 'Do not overuse reflexive pronouns.'],
        'aux:pass': ['Use passive voice where appropriate.',
                     'Avoid passive constructions.'],
        'dens': ['Make your vocabulary more varied.',
                 'Repeat words if it is necessary for better cohesion.'],
        'simple': ['Prefer simple sentences.',
                   'Avoid simple sentences'],
        'mpred': [
            'Prefer modal predicates, i.e. clauses with such verbs as must, can, have to, should, ought to, need.',
            'Avoid modal predicates, i.e. clauses with such verbs as must, can, have to, should, ought to, need.'],
        'wdlen': [
            'Use longer words.',
            'Prefer shorter words.'],
        'vorfeld': [
            'Where possible, prefer the word order where the sentence starts with a topicalised element, which is not the subject of the sentence.',
            'Avoid sentences starting with non-subject members of the sentence.'],
        'cop': [
            'Use constructions with copula verbs where appropriate. A copula verb is a function word used to link a subject to a nonverbal predicate, '
            'including the expression of identity predication (e.g. sentences like "Kim is the President").',
            'Avoid constructions with copula verbs. A copula verb is a function word used to link a subject to a nonverbal predicate, '
            'including the expression of identity predication (e.g. sentences like "Kim is the President").'],
        'mark': ['Use subordinate clauses if necessary.',
                 'Avoid sentences with several subordinate clauses.'],
        'nn': ['Prefer constructions with nouns to express ideas.',
               'Reduce the number of nouns per sentence.'],
        'nsubj': ['Increase the number of subject relations per sentence. '
                  'A subject is a nominal which is the syntactic subject and the proto-agent of a clause.',
                  'Reduce the number of subject relations per sentence. '
                  'A subject is a nominal which is the syntactic subject and the proto-agent of a clause.'],
        'cconj': [
            'Increase the number of coordinating conjunctions per sentence. Coordinating conjunctions include function words like and, but, or, both, yet, either.',
            'Reduce the number of coordinating conjunctions per sentence. Coordinating conjunctions include function words like and, but, or, both, yet, either.'],
        'epist': ['Where appropriate use epistemic constructions. Epistemic constructions are introductory phrases, '
                  'which reflect the attitude of the speaker to the truth-value of the statement. They include: probably, to my mind, it is (un)likely, I believe that, I am sure, it is obvious.',
                  'Avoid epistemic constructions. Epistemic constructions are introductory phrases, '
                  'which reflect the attitude of the speaker to the truth-value of the statement. They include: probably, to my mind, it is (un)likely, I believe that, I am sure, it is obvious.'],
        'prep': ['Where necessary use prepositional phrases.',
                 'Avoid prepositional phrases.'],
        'deverb': ['Use nouns derived from verbs where possible.',
                   'Avoid nouns derived from verbs, prefer verbs or other ways of expression instead.'],
        'tempseq': [
            'Use temporal and sequential discourse markers (such as at last, at the same time, basically, finally, concurrently, earlier, in a nutshell, previously, in short) to increase text cohesion.',
            'Avoid temporal and sequential discourse markers (such as at last, at the same time, basically, finally, concurrently, earlier, in a nutshell, previously, in short).'],
        'vs_noun': [
            'Allow inversion in main clause in affirmative sentences. Inversion is a form of sentence structure, which reverses the traditional order of subject/verb.',
            'Avoid inversion in main clause in affirmative sentences if possible. Inversion is a form of sentence structure, which reverses the traditional order of subject/verb..'],
        'ccomp': [
            'Increase the frequency of clausal complements. A clausal complement of a verb or adjective is a dependent clause which is an obligatory argument. '
            'That is, it functions like an object of the verb, or adjective.',
            'Avoid clausal complements. A clausal complement of a verb or adjective is a dependent clause which is an obligatory argument. '
            'That is, it functions like an object of the verb, or adjective.'],
        'xcomp': ['Prefer non-finite constructions.',
                  'Avoid non-finite constructions.']
    }
}

# concise
SCRIPTURES = {'fin': ['Use full clauses instead of non-finite constructions.',
                      'Use non-finite constructions instead of full clauses.'],
              'case': ['ERROR_case', 'ERROR_case'],  # # this is colinear with obl in EN
              'nmod': ['Use nominal modifiers of nouns such as prepositional and genitive case phrases.',
                       'Avoid nominal modifiers of nouns such as prepositional and genitive case phrases.'],
              'advmod': ['Use more adverbial modifiers.', 'Avoid adverbial modifiers.'],

              'addit': ['Add connectives that introduce additional information.',
                        'Avoid connectives that introduce additional information.'],
              'appos': ['Add appositional modifiers.',
                        'Remove appositional modifiers.'],
              'advqua': ['Add adverbial quantifiers.',
                         'Avoid adverbial quantifiers.'],
              'advmod_verb': ['Use adverbial modifiers in the beginning of the sentence.',
                              'Avoid adverbial modifiers in the beginning of the sentence.'],
              'amod': ['Use adjectival modifier.', 'Avoid adjectival modifier.'],
              'caus': ['Make causative-consecutive relations between parts of the sentence more explicit.',
                       'Avoid causative-consecutive connectives.'],
              'mhd': ['Produce sentences with the more complex hierarchy of syntactic relations.',
                      'Avoid complex syntactic structures with the deep hierarchy of dependency relations between the elements.'],
              'mdd': [
                  'Allow more intervening tokens between words and their syntactic heads.',
                  'Keep words closer to their syntactic heads.'],
              'mean_sent_wc': ['Make the sentences a bit longer.', 'Make sentences shorter.'],
              'iobj': ['Prefer constructions with indirect objects.', 'Avoid constructions with indirect objects.'],

              'acl': ['Allow more attributive clauses.', 'Avoid attributive clauses.'],
              'advcl': ['Use adverbial clauses.', 'Avoid adverbial clauses.'],
              'parataxis': ['Avoid any types of explicit connectives.', 'Use explicit markers of discourse relations.'],
              'pastv': ['Prefer Simple Past/Präteritum to other ways of expressing past.',
                        'Avoid Simple Past/Präteritum in favour of other ways of expressing past.'],
              'poss': ['Where possible explicitly mark possessive relations using pronouns.',
                       'Avoid possessive pronouns.'],
              'relcl': ['Add relative clauses.', 'Avoid relative clauses.'],
              'ttr': ['Rely on more varied vocabulary.', 'Rely on more frequent words.'],
              'ppron': [
                  'Use more personal pronouns.',
                  'Avoid personal pronouns.'],
              'negs': [
                  'Use more sentences with negation.',
                  'Rephrase without using negative particles.'],

              'advers': ['Add connectives that introduce contrastive relations between parts of the sentence.',
                         'Avoid connectives that signal contrastive relations between parts of the sentence.'],
              'compound': ['Make use of nouns as modifiers of other nouns.', 'Avoid sequences of two nouns.'],
              'conj': [
                  'Use coordinated members of the sentence where possible.'
                  'Avoid coordinated members of the sentence.'],
              'numcls': ['use more dependent clauses', 'use fewer clauses'],
              'obl': ['Where possible use prepositional phrases in adverbial function.',
                      'Avoid prepositional phrases in adverbial function.'],
              'xcomp': ['Prefer non-finite constructions.', 'Avoid non-finite constructions.'],
              'ccomp': ['Prefer structures with clausal objects.', 'Avoid structures with clausal objects.'],
              'obj': ['Prefer structures with direct objects.', 'Avoid structures with direct objects.'],
              'nnargs': ['Make sure that verbs only have nouns or proper names as dependents.',
                         'Use pronouns instead of nouns as verbal arguments where possible.'],
              'sconj': ['Make subordinating relations between clauses more explicit.',
                        'Make subordinating relations between clauses more implicit.'],
              'self': ['Use reflexive pronouns where appropriate.',
                       'Do not overuse reflexive pronouns.'],
              'aux:pass': ['Use passive voice where appropriate.',
                           'Avoid passive constructions.'],
              'dens': ['Make your vocabulary more varied.',
                       'Repeat words if it is necessary for better cohesion.'],
              'simple': ['Prefer simple sentences.',
                         'Avoid simple sentences'],
              'wdlen': [
                  'Use longer words.',
                  'Prefer shorter words.'],
              'mpred': [
                  'Prefer modal predicates.',
                  'Avoid modal predicates.'],
              'vorfeld': [
                  'Where possible, prefer the word order where the sentence starts with a topicalised element, which is not the subject of the sentence.',
                  'Avoid sentences starting with non-subject members of the sentence.'],
              'cop': ['Use constructions with copula verbs where appropriate. ',
                      'Avoid constructions with copula verbs.'],
              'mark': ['Use subordinate clauses if necessary.',
                       'Avoid sentences with several subordinate clauses.'],
              'nn': ['Prefer constructions with nouns to express ideas.',
                     'Reduce the number of nouns per sentence.'],
              'aux': [
                  'Increase the frequency of auxiliary verbs.',
                  'Reduce the frequency of auxiliary verbs.'],
              'nsubj': ['Increase the number of subject relations per sentence.',
                        'Reduce the number of subject relations per sentence.'],
              'cconj': ['Increase the number of coordinating conjunctions per sentence.',
                        'Reduce the number of coordinating conjunctions per sentence.'],
              'epist': ['Where appropriate use epistemic constructions.',
                        'Avoid epistemic constructions.'],
              'prep': ['Where necessary use prepositional phrases.',
                       'Avoid prepositional phrases.'],
              'deverb': ['Use nouns derived from verbs where possible.',
                         'Avoid nouns derived from verbs, prefer verbs or other ways of expression instead.'],
              'tempseq': [
                  'Use temporal and sequential discourse markers to increase text cohesion.',
                  'Avoid temporal and sequential discourse markers.'],
              'demdets': [
                  'Use demonstrative pronouns where necessary.',
                  'Avoid demonstrative pronouns where necessary.'],
              'vs_noun': [
                  'Allow inversion in main clause (in affirmative sentences).',
                  'Avoid demonstrative pronouns where necessary.']
              }

# We experimented with several re-writing modes to explore the effects of re-writing intensity (max, min, tiny),
# model hyperparameters (0, 0.3, 0.5, 0.7) and
# various types of information in the prompt (no source [srclos], no target[tgtlos], no detailed linguistic instruction [lazy]).


def make_dirs(my_new_dirs=None, sub=None, more4logging=None):
    for i in my_new_dirs:
        if 'logs' in i:
            i = i.replace('logs/', f'logs/pregenerate/')
            os.makedirs(i, exist_ok=True)
            script_name = sys.argv[0].split("/")[-1].split(".")[0]
            log_file = f'{i}{more4logging}_{script_name}.log'
            sys.stdout = Logger(logfile=log_file)
        else:
            os.makedirs(i, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tables', help="segs from top 100 documents by SVM probability with 60 feat vals",
                        default='4_prompting/input/')
    parser.add_argument('--level', choices=['seg'],
                        help='feed segments or entire documents', default='seg')
    parser.add_argument('--mode', choices=['min', 'detailed'],
                        help='select intensity of the instruction mode', required=True)
    parser.add_argument('--approach', choices=['self-guided', 'feature-based', 'translated'],
                        help='which component of the instruction to run with: all three, no sources, no human targets, no specific instruction',
                        required=True)
    # run python3 3_feats_analysis/univariate_analysis.py --best_selection --sample contrastive --level seg
    parser.add_argument('--thresholds', help="a results table from feature analysis",
                        default='3_feats_analysis/res/')
    parser.add_argument('--thres_type', choices=['ratio2.5', 'std2'], default='ratio2.5', required=True)
    parser.add_argument('--lang', required=True)
    parser.add_argument('--outdir', default='4_prompting/ol_prompts/')
    parser.add_argument('--statsto', default='4_prompting/stats/')
    parser.add_argument('--logsto', default='logs/')
    parser.add_argument('--verbosity', type=int, default=0)

    args = parser.parse_args()

    start = time.time()

    make_dirs(my_new_dirs=[args.outdir, args.statsto, args.logsto], sub=args.thres_type,
              more4logging=f'{args.lang}_{args.approach}_{args.mode}')

    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")

    best_selection_seg = list(set(feat_dict['de']['seg']).union(feat_dict['en']['seg']))

    # prepare threshold information
    thres_df = pd.read_csv(f'{args.thresholds}{args.level}_feat-stats_all_sample_contrastive.tsv', sep='\t')
    # 'advcl': (0.507, 'remove'), 'compound': (0.787, 'add')
    this_lang_thres = get_features_thresholds(df0=thres_df, lang=args.lang)
    # print(f'\n***\n{this_lang_thres}\n***\n')

    # print(len(thres_df[thres_df['lang'] == args.lang]['feature']), len(
    #     feat_dict[args.lang][args.level]))
    # assert len(thres_df[thres_df['lang'] == args.lang]['feature']) == len(
    #     feat_dict[args.lang][args.level]), 'Huston, we have problems!'

    # the my_input table has seg_id, src_raw, raw, most_translated_seg_index columns
    my_file = [f for f in os.listdir(f'{args.tables}') if f.startswith(f'{args.lang}_most_translated_aligned_seg_ol')][
        0]
    # this is a seg-level input file containing 100 "most_translated" documents
    # (top 100 by SVM probability minus confident errors)
    my_input = pd.read_csv(f'{args.tables}{my_file}', sep='\t', compression='gzip')

    # print(my_input.head(3))
    # # print(my_input.shape)
    # print(my_input.columns.tolist())

    meta = ['src_raw', 'src_seg_id', 'most_translated', 'doc_id', 'seg_num', 'raw', 'ttype', 'corpus', 'direction',
            'lang', 'raw_tok', 'wc_lemma', 'sents', 'org_typical']
    muted = ['relcl', 'case']
    ol_feats = [i for i in my_input.columns.tolist() if i not in meta and i not in muted]
    # print(len(ol_feats))

    info_on_intensity_dict = defaultdict()
    shorts = 0
    low_vratio = 0
    total_wc = []
    prompts_dict = defaultdict(list)

    # if args.level == 'doc':
    #     drop_meta = ['src_seg_id', 'most_translated', 'doc_id', 'seg_num', 'ttype', 'corpus',
    #                  'direction', 'lang', 'raw_tok', 'wc_lemma', 'sents', 'org_typical']
    #     aggregate_functions = {col: 'mean' for col in ol_feats}
    #     my_input = my_input.groupby('doc_id', as_index=False).aggregate({'src_raw': lambda x: " ".join(x.tolist()),
    #                                                                      'raw': lambda x: " ".join(x.tolist()),
    #                                                                      **aggregate_functions
    #                                                                      })
    #     my_input = my_input.rename(columns={'doc_id': 'src_seg_id'})
    #     print(my_input.shape)
    #     print(my_input.columns.tolist())
    global_lst = []
    # Serialize DataFrame row by row
    for index, row in tqdm(my_input.iterrows(), position=0, leave=True):
        # Access individual elements of the row using column names
        seg = row['raw']
        seg_id = row['src_seg_id']
        src_seg = row['src_raw']
        # print(seg_id)
        # print(src_seg)
        # print(seg)
        # exit()
        # each instruction is added to the list of instructions only if
        # the current translated segment exceeds or falls below the threshold
        # depending on the direction of the anticipated translationese deviation
        if args.lang == 'en':
            src_lang = 'German'
            tgt_lang = 'English'
        else:
            src_lang = 'English'
            tgt_lang = 'German'

        if args.approach == 'feature-based':
            if len(seg.split()) > 8:
                # set up functions to access vals for any of 60 feats, thres for that feat and the thres type (add, remove),
                # each function returns an instruction if the current segment deviates from the expected norm
                # (i.e. mean segment value in originals) in the established "translationese" direction, not in the other direction
                ol_tuples = []
                ol_instructions = []
                # collect max potentially applicable instructions and tuples based on the entire feature set of 60 items
                # if args.mode in ['min', 'detailed']:
                for feat in feat_dict[args.lang][args.level]:
                    n_feats = len(feat_dict[args.lang][args.level])
                    # exit()
                    # res_tuple = (feat_name, observed_val, expected_val, thres_type)
                    if args.mode == 'detailed':
                        inst, sanity_tuple = rewriting_oracle(feat_name=feat,
                                                              bins=detailed_seg_instructions[args.lang][feat],
                                                              observed_val=row[feat],
                                                              expected_val=this_lang_thres[feat][0],
                                                              thres_type=this_lang_thres[feat][1],
                                                              org_std=this_lang_thres[feat][2])
                    else:  # min
                        inst, sanity_tuple = rewriting_oracle(feat_name=feat, bins=SCRIPTURES[feat],
                                                              observed_val=row[feat],
                                                              expected_val=this_lang_thres[feat][0],
                                                              thres_type=this_lang_thres[feat][1],
                                                              org_std=this_lang_thres[feat][2])
                    # filter out 'ERROR_nmod', None
                    if inst and not inst.startswith('ERROR'):
                        ol_tuples.append(sanity_tuple)
                        ol_instructions.append(inst)
                # print(seg)
                # print(ol_instructions)
                # input()
                # Sort instructions by anticipated bigger / anticipated smaller ratio and if tiny reduced to 2 top feats
                fired_feats, selected_instructions, mode_selected_tuples = filter_n_sort(
                    my_instr=ol_instructions,
                    my_tuples=ol_tuples,
                    my_ratio=args.thres_type)
                # print(selected_instructions)
                # print(mode_selected_tuples)
                # input()

                global_lst.extend(fired_feats)

                if selected_instructions:
                    string_of_instructions = generate_instuction_string(instructions=selected_instructions)
                    # print(string_of_instructions)
                    task_count, my_prompt = generate_context(src=row['src_raw'], tgt=row['raw'],
                                                             instructions=string_of_instructions,
                                                             slang=src_lang,
                                                             tlang=tgt_lang,
                                                             approach=args.approach,
                                                             mode=args.mode)

                    # print(task_count, my_prompt)
                    prompts_dict['seg_id'].append(seg_id)
                    prompts_dict['src_raw'].append(src_seg)
                    prompts_dict['tgt_raw'].append(seg)
                    prompts_dict['prompt'].append(my_prompt)
                    prompts_dict['prompt_len'].append(task_count * 0.75)
                    prompts_dict['thresholds'].append(mode_selected_tuples)

                    total_wc.append(task_count * 0.75)
                else:  # an empty list
                    prompts_dict['seg_id'].append(seg_id)
                    prompts_dict['src_raw'].append(src_seg)
                    prompts_dict['tgt_raw'].append(seg)
                    prompts_dict['prompt'].append("copy over the translation intact")
                    prompts_dict['prompt_len'].append(0)
                    prompts_dict['thresholds'].append([])

                    low_vratio += 1
            else:  # short segments
                prompts_dict['seg_id'].append(seg_id)
                prompts_dict['src_raw'].append(src_seg)
                prompts_dict['tgt_raw'].append(seg)
                prompts_dict['prompt'].append("bypassed re-writing pipeline")
                prompts_dict['prompt_len'].append(0)
                prompts_dict['thresholds'].append([])

                shorts += 1
        else:  # self_guided
            if len(seg.split()) > 8:
                # this is no-instructions (lazy or expert) approach:
                task_count, my_prompt = generate_context(src=row['src_raw'], tgt=row['raw'],
                                                         instructions=None,
                                                         slang=src_lang,
                                                         tlang=tgt_lang,
                                                         approach=args.approach,
                                                         mode=args.mode)
                prompts_dict['seg_id'].append(seg_id)
                prompts_dict['src_raw'].append(src_seg)
                prompts_dict['tgt_raw'].append(seg)
                prompts_dict['prompt'].append(my_prompt)
                prompts_dict['prompt_len'].append(task_count * 0.75)
                prompts_dict['thresholds'].append([])

                total_wc.append(task_count * 0.75)
            else:
                prompts_dict['seg_id'].append(seg_id)
                prompts_dict['src_raw'].append(src_seg)
                prompts_dict['tgt_raw'].append(seg)
                prompts_dict['prompt'].append("bypassed re-writing pipeline")
                # the context window of 4096 tokens
                prompts_dict['prompt_len'].append(0)
                prompts_dict['thresholds'].append([])

                shorts += 1

    job_size = round(sum(total_wc), 1)
    max_size = round(np.max(total_wc), 1)
    input_costs = job_size / 1000 * 0.0010  # see https://openai.com/pricing#language-models
    input_costs4 = job_size / 1000 * 0.03
    # print(f'{args.lang.upper()}: total size of the job (total_wc * 0.75): {job_size} tokens. Max prompt: {max_size}')
    # print(f'GPT-3.5 API cost (job_size / 1000 * 0.0010): {input_costs} USD (input only!)')
    # print(
    #     f'GPT-4 API cost (job_size / 1000 * 0.03): {input_costs4} USD (input only!, output should be smaller in tokens but twice as expensive)')
    print(f'Number of bypassed segments (ratio < {args.thres_type} for all considered feats): {low_vratio}')
    print(f'Number of copied over segments (len < 8 words ): {shorts}')
    info_on_intensity_dict['lang'] = args.lang
    info_on_intensity_dict['job_size (tokens)'] = job_size
    info_on_intensity_dict['max prompt (tokens)'] = max_size
    info_on_intensity_dict['mean prompt (tokens)'] = round(np.mean(total_wc), 1)
    info_on_intensity_dict['GPT-4 input costs, USD'] = input_costs4
    info_on_intensity_dict['GPT-3.5 input costs, USD'] = input_costs
    info_on_intensity_dict[f'ratio<{args.thres_type}'] = low_vratio
    info_on_intensity_dict[f'length<8'] = shorts

    df = pd.DataFrame(prompts_dict)
    print(df.head())
    print()

    # make sure thres are interpreted as lists, not as strings
    # df['thresholds'] = df['thresholds'].apply(ast.literal_eval)
    if args.approach == 'feature-based':
        global_counter = Counter(global_lst)
        dic_sort = OrderedDict(sorted(global_counter.items(), key=itemgetter(1), reverse=True))
        print()
        tuples = list(dic_sort.items())
        # if args.thres_type == 2.5:
        #     rtype = 'ratio'
        # else:
        #     rtype = 'STD'
        print(f'{args.lang.upper()}: Counts of features that went through the {args.thres_type} cut-off')

        os.makedirs('4_prompting/cutoffs/', exist_ok=True)
        with open(
                f'4_prompting/cutoffs/{args.lang}_{len(feat_dict[args.lang][args.level])}_{args.thres_type}_{args.approach}_{args.mode}.tsv',
                'w') as outf:
            for tu in tuples:
                print('\t'.join(i for i in [tu[0], str(tu[1])]))
                outf.write('\t'.join(i for i in [tu[0], str(tu[1])]) + '\n')

        print("\nDict size", len(list(dic_sort.items())))

        thres_llst = df['thresholds'].tolist()

        most_last, tot_instructed_segs = typical_diviations(thres_lst=thres_llst, order=-1)
        most_lastbutone, _ = typical_diviations(thres_lst=thres_llst, order=-2)
        print()
        print(f'Most last: {most_last}')
        print(f"Most lastbutone: {most_lastbutone}")

        nums = [len(lst) for lst in thres_llst if lst]  # collect num instructions skipping empty lists
        print(len(thres_llst), len(nums))
        assert len(thres_llst) >= len(nums), 'Huston, we have problems!'

        num_instr_min = np.min(nums)
        num_instr_max = np.max(nums)
        num_instr_mean = round(np.mean(nums), 1)

        info_on_intensity_dict['min_instr'] = num_instr_min
        info_on_intensity_dict['max_instr'] = num_instr_max
        info_on_intensity_dict['mean_instr'] = num_instr_mean
        info_on_intensity_dict['most_last%'] = most_last
        info_on_intensity_dict['most_lastbutone%'] = most_lastbutone
        info_on_intensity_dict['instructed_segs'] = tot_instructed_segs

        bypassed = df['prompt'].str.count('bypassed').sum()
        copy_over = df['prompt'].str.contains('copy over the translation intact').sum()
        tot_skipped = bypassed + copy_over
        print(f'The substrings "bypassed or copy-over" occurs {tot_skipped} times in the "prompt" column,')
        print(f'i.e. we are not sending {tot_skipped / df.shape[0] * 100:.2f}% for {args.lang} to the model')

    # ====== this is my key output =======
    if args.approach == 'feature-based':
        oname = f'{args.outdir}{args.lang}_{args.level}_{args.approach}_{args.mode}_{args.thres_type}.tsv'
    else:
        # threshold does not matter for feature-less approaches: self-guided and translated
        oname = f'{args.outdir}{args.lang}_{args.level}_{args.approach}_{args.mode}.tsv'
    df.to_csv(oname, sep='\t', index=False)

    stats_df = pd.DataFrame(info_on_intensity_dict, index=[0])

    stats_df.to_csv(
        f'{args.statsto}{args.lang}_{args.level}_{args.approach}_{args.mode}_how_much_instr={args.thres_type}.tsv',
        sep='\t', index=False)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
