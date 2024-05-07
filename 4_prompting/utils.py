"""
2 October 2023
this is a module to be imported to en_template.py and de_template.py

functions required to extract the best-performing features from UD formatted input
including obtaining and processing conllu output for each segment

'de': 'mean_sent_wc', 'obl', 'addit', 'case', 'numcls', 'sconj', 'advers', 'advmod', 'xcomp', 'compound', 'fin', 'nmod', 'advcl', 'mhd'
'en': 'nmod', 'pastv', 'ttr', 'poss', 'mhd', 'advers', 'nnargs', 'mdd', 'numcls', 'advmod', 'parataxis', 'addit', 'fin'

"""

from collections import defaultdict

import numpy as np
from igraph import *
from igraph._igraph import arpack_options


def get_thres(my_df=None, thres_type=None):
    this_dict = defaultdict()

    feats = my_df['feature'].tolist()
    org_seg_mean = my_df['org_mean'].tolist()
    org_seg_std = my_df['org_std'].tolist()

    for feat, val, std in zip(feats, org_seg_mean, org_seg_std):
        this_dict[feat] = (val, thres_type, std)

    return this_dict


def get_features_thresholds(df0=None, lang=None):
    df0 = df0[df0['lang'] == lang]

    remove_thres = df0[~df0['compare'].str.contains('--')]
    add_thres = df0[df0['compare'].str.contains('--')]

    remove_thres_dict = get_thres(my_df=remove_thres, thres_type='remove')
    add_thres_dict = get_thres(my_df=add_thres, thres_type='add')

    remove_thres_dict.update(add_thres_dict)
    return remove_thres_dict


# shared DE/EN functions (7): nmod, mhd, advers, numcls, advmod, addit, fin (in the order of importance for German)
def ud_freqs(trees, rel=None, thres=None, thres_type=None):
    remove_ud_rel_instructions = {
        'acl': 'avoid attributive clauses',
        'obj': 'avoid structures with direct objects',
        'ccomp': 'avoid structures with clausal objects',
        'nmod': 'avoid nominal modifiers of nouns such as prepositional and genitive case phrases',
        'parataxis': 'avoid explicit connectives',
        'obl': 'avoid prepositional phrases in adverbial function',
        'case': 'ERROR_case',  # this is colinear with obl in EN
        'xcomp': 'avoid non-finite constructions',
        'compound': 'ERROR_compound',
        'advcl': 'avoid adverbial clauses'
    }
    add_ud_rel_instructions = {
        'acl': 'allow more attributive clauses',
        'obj': 'prefer structures with direct objects',
        'ccomp': 'prefer structures with clausal objects',
        'nmod': 'ERROR_nmod',
        'parataxis': 'ERROR_parataxis',
        'obl': 'ERROR_obl',
        'case': 'ERROR_case',
        'xcomp': 'prefer non-finite constructions',
        'compound': 'prefer nominal compounds',
        'advcl': 'ERROR_advcl'
    }

    # remove_ud_rel_instructions = {
    #     'advmod': 'remove non-clausal adverbials, i.e. modifiers of predicates or other modifiers (ex. Er weiß es möglicherweise selbst noch nicht)',
    #     'nmod': 'use fewer nominal modifiers of nouns, including prepositional phrases (ex. the house with the big garden) and genitive case (the number of participants)',
    #     'parataxis': 'use asyndeton (parataxis) more often, i.e. if possible avoid using explicit connectives',
    #     'obl': 'avoid adverbials realised as prepositional nominal phrases (ex. unfortunately for you)',
    #     'case': 'ERROR_case',
    #     'xcomp': 'avoid clausal complements without its own subject such as: I started to work there yesterday, We expect them to change their minds.',
    #     'compound': 'ERROR_compound',
    #     'advcl': 'use fewer adverbial clauses, finite or non-finite'
    # }
    # add_ud_rel_instructions = {
    #     'advmod': 're-phrase adding non-clausal adverbials, i.e. modifiers of predicates or other modifiers (ex. Er weiß es möglicherweise selbst noch nicht).',
    #     'nmod': 'ERROR_nmod',
    #     'parataxis': 'ERROR_parataxis',
    #     'obl': 'ERROR_obl',
    #     'case': 'ERROR_case',
    #     'xcomp': 'ERROR_xcomp',
    #     'compound': 'where possible use pre-posed nominal attributes for other nouns, ex. ice cream flavors, phone book',
    #     'advcl': 'ERROR_advcl'
    # }

    tot_seg_counts = []
    for tree in trees:
        sent_relations = [w[6] for w in tree]

        this_sent_rel_counts = sent_relations.count(rel)
        tot_seg_counts.append(this_sent_rel_counts)

    res = np.sum(tot_seg_counts) / len(trees)
    # print(rel, res, thres, thres_type)
    # thres_type is needed to avoid re-writing ALL sentences, but limit LLM interference only to cases
    # where the existing translations exhibit translationese-like deviations
    res_tuple = (rel, res, thres, thres_type)
    if thres_type == 'remove':
        if res > thres:
            return remove_ud_rel_instructions[rel], res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res <= thres:
            return add_ud_rel_instructions[rel], res_tuple
        else:
            return None, res_tuple
    else:
        print(f'{rel} ERROR!')
        exit()


# mhd support function: this function tests connectedness of the graph; it helps to skip malformed sentences
def test_sanity(sent):
    arpack_options.maxiter = 3000
    sentence_graph = Graph(len(sent) + 1)

    sentence_graph = sentence_graph.as_directed()
    sentence_graph.vs["name"] = ['ROOT'] + [word[1] for word in sent]  # string of vertices' attributes called name

    sentence_graph.vs["label"] = sentence_graph.vs["name"]
    edges = [(word[5], word[0]) for word in sent if word[6] != 'punct']  # (int(identifier), int(head), token, rel)

    sentence_graph.add_edges(edges)
    sentence_graph.vs.find("ROOT")["shape"] = 'diamond'
    sentence_graph.vs.find("ROOT")["size"] = 40
    disconnected = [vertex.index for vertex in sentence_graph.vs if vertex.degree() == 0]
    sentence_graph.delete_vertices(disconnected)

    return sentence_graph  # , bad_trees


# mhd: cost of linearising a hierarchical idea, depth of the tree, complexity of the idea
def speakdiff(segment, thres=None, thres_type=None):
    # some segments can have more than one tree!
    across_segment_trees = []
    errors = 0
    for tree in segment:
        # call speakdiff_visuals function (needs revision!) to get the graph and counts for disintegrated trees
        # (syntactic analysis errors of assigning dependents to punctuation)
        graph = test_sanity(tree)
        parts = graph.components(mode=WEAK)
        all_hds = []
        if len(parts) == 1:
            nodes = [word[1] for word in tree if word[6] != 'punct' and word[6] != 'root']
            for node in nodes:
                try:
                    hd = graph.distances('ROOT', node, mode=ALL)[0][0]  # [[3]]
                    all_hds.append(hd)
                except ValueError:
                    errors += 1
                    print(tree)
                    print(node)
                    print(f'{" ".join(w[1] for w in tree)}')
                    hd = 1
                all_hds.append(hd)  # fake hd to avoid none

            if all_hds:
                # ignore nans, I have three annoying cases
                mhd0 = np.nanmean(all_hds)  # np.average
            else:
                mhd0 = 2  # fake hd to avoid none
        else:
            mhd0 = 1

        across_segment_trees.append(mhd0)
    res = np.nanmean(across_segment_trees)
    res_tuple = ('mhd', res, thres, thres_type)
    if thres_type == 'remove':
        if res > thres:
            return 'avoid chains of elements dependent on each other', res_tuple
            # return 'avoid structures where dependent syntactic elements have chains of other dependents ' \
            #        '(ex. This is the cat that killed the rat that ate the malt that lay in the house that Jack built .)', res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res <= thres:
            return 'produce more complex sentences', res_tuple
            # return 'produce syntactic structures with deeper hierarchy, i.e. allow dependents have their own dependents', res_tuple
        else:
            return None, res_tuple
    else:
        print('ERROR! mhd')
        exit()
        # return None, res_tuple


# for advers(ative), addit(ive)
def count_dms(trees, lang=None, list_link=None, dm_feat=None, thres=None, thres_type=None):
    if dm_feat == 'advers':
        lst = [i.strip() for i in open(f"{list_link}{lang}_adversative.lst", 'r').readlines()]
    elif dm_feat == 'addit':
        lst = [i.strip() for i in open(f"{list_link}{lang}_additive.lst", 'r').readlines()]
    elif dm_feat == 'caus':
        lst = [i.strip() for i in open(f"{list_link}{lang}_causal.lst", 'r').readlines()]
    else:
        lst = [i.strip() for i in open(f"{list_link}{lang}_temp_sequen.lst", 'r').readlines()]
    res = 0
    wc = 0
    for tree in trees:
        wc += len(tree)
        sent = ' '.join(w[1] for w in tree)
        for i in lst:
            try:
                if i[0].isupper():
                    padded0 = i + ' '
                    if padded0 in sent:
                        res += 1
                else:
                    padded1 = ' ' + i + ' '

                    # if not surrounded by spaces gets: the best areAS FOR expats for as for
                    # ex: enTHUSiastic for thus

                    if padded1 in sent or i.capitalize() + ' ' in sent:
                        # capitalization changes the count for additive from 1822 to 2182
                        # (or to 2120 if padded with spaces)
                        # (furthermore: 1 to 10)
                        res += 1
            except IndexError:
                print('out of range index (blank line in list?: \n', i)
                pass
    examples = {'advers': 'by contrast, instead, on the contrary, conversely, as opposed to, rather than, '
                          'but anyway, nevertheless, aside from this',
                'addit': 'additionally, besides, by the way, i mean, in other words, moreover, '
                         'more accurately, on top of that, what is more'}
    my_map = {'advers': 'adversative', 'addit': 'additive', 'caus': 'causative', 'tempseq': 'temporal-sequencial'}
    res = res / wc  # len(trees)  should be normalised to wc, not n_sents!
    res_tuple = (dm_feat, res, thres, thres_type)
    if thres_type == 'remove':
        if res > thres:
            return f'remove phrases that introduce {my_map[dm_feat]} information', res_tuple
            # return f'remove some of the {dm_type} discourse markers, i.e. parenthetical constructions similar to {examples[dm_type]}', res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res <= thres:
            # constructions -> phrases
            return f'add phrases that introduce {my_map[dm_feat]} information', res_tuple
            # return f'add appropriate {dm_type} discourse markers, i.e. parenthetical constructions similar to {examples[dm_type]}', res_tuple
        else:
            return None, res_tuple
    else:
        print(f'{examples[dm_feat]} dms ERROR!')
        exit()
        # return None, res_tuple


def advmod_no_negation(trees, thres=None, thres_type=None):
    res = 0
    for tree in trees:
        for w in tree:
            if w[6] == 'advmod':
                if '=Neg' not in w[4]:
                    res += 1

    res = res / len(trees)
    res_tuple = ('advmod', res, thres, thres_type)

    if thres_type == 'remove':
        if res > thres:
            return 'avoid adverbial modifiers', res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res <= thres:
            return 'use more adverbial modifiers', res_tuple
            # return 're-phrase to have more dependent clauses', res_tuple
        else:
            return None, res_tuple
    else:
        print('advmod ERROR!')
        exit()
        # return None, res_tuple


# numcls
def sents_complexity(trees, thres=None, thres_type=None):
    types = ['csubj', 'advcl', 'acl', 'parataxis', 'ccomp']  # 'xcomp',
    simples = 0
    clauses_counts = []
    for tree in trees:
        this_sent_cls = 0
        for w in tree:
            if w[6] in types:
                this_sent_cls += 1
        if this_sent_cls == 0:
            simples += 1
        clauses_counts.append(this_sent_cls)

    res = np.average(clauses_counts)
    res_tuple = ('numcls', res, thres, thres_type)
    if thres_type == 'remove':
        if res > thres:
            return 'use fewer clauses', res_tuple
            # return 'reduce the number of dependent clauses', res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res <= thres:
            return 'use more dependent clauses', res_tuple
            # return 're-phrase to have more dependent clauses', res_tuple
        else:
            return None, res_tuple
    else:
        print('numcls ERROR!')
        exit()
        # return None, res_tuple


# fin
def finites(segment, thres=None, thres_type=None):
    fins = 0
    wc = 0
    for tree in segment:
        wc += len(tree)
        for w in tree:
            if 'VerbForm=Fin' in w[4]:
                fins += 1
    res = fins / len(segment)
    # print(f'Finites: {thres}, {thres_type}')
    res_tuple = ('fin', res, thres, thres_type)
    if thres_type == 'remove':
        if res > thres:
            return 'use non-finite constructions instead of full clauses', res_tuple
            # return 'reduce the number of finite verbs', res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res <= thres:
            return 'use full clauses instead of non-finite constructions', res_tuple
            # return 'increase the number of finite verbs', res_tuple
        else:
            return None, res_tuple
    else:
        print('fins ERROR!')
        exit()
        # return None, res_tuple


# German specific features, in the order of importance (6): pastv, ttr, poss, nnargs, mdd, parataxis
# pastv
def pasttense(segment, thres=None, thres_type=None):
    res = 0
    wc = 0
    for tree in segment:
        wc += len(tree)
        for w in tree:
            if 'Tense=Past' in w[4]:
                res += 1
    res = res / len(segment)
    res_tuple = ('pastv', res, thres, thres_type)
    if thres_type == 'remove':
        if res > thres:
            return 'avoid Präteritum in favour of other ways of expressing past', res_tuple
            # return 'avoid using verbs in past tense (Präteritum)', res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res <= thres:
            return 'prefer Präteritum to other ways of expressing past', res_tuple
            # return 'prefer the past tense of the verb (Präteritum) to other options', res_tuple
        else:
            return None, res_tuple
    else:
        print('pasttense ERROR!')
        exit()
        # return None, res_tuple


# ttr
# exclude PUNCT, SYM, X from tokens
def content_ttr_and_density(segment, thres=None, thres_type=None):
    # some segments can have more than one tree!
    # print(f'TTR: {thres}, {thres_type}')
    counter = 0
    across_segment_trees_ttr = []
    across_segment_trees_dens = []
    for tree in segment:
        content_types = []
        content_tokens = []
        for w in tree:
            if 'ADJ' in w[3] or 'ADV' in w[3] or 'VERB' in w[3] or 'NOUN' in w[3]:
                content_type = w[2] + '_' + w[3]
                content_types.append(content_type)
                content_token = w[1] + '_' + w[3]
                content_tokens.append(content_token)
        tree_types = len(set(content_types))
        tree_tokens = len(content_tokens)
        try:
            tree_ttr = tree_types / tree_tokens
        except ZeroDivisionError:
            # long sentences without content words (errors!: ex. Aber wenn die.
            tree_ttr = 0
        try:
            tree_dens = tree_tokens / len(tree)
        except ZeroDivisionError:
            # print('zero length tree? How?')
            # print(segment)
            tree_dens = 0
            counter += 1

        across_segment_trees_ttr.append(tree_ttr)
        across_segment_trees_dens.append(tree_dens)

    res = np.average(across_segment_trees_ttr)
    res_tuple = ('ttr', res, thres, thres_type)
    if thres_type == 'remove':
        if res > thres:
            # return 'use more standardised vocabulary, avoid new coinages and ad hoc compounding', res_tuple
            return 'avoid rare words and non-standard contextual vocabulary', res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res < thres:
            # use more varied vocabulary,
            return 'avoid repetitions', res_tuple
        else:
            return None, res_tuple
    else:
        print('ttr ERROR!')
        exit()
        # return None, res_tuple


# poss
def possdet(segment, thres=None, thres_type=None):  # lang=None,
    res = 0
    wc = 0
    for tree in segment:
        wc += len(tree)
        for w in tree:
            # lemma = w[2].lower()
            # # own and eigen are not included as they do not compare to свой, it seems
            # if lang == 'en':
            #     if lemma in ['my', 'your', 'his', 'her', 'its', 'our', 'their']:
            #         if w[3] in ['DET', 'PRON'] and 'Poss=Yes' in w[4]:
            #             res += 1
            #             # matches.append(w[2].lower())
            # elif lang == 'de':
            #     if lemma in ['mein', 'dein', 'sein', 'ihr', 'Ihr|ihr', 'unser', 'eurer']:  # eurer does not occur
            if w[3] in ['DET', 'PRON'] and 'Poss=Yes' in w[4]:
                res += 1
                # matches.append(w[2].lower())
    res = res / wc
    res_tuple = ('poss', res, thres, thres_type)
    if thres_type == 'remove':
        if res > thres:
            return 'replace some possessive pronouns with determiners', res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res <= thres:
            # this might require access to document level
            return 'use possessive pronouns instead of determiners', res_tuple
        else:
            return None, res_tuple
    else:
        print('poss det ERROR!')
        exit()
        # return None, res_tuple


# nnargs
# ratio of NOUNS+proper functioning as core verbal arguments to the count of all core verbal arguments
def nouns_to_all_args(trees, thres=None, thres_type=None):
    count = 0
    nouns = 0
    for tree in trees:
        for w in tree:
            if w[6] in ['nsubj', 'obj', 'iobj']:
                count += 1
                if w[3] == 'NOUN' or w[3] == 'PROPN':
                    nouns += 1
    try:
        # nouns-as-core-args over all core args
        res = nouns / count
    except ZeroDivisionError:
        res = 0

    res = res / len(trees)
    res_tuple = ('nnargs', res, thres, thres_type)
    if thres_type == 'remove':
        if res > thres:
            # this requires access to document!
            return 'use pronouns as verbal arguments where possible', res_tuple
            # return 'reduce the proportion of nouns functioning as core verbal arguments, i.e. subjects, objects and indirect objects', res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res <= thres:
            return 'make sure that verbs only have nouns or proper names as dependents', res_tuple
            # return 'increase the proportion of nouns functioning as core verbal arguments, i.e. subjects, objects and indirect objects', res_tuple
        else:
            return None, res_tuple
    else:
        print('nnargs ERROR!')
        exit()
        # return None, res_tuple


# mdd
# calculate comprehension difficulty=mean dependency distance(MDD) as “the distance between words and their parents,
# measured in terms of intervening words.” (Hudson 1995 : 16)
def readerdiff(segment, thres=None, thres_type=None):
    # some segments can have for than one tree!
    across_segment_trees = []
    for tree in segment:
        dd = []
        s = [q for q in tree if q[6] != 'punct']
        if len(s) > 1:
            for w1 in s:
                s_word_id = w1[0]
                head_id = w1[5]  # use conllu 1-based index
                if head_id == 0:  # root
                    continue
                s_head_id = None
                for w2 in tree:
                    s_word_id_2 = w2[0]
                    if head_id == s_word_id_2:
                        s_head_id = s_word_id_2
                        break
                try:
                    dd.append(abs(s_word_id - s_head_id))
                except TypeError:
                    # in some trees there are wds that are not dependent on any other el:
                    # el that is listed in their w[5] is not in the tree?
                    dd.append(1)
        else:
            dd = [1]  # for sentences like Bravo ! Nein ! Warum ? I need a reasonable floor

        # use this function instead of overt division of list sum by list length: if smth is wrong you'll get a warning!
        mdd0 = np.nanmean(dd)
        across_segment_trees.append(mdd0)
    res = np.nanmean(across_segment_trees)

    # I still get NaNs in 96 short segments! I will just drop these segments! in classifier and feature analysis
    if not res:
        res = 1
    # average MDD for the segment, including 1 for Bravo !, Warum ? two-token segments with only a root and a punct
    res_tuple = ('mdd', res, thres, thres_type)
    if thres_type == 'remove':
        if res > thres:
            return 'keep words closer to their syntactic heads', res_tuple
            # return 'reduce the distance between syntactic heads and their dependents in terms of intervening words', res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res <= thres:
            return 'allow more intervening tokens between words and their syntactic heads', res_tuple
            # return 'increase the distance between syntactic heads and their dependents in terms of intervening words', res_tuple
        else:
            return None, res_tuple
    else:
        print('mdd ERROR!')
        exit()
        # return None, res_tuple


############ English specific features, in the order of importance (7): mean_sent_wc, obl, case, sconj, xcomp, compond, advcl

# mean_sent_wc
def mean_sent_length(trees, thres=None, thres_type=None):
    sent_lengths_lst = []
    for tree in trees:
        sent_lengths_lst.append(len(tree))

    res = np.average(sent_lengths_lst)
    # consistency in feature names, please!
    res_tuple = ('mean_sent_wc', res, thres, thres_type)
    if thres_type == 'remove':
        if res > thres:
            return 'make sentences shorter', res_tuple
            # return 'reduce the sentence length', res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res <= thres:
            return 'make the sentences a bit longer', res_tuple
        else:
            return None, res_tuple
    else:
        print('sent_length ERROR!')
        exit()
        # return None, res_tuple


# sconj
def sconj(segment, thres=None, thres_type=None):  # , lang
    res = 0
    wc = 0
    for tree in segment:
        wc += len(tree)
        for w in tree:
            # if lang == 'en':
            #     if 'SCONJ' in w[3]:  # and w[2] in ['that', 'if', 'as', 'of', 'while', 'because', 'by', 'for', 'to', 'than',
            #                                     # 'whether', 'in', 'about', 'before', 'after', 'on', 'with', 'from', 'like',
            #                                     # 'although', 'though', 'since', 'once', 'so', 'at', 'without', 'until',
            #                                     # 'into', 'despite', 'unless', 'whereas', 'over', 'upon', 'whilst', 'beyond',
            #                                     # 'towards', 'toward', 'but', 'except', 'cause', 'together']:
            #         count += 1
            #
            # elif lang == 'de':
            if 'SCONJ' in w[3]:  # and w[2] in ['daß', 'wenn', 'dass', 'weil', 'da', 'ob', 'wie', 'als',
                # 'indem', 'während', 'obwohl', 'wobei', 'damit', 'bevor',
                # 'nachdem', 'sodass', 'denn', 'falls', 'bis', 'sobald',
                # 'solange', 'weshalb', 'ditzen', 'sofern', 'warum', 'obgleich',
                # 'zumal', 'sodaß', 'aber', 'wenngleich', 'wennen', 'wodurch',
                # 'wohingegen', 'ehe', 'worauf', 'seit', 'inwiefern', 'anstatt', 'der',
                # 'vordem', 'insofern', 'nahezu', 'wohl', 'manchmal', 'weilen', 'weiterhin',
                # 'doch', 'mit', 'gleichfalls']:
                res += 1
    res = res / wc
    res_tuple = ('sconj', res, thres, thres_type)
    if thres_type == 'remove':
        if res > thres:
            return 'make subordinating relations between clauses more implicit', res_tuple
            # return 'use fewer words functioning as subordinating conjunctions such as while, because, if, though, whereas, since', res_tuple
        else:
            return None, res_tuple
    elif thres_type == 'add':
        if res <= thres:
            return 'make subordinating relations between clauses more explicit', res_tuple
            # return 'use more words functioning as subordinating conjunctions such as while, because, if, though, whereas, since', res_tuple
        else:
            return None, res_tuple
    else:
        print('sconj ERROR!')
        exit()
        # return None, res_tuple


def get_tree(data):
    current_sentence = []
    for line in data:
        if len(line) > 2:
            if line.startswith('<'):
                continue

            res = line.strip().split('\t')
            try:
                (identifier, token, lemma, upos, feats, head, rel) = res
            except ValueError:
                print('Am I here in get_trees')
                print(line)
                exit()
            if '.' in identifier or '-' in identifier:  # ignore empty nodes possible in the enhanced representations
                continue
            try:
                current_sentence.append([int(identifier), token, lemma, upos, feats, int(head), rel])
            except ValueError:
                # this error is caused by a few force-added 2 . . SENT _ _ _ without true head id
                current_sentence.append([int(identifier), token, lemma, upos, feats, int(identifier) - 1, rel])

    return current_sentence
