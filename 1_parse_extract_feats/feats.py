import re
import numpy as np
from igraph import *
from igraph._igraph import arpack_options
from itertools import product

from helpfunctions import get_kids_lem, get_kids_pos, choose_kid_by_posfeat, choose_kid_by_posrel, get_kids_feats, get_headwd
from helpfunctions import has_auxkid_by_lem, choose_kid_by_lempos, choose_kid_by_featrel, get_kids

# identifier, token, lemma, upos, feats, head, rel


def personal_pron(segment, lang):
    count = 0
    for tree in segment:

        # matches = []
        for w in tree:
            token = w[1].lower()
            if lang == 'en':
                if 'PRON' in w[3] and 'Person=' in w[4] and 'Poss=Yes' not in w[4]:
                    if token in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']:
                        count += 1
                        # matches.append(w[2].lower())
            elif lang == 'de':
                if 'PRON' in w[3] and 'Person=' in w[4] and 'Poss=Yes' not in w[4]:
                    if token in ['ich', 'ihr', 'du', 'er', 'sie', 'es', 'wir', 'mich', 'mir', 'dich', 'dir', 'ihm', 'ihn',
                                 'uns', 'ihnen']:
                        count += 1
                        # matches.append(w[2].lower())
            elif lang == 'ru':
                if 'PRON' in w[3] and 'Person=' in w[4] and 'Poss=Yes' not in w[4]:
                    if token in ['я', 'ты', 'вы', 'он', 'она', 'оно', 'мы', 'они', 'меня', 'тебя', 'его', 'её', 'ее',
                                 'нас', 'вас', 'их', 'неё', 'нее', 'него', 'них', 'мне', 'тебе', 'ей', 'ему', 'нам', 'вам',
                                 'им', 'ней', 'нему', 'ним', 'меня', 'тебя', 'него', 'мной', 'мною', 'тобой', 'тобою',
                                 'Вами', 'им', 'ей', 'ею', 'нами', 'вами', 'ими', 'ним', 'нем', 'нём', 'ней', 'нею',
                                 'ними']:
                        count += 1
                        # matches.append(w[2].lower())

    return count  # , matches


def possdet(segment, lang):
    count = 0
    for tree in segment:
        for w in tree:
            if w[3] in ['DET', 'PRON'] and 'Poss=Yes' in w[4]:
                count += 1
            # lemma = w[2].lower()
            # # own and eigen are not included as they do not compare to свой, it seems
            # if lang == 'en':
            #     if lemma in ['my', 'your', 'his', 'her', 'its', 'our', 'their']:
            #         if w[3] in ['DET', 'PRON'] and 'Poss=Yes' in w[4]:
            #             count += 1
            #             # matches.append(w[2].lower())
            # elif lang == 'de':
            #     if lemma in ['mein', 'dein', 'sein', 'ihr', 'Ihr|ihr', 'unser', 'eurer']:  # eurer does not occur
            #         if w[3] in ['DET', 'PRON'] and 'Poss=Yes' in w[4]:
            #             count += 1
            #             # matches.append(w[2].lower())
            # elif lang == 'ru':
            #     if 'DET' in w[3] and lemma in ['мой', 'твой', 'ваш', 'его', 'ее', 'её', 'наш', 'их', 'ихний', 'свой']:
            #         count += 1
                    # matches.append(w[2].lower())

    return count


# include noun substituters, i.e. pronouns par excellence, of indefinite+total and excluding negative semantic subtypes
def anysome(segment, lang):
    count = 0
    for tree in segment:
        for w in tree:
            if lang == 'en':
                if w[2] in ['anybody', 'anyone', 'anything', 'everybody', 'everyone', 'everything',
                             'somebody', 'someone', 'something',
                            'elsewhere', 'everywhere', 'somewhere', 'anywhere']:
                    count += 1
            elif lang == 'de':
                if w[2] in ['etwas', 'irgendetwas', 'irgendwelch', 'irgendwas', 'jedermann', 'jedermanns', 'jemand',
                            'alles', 'irgendwo', 'manch'] and 'PronType=Ind' in w[4]:
                    count += 1
            elif lang == 'ru':
                if w[2] in ['некто', 'нечто', 'нечего'] and w[3] == 'PRON':
                    # 'никто', 'ничто', 'нигде', 'никуда', 'ниоткуда'
                    count += 1
                if re.search(r'-то|-нибудь|-либо', w[2], re.UNICODE) and 'какой' not in w[2]:
                    '''
                    какой-нибудь, любой и всякий учитываются в demdeterm
                    '''
                    count += 1
                if re.match('кое', w[2], re.UNICODE) and 'какой' not in w[2]:
                    count += 1
                if w[2] in ['кто-кто', 'кого-кого', 'кому-кому', 'кем-кем', 'ком-ком', 'что-что',
                            'чего-чего', 'чему-чему', 'чем-чем', 'куда-куда', 'где-где']:
                    count += 1

    return count


def cconj(segment, lang):
    count = 0
    for tree in segment:
        for w in tree:
            if lang == 'en':
                if 'CCONJ' in w[3]:  # and w[2] in ['and', 'but', 'or', 'both', 'yet', 'either',
                                                # '&', 'nor', 'plus', 'neither', 'ether']:
                    count += 1
            if lang == 'de':
                if 'CCONJ' in w[3]:  # and w[2] in ['und', 'oder', 'aber', 'sondern', 'sowie', 'als', 'wie', 'doch',
                                                # 'sowohl', 'desto', 'noch', 'weder', 'entweder', 'bzw',
                                                # 'beziehungsweise', 'weshalb', 'und/oder', 'ob', 'woher', 'wenn',
                                                # 'jedoch', 'wofür', 'insbesondere', 'obwohl', 'um']:
                    count += 1
            elif lang == 'ru':
                if 'CCONJ' in w[3] and w[2] in ['и', 'а', 'но', 'или', 'ни', 'да', 'причем', 'либо', 'зато', 'иначе',
                                                'только', 'ан', 'и/или', 'иль']:
                    count += 1

    return count


def sconj(segment, lang):
    count = 0
    for tree in segment:
        for w in tree:
            if lang == 'en':
                if 'SCONJ' in w[3]: # and w[2] in ['that', 'if', 'as', 'of', 'while', 'because', 'by', 'for', 'to', 'than',
                                                # 'whether', 'in', 'about', 'before', 'after', 'on', 'with', 'from', 'like',
                                                # 'although', 'though', 'since', 'once', 'so', 'at', 'without', 'until',
                                                # 'into', 'despite', 'unless', 'whereas', 'over', 'upon', 'whilst', 'beyond',
                                                # 'towards', 'toward', 'but', 'except', 'cause', 'together']:
                    count += 1

            elif lang == 'de':
                if 'SCONJ' in w[3]:  # and w[2] in ['daß', 'wenn', 'dass', 'weil', 'da', 'ob', 'wie', 'als',
                                                # 'indem', 'während', 'obwohl', 'wobei', 'damit', 'bevor',
                                                # 'nachdem', 'sodass', 'denn', 'falls', 'bis', 'sobald',
                                                # 'solange', 'weshalb', 'ditzen', 'sofern', 'warum', 'obgleich',
                                                # 'zumal', 'sodaß', 'aber', 'wenngleich', 'wennen', 'wodurch',
                                                # 'wohingegen', 'ehe', 'worauf', 'seit', 'inwiefern', 'anstatt', 'der',
                                                # 'vordem', 'insofern', 'nahezu', 'wohl', 'manchmal', 'weilen', 'weiterhin',
                                                # 'doch', 'mit', 'gleichfalls']:
                    count += 1

            elif lang == 'ru':
                if 'SCONJ' in w[3] and w[2] in ['что', 'как', 'если', 'чтобы', 'то', 'когда', 'чем', 'хотя', 'поскольку',
                                                'пока', 'тем', 'ведь', 'нежели', 'ибо', 'пусть', 'будто', 'словно', 'дабы',
                                                'раз', 'насколько', 'тот', 'коли', 'коль', 'хоть', 'разве', 'сколь',
                                                'ежели', 'покуда', 'постольку']:
                    count += 1

    return count


def word_length(segment):
    words = 0
    letters = 0
    for tree in segment:
        if tree:
            for el in tree:
                if not el[1] in ['.', ',', '!', '?', ':', ';', '"', '-', '—', '(', ')']:
                    words += 1
                    for let in el[1]:
                        letters += 1
                else:
                    continue
        else:
            print('EMPTY TREE')
            print(tree)
    if words == 0:
        av_wordlength = 0
    else:
        av_wordlength = letters / words

    return av_wordlength


def nn(segment, lang):
    count = 0
    for tree in segment:
        if lang in ['en', 'de', 'ru']:
            for w in tree:
                lemma = w[2].lower()
                if 'NOUN' in w[3]:
                    count += 1

    return count


def modpred(segment, lang, list_link=None):
    lang_lst = open(f"{list_link}{lang}_modal-adj_predicates.lst", 'r').readlines()
    mpred_lst = []
    for adj in lang_lst:
        adj = adj.strip()
        mpred_lst.append(adj)

    counter_can = 0
    counter_haveto = 0
    counter_adj = 0
    counter_adv = 0
    mpred = 0
    for tree in segment:
        # matches = []
        if lang == 'en':
            for w in tree:
                if w[4] == 'MD' and w[2] != 'will' and w[2] != 'shall':
                    mpred += 1
                if w[2] in mpred_lst:
                    kids_pos = get_kids_pos(w, tree)
                    if 'AUX' in kids_pos:
                        counter_adj += 1
                if w[2] == 'have' and w[3] != 'AUX':
                    own_id = w[0]
                    inf_kid_id_conllu = choose_kid_by_posfeat(w, tree, 'VERB', 'VerbForm=Inf')
                    if inf_kid_id_conllu != None and abs(own_id - inf_kid_id_conllu) < 4:
                        causative1_conllu = choose_kid_by_posrel(w, tree, 'NOUN', 'obj')
                        if causative1_conllu:
                            if causative1_conllu < inf_kid_id_conllu:
                                '''
                                this set of rules gets:
                                you have time to practise more
                                have a colleague throw another ball onto the table
                                '''
                                continue
                        else:
                            counter_haveto += 1
            mpred = mpred + counter_haveto + counter_adj

        if lang == 'de':
            for w in tree:
                if w[2] in ['dürfen', 'können', 'mögen', 'müssen', 'sollen', 'wollen']:
                    mpred += 1

                elif w[2] in mpred_lst:
                    '''
                    Es ist jedoch offensichtlich, dass weiterhin Druck ausgeübt wird.
                    Es ist unmöglich abzulehnen.
                    Selbstreparatur ist immerhin möglich .
                    Natürlich ist es notwendig , Europas Wettbewerbsfähigkeit zu verbessern
                    Es ist klar , dass
                    Auch auf dem Arbeitsmarkt sind zusätzliche Impulse notwendig .
                    In dem Gespräch mit den Menschen wurde uns klar , daß
                    '''
                    kids_lem = get_kids_lem(w, tree)
                    if 'sein' in kids_lem or 'werden' in kids_lem:
                        counter_adj += 1
            '''
            other modal verbs -- verstehen, pflegen, drohen, pflegen scheinen -- are not included;
            they are omitted from analysis for other langs, too
    
            DONE TODO что-то я не нашла в немецком модальных структур типа:
                RU: в Америке должны наконец задаться вопросом, как ни банально это звучит ,...желая мира, нужно готовиться к войне .
                    Бороться с националистическим подтекстом, доказывать вину этих уродов - жизненно необходимо
                    DE: ... sind von entscheidender Bedeutung.
                EN: But with all that said, I'm not sure Putin is panicking.
                        Trotzdem bin ich mir nicht sicher, ob Putin in Panik gerät.
                    It is obvious, however, that pressure continues to be applied
                        Es ist jedoch offensichtlich, dass weiterhin Druck ausgeübt wird
                    And shame is likely what Trump supporters will feel if he wins .
                    He is likely to decline. -- Er wird wahrscheinlich ablehnen.
            Если такие есть нужен список соответствующих прилагательных (см. EN-RU списки)
            '''
            mpred = mpred + counter_adj  # + sein_mpred
        if lang == 'ru':
            for w in tree:
                # 2 verbs
                if w[2] == 'мочь':
                    counter_can += 1
                if w[2] == 'следовать':
                    kids_pos = get_kids_pos(w, tree)
                    kids_gr = get_kids_feats(w, tree)
                    if 'VERB' in kids_pos and 'VerbForm=Inf' in kids_gr:
                        counter_haveto += 1
                # 3 modal adverbs
                if w[2] == 'можно' or w[2] == 'нельзя' or w[2] == 'надо':
                    counter_adv += 1
                # 11 listed short ADJ
                if w[2] in mpred_lst and 'Variant=Short' in w[4]:
                    counter_adj += 1
            mpred = counter_can + counter_haveto + counter_adj + counter_adv

    return mpred/len(segment)


def advquantif(segment, lang, list_link=None):
    lang_lst = open(f"{list_link}{lang}_adv_quantifiers.lst", 'r').readlines()
    madv_lst = []
    for adv in lang_lst:
        adv = adv.strip()
        madv_lst.append(adv)

    mod_quantif = 0
    for tree in segment:
        for w in tree:
            if lang == 'en':
                if w[2] in madv_lst and w[3] == 'ADV':
                    mod_quantif += 1
            if lang == 'de':
                if w[2] in madv_lst and w[3] == 'ADV':
                    head = get_headwd(w, tree)
                    if head:
                        if head[3] == 'NOUN':
                            continue
                        else:
                            mod_quantif += 1
            if lang == 'ru':
                non_ADVquantif = ['еле', 'очень', 'вшестеро', 'невыразимо', 'излишне',
                                  'еле-еле', 'чуть-чуть', 'едва-едва',
                                  'только', 'капельку', 'чуточку', 'едва']
                if w[2] in madv_lst and w[3] == 'ADV':
                    mod_quantif += 1
                if w[1] in non_ADVquantif:  # based on token, not lemma
                    mod_quantif += 1

    return mod_quantif


def count_dms(trees, lang, list_link=None, dm_type=None):
    lang_lst = open(f"{list_link}{lang}_{dm_type}.lst", 'r').readlines()
    lst = []
    for itm in lang_lst:
        itm = itm.strip()
        lst.append(itm)

    res = 0
    for tree in trees:
        sent = ' '.join(w[1] for w in tree)
        for i in lst:
            i = i.strip()
            try:
                if i[0].isupper():
                    padded0 = i + ' '
                else:
                    padded0 = ' ' + i + ' '
                    # if not surrounded by spaces gets: the best areAS FOR expats for as for
                    # ex: enTHUSiastic for thus

                if padded0 in sent or i.capitalize() + ' ' in sent:
                    # capitalization changes the count for additive from 1822 to 2182
                    # (or to 2120 if padded with spaces)
                    # (furthermore: 1 to 10)
                    res += 1
            except IndexError:
                print('out of range index (blank line in list?: \n', i)
                res = 0

    return res


# Polarity=Neg or PronType=Neg
def polarity(segment, lang):  # "=Neg" in w[4]
    negs = 0
    for tree in segment:
        for w in tree:
            if lang == 'en':
                if w[2] in ['no', 'not', 'neither']:
                    negs += 1
                    '''
                    The UK is n't some offshore tax paradise .
                    America no longer has a Greatest Generation .
                    But almost no major economy scores in the top 10
                    '''
            if lang == 'de':
                if w[2] in ['kein', 'nicht']:
                    negs += 1
                    '''
                    Aber es gibt wohl keinen Patienten , der gegen ..
                    In diesem Fall wirkt das Kalzium allerdings nicht elektrisch , sondern chemisch .
                    '''
            if lang == 'ru':
                if w[2] in ['нет', 'не']:
                    negs += 1
                    '''
                    которых ни у каких претендентов на власть , как правило , нет
                    Никаких сенсаций не будет , не рассчитывайте " , - сказал он журналистам .
                    '''
    return negs


def sents_complexity(trees):
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
    return np.average(clauses_counts), simples / len(trees)


def demdeterm(trees, lang):
    res = 0
    for tree in trees:
        for w in tree:
            if lang == 'en':
                '''
                the list is ranked by frequency
                '''
                if w[6] == 'det' and w[2] in ['this', 'some', 'these', 'that', 'any', 'all',
                                              'every', 'another', 'each', 'those',
                                              'either', 'such']:
                    res += 1

            if lang == 'de':
                if w[6] == 'det' and w[2] in ['dies', 'alle', 'jed', 'einige', 'solch', 'viel',
                                              'ander ', 'jen', 'all', 'irgendwelch',
                                              'dieselbe', 'jeglich', 'daßelbe', 'irgendein', 'diejenigen']:
                    res += 1

            if lang == 'ru':
                '''
                for Russian there is no distinction between эти полномочия и его полномочия
                listed here in order of freq
                '''
                ## muted item: 'свой',
                if w[6] == 'det' and w[2] in ['этот', 'весь', 'тот', 'такой', 'какой',
                                              'каждый', 'любой', 'некоторый', 'какой-то',
                                              'один', 'сей', 'это', 'всякий', 'некий', 'какой-либо',
                                              'какой-нибудь', 'кое-какой']:
                    res += 1
    return res


# ratio of NOUNS+proper names in these functions to the count of these functions
def nouns_to_all_args(trees):
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

    return res/len(trees)


def pasttense(segment):
    count = 0
    for tree in segment:
        for w in tree:
            if 'Tense=Past' in w[4]:
                count += 1

    return count/len(segment)


def finites(segment):
    fins = 0
    for tree in segment:
        for w in tree:
            if 'VerbForm=Fin' in w[4]:
                fins += 1

    return fins/len(segment)


def preps(segment, lang):
    res = 0
    for tree in segment:
        for w in tree:
            if lang == 'en':
                if w[3] == 'ADP':  # and w[2] in ['aboard', 'about', 'above', 'across', 'after', 'against', 'albeit',
                                              # 'along', 'alongside', 'amid', 'amidst', 'among', 'amongst', 'amonst',
                                              # 'around', 'as', 'aside', 'at', 'away', 'back', 'because', 'before',
                                              # 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond',
                                              # 'but', 'by', 'contrary', 'despite', 'down', 'due', 'during', 'en',
                                              # 'except', 'for', 'forthwith', 'forward', 'from', 'in', 'inside',
                                              # 'into', 'like', 'near', 'notwithstanding', 'of', 'off', 'on', 'onto',
                                              # 'opposite', 'out', 'outside', 'outwith', 'over', 'past', 'per', 'post',
                                              # 'round', 'since', 'than', 'that', 'through', 'throughout', 'till', 'to',
                                              # 'together', 'toward', 'towards', 'under', 'underneath', 'unlike', 'until',
                                              # 'up', 'upon', 'versus', 'via', 'vis', 'whereby', 'with', 'within', 'without']:
                    res += 1
            elif lang == 'de':
                if w[3] == 'ADP':  # and w[2] in ['ab', 'abseits', 'abzüglich', 'als', 'am', 'an', 'angesichts', 'anhand',
                                              # 'anläßlich', 'ans', 'anstatt', 'anstelle', 'auf', 'aufgrund', 'aufseiten',
                                              # 'aus', 'außer', 'außerhalb', 'bei', 'beiderseits', 'betreffend', 'bevor',
                                              # 'bezüglich', 'binnen', 'bis', 'dank', 'Dank', 'durch', 'einschließlich',
                                              # 'entgegen', 'entlang', 'für', 'gegen', 'gegenüber', 'gemäß', 'getreu',
                                              # 'hinsichtlich', 'hinter', 'hinterm', 'hinters', 'hinweg', 'im', 'in',
                                              # 'infolge', 'inklusive', 'inmitten', 'innerhalb', 'jenseits', 'mangels',
                                              # 'mit', 'mithilfe', 'Mithilfe', 'mittels', 'nach', 'nachdem', 'neben',
                                              # 'oberhalb', 'ohne', 'per', 'plus', 'pro', 'samt', 'seit', 'seitens',
                                              # 'statt', 'trotz', 'über', 'um', 'ungeachtet', 'unter', 'unterhalb',
                                              # 'unterm', 'von', 'vonseiten', 'vor', 'voran', 'vorbehaltlich', 'während',
                                              # 'wegen', 'wenn', 'wie', 'willen', 'zeit', 'zu', 'zufolge', 'zufolgen',
                                              # 'zugunsten', 'zulasten', 'zur', 'zuzüglich', 'zwecks', 'zwischen']:
                    res += 1
            # elif lang == 'ru':
            #     if w[3] == 'ADP' and w[2] in ['в', 'по', 'за', 'на', 'от', 'с', 'под', 'у', 'из-под', 'из-за', 'до',
            #                                   'к', 'о', 'через', 'из', 'над', 'про', 'после', 'вроде', 'перед', 'между',
            #                                   'насчет', 'около', 'внутрь', 'без', 'кроме', 'для', 'со', 'при', 'сквозь',
            #                                   'вместо', 'вокруг', 'мимо', 'позади', 'возле', 'против', 'согласно', 'вдоль',
            #                                   'во', 'среди', 'напротив', 'благодаря', 'помимо', 'ради',
            #                                   'поверх', 'посреди', 'меж']:
                    res += 1
    return res


# 4-6 July: infinitives, participle, nominatives, passives done for three languages
def infinitives(segment, lang, list_link=None):
    lang_lst = open(f"{list_link}{lang}_modal-adj_predicates.lst", 'r').readlines()
    mpred_lst = []
    for adj in lang_lst:
        adj = adj.strip()
        mpred_lst.append(adj)
    infs = 0
    for tree in segment:
        # General approach for EN/DE: get all Inf with particle, excluding after modal phrases + cases of true bare inf
        # (but not analytical forms or modal verbs)
        # get all to-inf excluding after have-to, going-to and modal phrases + true bare inf (inc. causatives, perception)
        if lang == 'en':
            for w in tree:
                if 'VerbForm=Inf' in w[4]:
                    own_id = w[0]
                    head = get_headwd(w, tree)
                    if choose_kid_by_lempos(w, tree, 'to', 'PART'):  # cases of to-inf
                        if head:
                            if head[2] == 'have':
                                objective1 = choose_kid_by_posrel(head, tree, 'NOUN', 'obj')
                                #                         print(' '.join(w[1] for w in tree))
                                if objective1:
                                    if objective1 < own_id:
                                        '''
                                        while excluding true have to V-Inf, retain causative constructions: to have the euro replace the dollar
                                        and Complex Objects:
                                            you have so many places to eat
                                            have a duty to protect the weakest in society
                                        '''
                                        infs += 1
                                else:
                                    '''
                                    People who have to fake emotions at work , burn out faster
                                    '''
                                    continue
                            elif head[2] == 'go' or head[2] in mpred_lst:
                                '''
                                ex. we 're likely to see more exciting findings
                                It 's dark in the cabin ; someone is going to step on your face !
                                They 're going to have to .
                                Bush is not going to withdraw the troops .
                                '''
                                continue
                            else:
                                '''
                                classic to-inf with head:
                                I think if we decided to make French the company 's global language we would have had a revolt on our hands .
                                They do n't want to lose face by using the wrong word
                                '''
                                # print(w[1].upper())
                                # print(' '.join(w[1] for w in tree))
                                infs += 1
                        else:
                            '''
                            classic to-inf without head:
                            When in doubt , it 's best to ask .
                            Or , to put it another way , maybe it 's selective laziness .
                            How to explain this cleanliness and punctuality ?
                            This is not to write the whole film off .
                            How are we to trust them ?
                            '''
                            infs += 1
                    # BARE Inf cases:
                    elif w[3] != 'AUX' and head and head[2] in ['help', 'make', 'bid', 'let', 'see', 'hear',
                                                                'watch', 'dare', 'feel', 'have']:
                        '''
                        the first bit excludes cases like:
                        we would have (Inf, dependent on have) had a revolt on our hands .
    
                        gets 528 good cases like:
                        you could see their culture come (Inf) to life .
                        to do anything to have Russia pay (Inf) a price for its aggressive behavior
                        Let 's start across the Atlantic
                        '''
                        infs += 1

        if lang == 'de':
            '''
            German rules:
            all zu-Inf, excluding after modal phrases: Es ist notwendig zu sagen; this supposedly gets all true Infs
            all bare inf dependent on
                hören, sehen, spüren
                lassen, gehen, bleiben, helfen, lehren
            The alternative approach of filtering analytical forms and modals
            and ALL Fins mistagged as Inf and misparsed sentences returns:
            '''
            for w in tree:
                if 'VerbForm=Inf' in w[4]:
                    head = get_headwd(w, tree)
                    '''
                    we have to exclude true negatives, mostly finite forms.
                    This does not help with false positives
                    '''
                    if choose_kid_by_lempos(w, tree, 'zu', 'PART'):  # cases of zu-inf
                        if head and head[2] in mpred_lst:
                            '''
                            exclude a few unwanted zu-Inf after modal predicates
                                ex.: Alle Bordinstrumente werden notwendig sein, um die Antworten auf diese Fragen zu finden .
                                ex. In jedem Fall ist es notwendig , vor der Nachrichtenübermittlung einen Schlüssel zu vereinbaren .
                            '''
                            continue
                        else:
                            '''
                            Diese Energie ist erforderlich , um die Molekülpaare wieder zu trennen .
                            Derzeit ist man dabei , die verschiedenen Varianten genauer zu untersuchen , und bisher scheinen sie größtenteils die gleichen Funktionen zu haben .
                            '''
                            infs += 1

                    elif head and head[2] in ['hören', 'sehen', 'spüren', 'lassen', 'gehen', 'bleiben', 'helfen', 'lehren']:
                        '''
                        Es bleibt zu erforschen , wie nützlich oder wie schädlich es sich auf die lebende Zelle auswirken kann .
                        '''
                        infs += 1

        if lang == 'ru':
            for w in tree:
                # exclude: "пока Россия будет проводить агрессивную политику", "отношения будут ухудшаться"
                if 'VerbForm=Inf' in w[4]:
                    head = get_headwd(w, tree)
                    if head and head[2] in mpred_lst:
                        continue
                    if has_auxkid_by_lem(w, tree, 'быть'):
                        # print(' '.join(w[1] for w in tree))
                        infs += 1

    return infs


## this is the enhanced function that expects a stoplist and a list of approved V>N converts
def nominals(trees, lang, list_link=None):
    stoplst = [i.strip() for i in open(f"{list_link}{lang}_deverbals_stop.lst", 'r').readlines()]
    converted = [i.strip() for i in open(f"{list_link}{lang}_converts.lst", 'r').readlines()]

    res = 0

    for tree in trees:
        for w in tree:
            if lang == 'en':
                if w[2] not in stoplst:
                    if 'NOUN' in w[3] and (w[2].endswith('ment') or w[2].endswith('tion')):
                        res += 1

                if 'NOUN' in w[3] and w[2] in converted:  # this is a filtered golist from our data
                    if w[1].endswith('ing'):
                        continue
                    kids_pos = get_kids_pos(w, tree)
                    if 'DET' not in kids_pos and 'ADJ' not in kids_pos and 'ADJ' not in kids_pos:
                        if 'Number=Sing' in w[4]:
                            continue
                        else:
                            res += 1
                    else:
                        res += 1

            if lang == 'de':
                if w[2] not in stoplst:
                    if 'NOUN' in w[3] and (w[2].endswith('ung') or w[2].endswith('tion')):
                        res += 1
                # this is a filtered golist from our data to get deverbals formed after diverse word formation patterns
                if 'NOUN' in w[3] and w[2] in converted:
                    res += 1

            if lang == 'ru':
                if w[2] not in stoplst and 'NOUN' in w[3] \
                        and (w[2].endswith('тие') or w[2].endswith('ение')
                             or w[2].endswith('ание') or w[2].endswith('ство')
                             or w[2].endswith('ция') or w[2].endswith('ота')) \
                        and 'Number=Plur' not in w[4]:
                    res += 1
    return res


def relativ(segment, lang):
    tot_rel = 0
    # matches = []  # this can be used to produce the freq dict of the relative PRON that introduce clauses
    # pied = 0
    # correlatives = 0
    for tree in segment:
        # relative: дом, который построил Джек; zwei Regionen , in denen wir versucht haben
        # correlative: то, что построил Джек ; Он настаивал на том, что считал важным.
        # pied: speech in which he made this argument, technology for which Sony could take credit
        try:
            if tree[-1][2] != '?':  #  and tree[-2][2] != '?'
                if lang == 'en':
                    for w in tree:
                        if w[2] in ['which', 'that', 'whose', 'whom', 'what', 'who'] and w[3] == 'PRON':
                            head = get_headwd(w, tree)
                            if head and head[6] == 'acl:relcl':
                                tot_rel += 1
                                # matches.append(f'\n{w[2].upper()}: {" ".join(w[1] for w in tree)}')
                elif lang == 'de':
                    for w in tree:
                        '''
                        entwickeln als ein fruchtbarer Streit darüber (PRON, conj),
                        welche (PRON, det) Grundannahmen der Analyse...
                        '''
                        if (w[2] in ['der', 'welch', 'was', 'wer'] and w[3] == 'PRON') \
                                or ('wo' in w[2] and 'PronType=Int,Rel' in w[4]):
                            correlat_id = w[0]
                            for ww in tree:
                                # allow up to 3 words between the comma and the Pron
                                if ww[1] == ',' and 0 < (correlat_id - ww[0]) <= 3:
                                    tot_rel += 1
                                    # matches.append(f'\n{w[2].upper()}: {" ".join(w[1] for w in tree)}')
                                    '''
                                    general count for all types of relative clauses (non-existant for DE in UD)
                                        zwei Regionen , in denen wir versucht haben
                                        Oberflächenwasser , das sich in einem tiefen Loch ansammelt
                                        viele andere Drogen , die abhängig machen
                                        konnten wir sehr viel mehr Erfahrungswerte dahingehend sammeln,
                                        welche Lösung für ..
                                    '''
        except IndexError:
            print(tree)
            print('relative clauses extraction ERROR')
            exit()

    return tot_rel/len(segment)


def selfsich(segment, lang):
    count = 0
    for tree in segment:
        for w in tree:
            if lang == 'en':
                if w[1].endswith('self'):
                    count += 1
            elif lang == 'de':
                if w[1] == 'sich':
                    count += 1
    return count


# EN<->DE WO functions
# verb_obj_order to all obj in the segment?
def verb_n_obj(segment):
    count = 0
    all_obj = 0
    for tree in segment:
        for w in tree:
            if w[6] == 'obj' and w[3] == 'NOUN':
                head = get_headwd(w, tree)
                own_id = w[0]
                if head and head[3] == 'VERB':
                    head_id = head[0]
                    all_obj += 1
                else:
                    continue
            else:
                continue

            if int(own_id) > int(head_id):  # VO order
                count += 1
    try:
        vo_order_ratio = count / all_obj
    except ZeroDivisionError:
        vo_order_ratio = 0

    return vo_order_ratio/len(segment)


def main_verb_nsubj_noun(segment):  # inversion in main clause
    count = 0
    all_nsubj = 0
    # matches = []
    for tree in segment:
        for w in tree:
            if w[6] == 'root':
                aux_id_conllu = choose_kid_by_featrel(w, tree, 'VerbForm=Fin', 'aux')
                if aux_id_conllu:
                    fin_id_conllu = aux_id_conllu
                else:
                    fin_id_conllu = w[0]
                # from conllu ID column, if needs to be used for list indexing, do -1
                nsubj_id_conllu = choose_kid_by_featrel(w, tree, 'Case=Nom', 'nsubj')

                # print(nsubj_id, fin_id)
                if nsubj_id_conllu:
                    all_nsubj += 1
                    try:
                        if tree[nsubj_id_conllu - 1][3] == 'NOUN':
                            # Insofern bin ich #2 > fin #1
                            if int(nsubj_id_conllu) > int(fin_id_conllu):
                                count += 1
                    except IndexError:
                        print(segment)
                        print()
                        print(tree)
                        print(len(tree), nsubj_id_conllu - 1)
                        exit()
            else:
                continue
    try:
        vs_order_ratio = count / all_nsubj
    except ZeroDivisionError:
        vs_order_ratio = 0

    return vs_order_ratio/len(segment)


def obl_obj(segment):
    obl_obj_order = 0
    all_pairs = 0
    for tree in segment:
        for w in tree:
            if 'VerbForm=Fin' in w[4]:
                oblique_ids = []
                obj_ids = []
                co_dependants = get_kids(w, tree)
                for child in co_dependants:
                    if 'obl' in child[6]:
                        this_obl_id = child[0]
                        oblique_ids.append(this_obl_id)
                    elif child[6] == 'obj':
                        this_obj_id = child[0]
                        obj_ids.append(this_obj_id)

                combinations = list(product(oblique_ids, obj_ids))
                all_pairs += len(combinations)
                for pair in combinations:
                    if pair[0] > pair[1]:
                        obl_obj_order += 1
    try:
        obl_obj_ratio = obl_obj_order/all_pairs
    except ZeroDivisionError:
        obl_obj_ratio = 0

    return obl_obj_ratio/len(segment)


def advmod_verb(segment):
    all_advmod = 0
    advmod_verb_order = 0

    for tree in segment:
        for w in tree:
            if w[3] == 'VERB':
                advmod_ids = []
                kids = get_kids(w, tree)
                for kid in kids:
                    # specify kids rel
                    if 'advmod' in kid[6]:
                        all_advmod += 1
                        kid_id = kid[0]
                        advmod_ids.append(kid_id)
                        if kid_id > w[0]:
                            advmod_verb_order += 1
    try:
        advmod_verb_ratio = advmod_verb_order / all_advmod
    except ZeroDivisionError:
        advmod_verb_ratio = 0

    return advmod_verb_ratio/len(segment)


# how often a non-nsubj is topicalised in affirmative sentences?
# non-nsubj BoS
# it is expected that the vorfield in non-translated German will have more topicalised non-subjects
def topical_vorfeld(segment):
    count = 0
    # matches = []
    for tree in segment:
        try:
            if tree[-1][2] != '?':  #  and tree[-2][2] != '?'
                for w in tree:
                    if w[0] == 1:
                        if w[1] == 'Ja' or w[1] == 'Yes':
                            continue
                        if w[6] == 'nsubj':
                            continue
                        else:
                            if w[6] == 'root':
                                count += 1
                            else:
                                head = get_headwd(w, tree)
                                # print(f'{w[2].upper()}, {w[0]}: {" ".join(w[1] for w in tree)}')
                                # print()
                                try:
                                    if head[6] == 'nsubj':  # skip nounal dependents
                                        continue
                                    else:
                                        count += 1
                                except TypeError:
                                    print(f'{w[2].upper()}, {w[0]}: {" ".join(w[1] for w in tree)}')
                                # matches.append(f'{w[2].upper()}: {" ".join(w[1] for w in tree)}')
                    else:
                        continue
        except IndexError:
            count += 0
    return count/len(segment)


# fewer pronominal prep phrases in EoS in English translations (about him .) NOPE! 12K (org) vs 20K (tra)
# more PP in EoS in translated German, z.B ``... , was Sie früher beschrieben haben <als gemeinsame Werte>, die wir haben.''
def expected_nachfeld(segment, lang):
    count = 0
    # matches = []
    for tree in segment:
        for w in tree:
            if lang == 'en':
                # sentence ending in obl, if head has obl-kids
                if 'VerbForm=Fin' in w[4]:
                    kids = get_kids(w, tree)
                    if any(kid[6] in ['obl', 'obl:tmod'] for kid in kids):
                        last_wd_id = len(tree)
                        if tree[last_wd_id-1][6] != 'punct':
                            # non-tokenised punct: ex. EU.
                            if 'obl' in tree[last_wd_id-1][6]:
                                count += 1
                                # matches.append(f'{tree[last_wd_id-1][2]} {w[2].upper()}: {" ".join(w[1] for w in tree)}')
                        else:
                            if 'obl' in tree[last_wd_id-2][6]:
                                count += 1
                                # matches.append(f'{tree[last_wd_id-2][2]} {w[2].upper()}: {" ".join(w[1] for w in tree)}')
            if lang == 'de':
                # subordinate clauses ending in finite vform
                if 'VerbForm=Fin' in w[4]:
                    if w[6] == 'root':
                        continue
                    elif w[6] == 'aux':
                        head = get_headwd(w, tree)
                        if head[6] == 'root':
                            continue
                        else:
                            # head of the subordinate clause
                            # and it is NOT followed by a dot, question or !
                            # you don't have to +1 because of zero-based indexing in Python
                            try:
                                if tree[int(w[0])][1] in ['.', '!', '?', ',']:
                                    # non-root finites in EoS

                                    count += 1
                                    # matches.append(f'{w[2].upper()}: {" ".join(w[1] for w in tree)}')
                            except IndexError:  # PUNCT missing
                                count += 1
                                # matches.append(f'{w[2].upper()}: {" ".join(w[1] for w in tree)}')
                else:
                    continue
    return count/len(segment)


def mean_sent_length(trees):
    sent_lengths_lst = []
    for tree in trees:
        sent_lengths_lst.append(len(tree))

    return sent_lengths_lst, np.average(sent_lengths_lst)


# this is a help function to produce distributions of UD dependencies in a sentence
def relation_counts(tree, i_relations):
    sent_relations = [w[6] for w in tree]
    counts = {rel: sent_relations.count(rel) for rel in i_relations}

    return counts


def advmod_no_negation(trees):
    corrected_count = 0
    for tree in trees:
        for w in tree:
            if w[6] == 'advmod':
                if '=Neg' not in w[4]:
                    corrected_count += 1

    corrected_count = corrected_count / len(trees)

    return corrected_count


def ud_freqs(trees, relations=None):
    relations_d = {rel: [] for rel in relations}
    for tree in trees:
        rel_distribution = relation_counts(tree, relations_d)
        # this returns probabilities for the picked dependencies in this sent
        for rel in relations_d.keys():  # reusing the empty dict
            # this collects counts from individual sents to the global empty dict as lists
            relations_d[rel].append(rel_distribution[rel])
            # values in this dict are lists of counts in each sentence; need to average them

    dict_out = {}
    for rel in relations_d.keys():
        dict_out[rel] = np.sum(relations_d[rel]) / len(trees)  # counts of this rel averaged to the number of sentences in the segment

    return dict_out


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


# exclude PUNCT, SYM, X from tokens
def content_ttr_and_density(segment):
    # some segments can have more than one tree!
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

    return np.average(across_segment_trees_ttr), np.average(across_segment_trees_dens), counter


# graph-based feature
# this function tests connectedness of the graph; it helps to skip malformed sentences
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

    return sentence_graph  #, bad_trees


# linearising a hierarchical idea, depth of the tree, complexity of the idea
def speakdiff(segment):
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
    mhd = np.nanmean(across_segment_trees)

    return mhd, errors


# this is a slower function which allows to print graphs for sentences (to be used in overall_freqs)
def speakdiff_visuals(segment):
    # some segments can have for than one tree!
    across_segment_trees = []
    tot_errors = 0
    my_graphs = []
    for tree in segment:
        # call the above function to get the graph and counts for disintegrated trees
        # (syntactic analysis errors of assigning dependents to punctuation)
        graph, bad_graphs = test_sanity(tree)
        my_graphs.append(graph)

        parts = graph.components(mode=WEAK)
        mhd0 = 0
        if len(parts) == 1:
            nodes = [word[1] for word in tree if word[6] != 'punct' and word[6] != 'root']
            all_hds = []  # or a counter = 0?
            for node in nodes:
                # Dijkstra's algorithm is an algorithm for finding the shortest paths between nodes in a weighted graph, which may represent, for example, road networks.
                # graph.shortest_paths_dijkstra is an alias for graph.shortest_paths, which is in turn depricated and replaced with graph.distances()
                try:
                    hd = graph.distances('ROOT', node, mode=ALL)

                except ValueError:
                    print('ERROR')
                    print(tree)
                    exit()

                all_hds.append(hd[0][0])
            if all_hds:
                mhd0 = np.average(all_hds)
            across_segment_trees.append(mhd0)
        tot_errors += bad_graphs
    mhd = np.average(across_segment_trees)

    return mhd, my_graphs


# calculate comprehension difficulty=mean dependency distance(MDD) as “the distance between words and their parents,
# measured in terms of intervening words.” (Hudson 1995 : 16)
def readerdiff(segment):
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
    mdd = np.nanmean(across_segment_trees)

    # I still get NaNs in 96 short segments! I will just drop these segments! in classifier and feature analysis
    if not mdd:
        mdd = 1
    # average MDD for the segment, including 1 for Bravo !, Warum ? two-token segments with only a root and a punct
    return mdd
