import re
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemmatisation_dict = {
    'wouldve*': ["would've", "wouldve", 'woulda'],
    'already*': ['already', 'allready'],
    'please*': ['please*', 'pleaseee', 'pleaseeee'],
    'squid*': ['squid', 'squid1', 'squid2', 'squid3', 'squid4'],
    'true*': ['true', 'truers', 'truee', 'trueee', 'trueeeee'],
    'thats*': ['thats', "that's"],
    'thatd*': ['thatd', "that'd"],
    'finish*': ['finish', "finnish"],
    ':)*': [':)', ':))', ':)))', ':))))', ':)))))', ':))))))'],
    ':(*': [':(', ':((', ':(((', ':((((', ':(((((', ':(((((('],
    'good*': ['good', 'goood', 'gooood', 'goooood', 'gooooood', 'goooooood', ],
    'shit*': ['shit', 'shiit', 'shiiit', 'shiiiit', 'shiiiiit', 'shiiiiiit', 'shitt', 'shittt', 'shitttt', 'shittttt',
              'shitttttt', ]
}


def haha_test(strg, search=re.compile(r'\b[bah]+\b').search):
    return (search(strg))


def hehe_test(strg, search=re.compile(r'\b[he]+\b').search):
    return (search(strg))


def go_test(strg, search=re.compile(r'\b[go]+\b').search):
    return (search(strg))


def no_test(strg, search=re.compile(r'\b[no]+\b').search):
    return (search(strg))


def way_test(strg, search=re.compile(r'\b[way]+\b').search):
    return (search(strg))


def why_test(strg, search=re.compile(r'\b[why]+\b').search):
    return (search(strg))


def what_test(strg, search=re.compile(r'\b(wha)t+\b').search):
    return (search(strg))


def hello_test(strg, search=re.compile(r'\b(hell)o+\b').search):
    return (search(strg))


def boom_test(strg, search=re.compile(r'\b[bom]+\b').search):
    return (search(strg))


def yay_test(strg, search=re.compile(r'\b[yay]+\b').search):
    return (search(strg))


def lol_test(strg, search=re.compile(r'\b[lo]+\b').search):
    return (search(strg))


def yo_test(strg, search=re.compile(r'\b[yo]+\b').search):
    return (search(strg))


def so_test(strg, search=re.compile(r'\b[so]+\b').search):
    return (search(strg))


def oh_test(strg, search=re.compile(r'\b[oh]+\b').search):
    return (search(strg))


def ey_test(strg, search=re.compile(r'\b(ey)y+\b').search):
    return (search(strg))


def hey_test(strg, search=re.compile(r'\b(hey)y+\b').search):
    return (search(strg))


def fuck_test(strg, search=re.compile(r'\b(fuc)k+\b').search):
    return (search(strg))


def wow_test(strg, search=re.compile(r'\b[wo]+\b').search):
    return (search(strg))


def hmm_test(strg, search=re.compile(r'\b[hm]+\b').search):
    return (search(strg))


def ewww_test(strg, search=re.compile(r'\b[ew]+\b').search):
    return (search(strg))


def yeah_test(strg, search=re.compile(r'\b(yea)h+\b').search):
    return (search(strg))


def baby_test(strg, search=re.compile(r'\b(bab)y+\b').search):
    return (search(strg))


def nah_test(strg, search=re.compile(r'\b(na)h+\b').search):
    return (search(strg))


def uhh_test(strg, search=re.compile(r'\b(uh)h+\b').search):
    return (search(strg))


def damn_test(strg, search=re.compile(r'\b(damn)n+\b').search):
    return (search(strg))


def omg_test(strg, search=re.compile(r'\b(om)g+\b').search):
    return (search(strg))


def lemmatise(word):
    # test for repeated chars e.g. hahahah or goooooo or lololol
    if haha_test(word): return "haha*"
    if hehe_test(word): return "hehe*"
    if go_test(word): return "go*"
    if no_test(word): return "no*"
    if way_test(word): return "way*"
    if why_test(word): return "why*"
    if what_test(word): return "what*"
    if boom_test(word): return "boom*"
    if yay_test(word): return "yay*"
    if lol_test(word): return "lol*"
    if so_test(word): return "so*"
    if yo_test(word): return "yo*"
    if oh_test(word): return "oh*"
    if hello_test(word): return "hello*"
    if ey_test(word): return "ey*"
    if hey_test(word): return "hey*"
    if wow_test(word): return "wow*"
    if fuck_test(word): return "fuck*"
    if hmm_test(word): return "hmm*"
    if ewww_test(word): return "ewww*"
    if yeah_test(word): return "yeah*"
    if baby_test(word): return "baby*"
    if nah_test(word): return "nah*"
    if uhh_test(word): return "uhh*"
    if damn_test(word): return "damn*"
    if omg_test(word): return "omg*"

    # test for popular emotes which often have many many different versions
    if 'kappa' in word: return 'kappa*'
    if 'lul' in word and word != 'lulw': return 'lul*'
    if 'pog' in word: return 'pog*'
    if 'pepe' in word: return 'pepe*'
    if 'jebait' in word: return 'jebait*'
    if 'lmfao' in word or 'lmao' in word: return 'lm(f)ao*'

    # test for a small amount of common mispelling/varients
    for lemma in lemmatisation_dict:
        if word in lemmatisation_dict[lemma]:
            return lemma
    # last check is to use the NLTK lemmatise
    return lemmatizer.lemmatize(word)
