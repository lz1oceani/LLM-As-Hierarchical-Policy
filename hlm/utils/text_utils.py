from pathlib import Path
from nltk import ngrams
from IPython import embed
from numbers import Number
import string, re


COMMON_RE_PATTENS = {
    "the": "(?:the|this)",
    "can't": "can(?:not|n't)",
    "don't": "do(?: not|n't)",
    "find": "find|determine|provide",
}


def load_text(filename):
    filename = str(filename)
    with open(filename) as f:
        txt = f.readlines()
    return "".join(txt)


def filter_stripped_lines(lines):
    return [_.strip() for _ in lines if len(_.strip()) > 0]


def unique_texts(texts):
    return list(dict.fromkeys(texts))


def eval_text(text):
    if not isinstance(text, str):
        return text
    try:
        return eval(text)
    except Exception as e:
        return None


def lstrip_string(text, prefix):
    return text[len(prefix) :] if text.startswith(prefix) else text


def rstrip_string(text, suffix):
    return text[: -len(suffix)] if text.endswith(suffix) else text


def strip_string(string, prefix, suffix):
    return lstrip_string(rstrip_string(string, suffix), prefix)


def squeeze_space(text):
    return re.sub(" +", " ", text)


def squeeze_newline(text):
    return re.sub("[\n]+", "\n", text)


def clean_text(text):
    # lower cased
    text = text.lower()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    text = re.sub(r"\s+", " ", text.strip())
    return text


def split_text(text, max_len_row):
    if len(text) > 4 * max_len_row:
        line = text.split("\n")
        ret = []
        now = ""
        for _ in line:
            if len(now) > 0:
                now += "\n"
            now += _
            if len(now) > max_len_row:
                ret.append(now)
                now = ""
        return ret
    else:
        return [text]


def all_matched_pos(pattern, text):
    if isinstance(pattern, (list, tuple)):
        pattern = "(" + "|".join(pattern) + ")"
    return sorted([match.start() for match in re.finditer(pattern, text)])
