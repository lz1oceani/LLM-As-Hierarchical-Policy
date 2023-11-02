import numpy as np, time, re, signal, math, os, warnings
from numbers import Number
from sympy import Symbol, Eq, simplify, solve
from sympy.parsing.latex import parse_latex
from math import *
from .text_utils import filter_stripped_lines, unique_texts, all_matched_pos

warnings.simplefilter("ignore", SyntaxWarning)


class TimeoutException(Exception):
    pass


ANSWER_SPLIT_PATTERNS = [
    "answer is:?",
    "answer:",
    "answer to (?:the|this) question is",
    # last letters
    "concatenated letters are",
    "concatenate the letters -",
    "The answer of ",
]

NEGATIVE_PATTERNS = [
    "is not needed to answer the question",
]

ANSWER_PREFIX = [
    "answer: ",
    "Therefore, there will be ",
    "Therefore, \w+ have ",
    "Therefore, \w+ and \w+ have ",
    "Therefore, \w+ has ",
    "Therefore,(.*?)is ",
    "there (are|is) ",
    "answer to(.*?)is ",
    "answer to(.*?)will be ",
    "answer to(.*?)would be ",
    "answer to(.*?)becomes ",
    "Therefore,(.*?)will be ",
    "Therefore,(.*?)would be ",
    "Therefore,(.*?)cost ",
    "Therefore,(.*?)costs ",
    "Therefore,(.*?)a total of ",
    "There will be ",
    "Therefore, ",
    "[A-Z]\w+ will have ",
    "[A-Z]\w+ have ",
    "[A-Z]\w+ has ",
    "\w+ still has ",
    "^[A-Z]\w+ \w+ ",
    " is ",
]

NUMBER_FIX_MAP = {
    " zero ": " 0 ",
    " no ": " 0 ",
    " a ": " 1 ",
    " one ": " 1 ",
    " two ": " 2 ",
    " three ": " 3 ",
    " four ": " 4 ",
    " five ": " 5 ",
    " six ": " 6 ",
    " seven ": " 7 ",
    " eight ": " 8 ",
    " nine ": " 9 ",
    " ten ": " 10 ",
    "\u2013": "-",
    "hundred": "*100",
    "thousand": "*1000",
    "million": "*(10**6)",
    "billion": "*(10**9)",
    "trillion": "*(10**12)",
}

NO_ANSWER_TEMPLATE = [
    "we can(not|n't)(?:.*)answer(?:.*)(?:the|this) question",
    "we do( not|n't)(?:.*)answer(?:.*)(?:the|this) question",
    "we can(not|n't) (determine|find)",
    "there (are|is) no (solutions|answer|answers)" "the answer(.*?)is unknown",
    "Finally,(.*?)to find the answer.$",
]


def get_answer_type(dataset_name):
    dataset_name = dataset_name.lower()
    num_names = ["gsm"]
    latex_names = ["math"]
    check = lambda name, keywords: any([_ in name for _ in keywords])
    
    if check(dataset_name, num_names):
        return "number"
    elif check(dataset_name, latex_names):
        return "latex"
    else:
        raise NotImplementedError


def get_re_templates(answer_type, choices=None):
    templates = {
        "number": ["(-?\(\d+\/\d+\)\/\d+|-?\d+\/\d+)", "(-?\d[\d,\. ]*)"],
        "latex": [],
    }
    return templates.get(answer_type, None)


def extract_all_numbers(text):
    templates = get_re_templates("number", None)
    for template in templates:
        nums = re.findall(template, text)
        if len(nums) > 0:
            return nums
    return []


def extract_all_expressions(text):
    if "$" in text:
        text = text.replace("$\$", "$")

        num = text.count("$")
        if num % 2 == 0:
            return list(re.findall(r"\$([^\$]*)\$", text))
        else:
            return []

    pairs = [[r"\[", r"\]"], [r"\\begin\{align\}", r"\\end\{align\}"], [r"\\begin\{align\*\}", r"\\end\{align\*\}"]]
    ret = []
    for start, end in pairs:
        sign = re.search(start, text) is not None and re.search(end, text) is not None
        if sign:
            ret += re.findall(rf"{start}([^{start}{end}]*){end}", text)
    return ret


def extract_text_answer(text, answer_type=None, final_answer=None):
    from .math_answer_utils import normalize_numbers, unwrap_latex_env, clean_up_latex_answer
    from .metric_utils import compare_items

    templates = get_re_templates(answer_type, None)
    split_words = ["Therefore", ", so", "is"]
    def remove_equal(nums):
        if answer_type == "number" or "=" not in final_answer:
            tmp = []
            for num in nums:
                if "=" in num:
                    num = num.split("=")[-1].strip()
                if "\equiv" in num:
                    num = re.split(r"\\equiv", num)[-1].strip()
                tmp.append(num)
            nums = tmp
        return nums

    if "\\boxed" in text:
        text = unwrap_latex_env(text, "boxed", is_single=True)
        text = unwrap_latex_env(text, "textsf", is_single=False)
        return remove_equal(clean_up_latex_answer(text))[0]
    check = lambda _: ("$" in _ or "\[" in _) and answer_type == "latex"
    clean_up_fn = clean_up_latex_answer if answer_type == "latex" else normalize_numbers

    nums = []
    for pos in all_matched_pos(split_words, text)[::-1]:
        extract_fn = extract_all_expressions if check(text[pos:]) else extract_all_numbers
        nums = extract_fn(text[pos:])
        if len(nums) > 0:
            break
    if len(nums) == 0:
        extract_fn = extract_all_expressions if check(text) else extract_all_numbers
        nums = extract_fn(text)
    if len(nums) >= 1:
        nums = remove_equal(nums)
        for num in nums:
            if compare_items(num, final_answer, answer_type if answer_type == "number" else "text"):  # About %1 in GSM
                return clean_up_fn(num)
        ret = nums[0]
        return clean_up_fn(ret)
    else:
        return None


def extract_answer_from_sentence(sentence):
    ret = sentence
    for pattern in ANSWER_SPLIT_PATTERNS:
        indices = list(re.finditer(pattern, sentence, flags=re.IGNORECASE))
        if len(indices) > 0:
            tmp = sentence[indices[-1].start() :]
            if len(tmp) < len(ret):
                ret = tmp
    return ret


def extract_answers(responses, answer_type=None, max_num_lines=3, final_answer=None, **kwargs):
    if isinstance(responses, list):
        return [extract_answers(_, answer_type, max_num_lines, final_answer, **kwargs) for _ in responses]
   
    sentences = re.split(r"\n", responses)  # Split text by new line or latex expression \]
    sentences = [re.sub(r"^#?\d+\. ", "", _) for _ in sentences]  # remove starting #1, #2, ...
    sentences = [_ for _ in sentences if not _.strip("#").lower().startswith("reference:")]  # remove reference lines in Natural Program
    sentences = filter_stripped_lines(sentences)

    if len(sentences) == 0:
        print(responses)
        print(sentences)
        exit(0)

    for template in NO_ANSWER_TEMPLATE:
        if len(re.findall(template, sentences[-1], flags=re.IGNORECASE)) > 0:
            return "No answer!"

    if (sentences[-1].count("$") == 1 or "\[" in sentences[-1] and "\]" not in sentences[-1]) and answer_type == "latex":
        sentences = sentences[:-1]  # Incorrect latex expression
    sentences = sentences[-max_num_lines:]
    contain_answer = np.array([re.search("(" + "|".join(ANSWER_SPLIT_PATTERNS) + ")", _) is not None for _ in sentences])
    contain_neg = np.array([re.search("(" + "|".join(NEGATIVE_PATTERNS) + ")", _) is not None for _ in sentences])
    try:
        contain_keywords = np.any(np.logical_and(contain_answer, ~contain_neg))
    except:
        return "No answer!"
    last_indices = len(sentences)
    answer = None

    for i in reversed(range(len(sentences))):
        if contain_neg[i] or (contain_keywords and not contain_answer[i]):
            continue
        sentence = "\n".join(sentences[i:last_indices])
        for source, target in NUMBER_FIX_MAP.items():
            sentence = sentence.replace(source, target)

        answer_sentence = extract_answer_from_sentence(sentence)
        answer = extract_text_answer(answer_sentence, answer_type, final_answer)
        if isinstance(answer, str) and (len(answer) == 0 or len(answer) >= 256):
            answer = None
        last_indices = i
        if answer is not None:
            break
    return "No answer!" if answer is None else answer.strip()
