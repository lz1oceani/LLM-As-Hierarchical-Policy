import numpy as np, re, time, signal
from numbers import Number
from IPython import embed
from sympy.parsing.latex import parse_latex
from sympy import oo, Interval, ImmutableDenseMatrix, FiniteSet, Complement
from sympy.geometry.point import Point
from math import *
from .text_utils import eval_text, squeeze_space


def normalize_numbers(text):
    text = re.sub(r"(\d+)\s+(\d+\/\d+)", r"\1 + \2", text)  # 1 1/2 -> 1+1/2
    text = re.sub(r"(\d+)\s*\\frac", r"\1 + \\frac", text)  # 1 \frac{1}{2} -> 1 + \frac{1}{2}
    # text = re.sub(r"(\d+)\/(\d+)", r"\\frac{\1}{\2}", text)  # 1/2 -> \frac{1}{2}
    while True:
        try:
            place = next(re.finditer("(\d),(\d\d\d)($|\d)", text))
        except:
            break
        # print(text, place, text[place.start() : place.end(2)], place.group(1) + place.group(2))
        pattern, number = [text[place.start(1) : place.end(2)], place.group(1) + place.group(2)]
        text = text.replace(pattern, number)

    # text = re.sub(r"(\d*)\s+\(\d*\/\d*\)", r"\1*\2", text)  # 1 (1/2) -> 1 * 1/2
    text = squeeze_space(text)
    if re.fullmatch("([\d|\.|\{|\s]*)", text) is not None:
        text = text.replace(" ", "")  # Only remove the sapce in pure number
    text = re.sub(r"(^|\s|\{)(\.\d+)", r"\g<1>0\2", text)  # .123 -> 0.123
    text = re.sub(r"\.0*$", "", text)  # 1.000 -> 1
    text = text.strip().rstrip(".,").strip()  # 1. -> 1
    return text


def unwrap_latex_env(text, env_name="boxed", is_single=False):
    if env_name not in text:
        return text if is_single else [text]
    env_template = r"\\" + env_name + r"(?:\{|\\{)"

    ret = []
    for match in re.finditer(env_template, text):
        idx = match.start()
        count = 0
        while idx < match.end():
            count += text[idx] == "{"
            idx += 1
        while count > 0 and idx < len(text):
            count += text[idx] == "{"
            count -= text[idx] == "}"
            idx += 1
        ret.append(text[match.end() : idx - 1].rstrip("\\"))
    if is_single:
        return ret[0] if len(ret) > 0 else text
    else:
        return ret if len(ret) > 0 else [text]


def remove_starting_text(text):
    text = text.strip()
    while True:
        sign = False
        for name in ["text", "textbf", "boxed"]:
            if re.search(r"^\\?\\text\s*\{", text) is not None:
                text = unwrap_latex_env(text, env_name=name, is_single=True)
                sign = True
        if not sign:
            break
    return text


def _fix_fracs(string):
    if string.startswith("\\frac"):
        cnt = 0
        for i, c in enumerate(string):
            if c == "{":
                cnt += 1
            elif c == "}":
                cnt -= 1
            if cnt < 0:
                break
        if cnt > 0:
            string += "}" * cnt
    return string


def clean_up_latex_answer(answer):
    # This function is only used for answer that extracted from latex environment

    if isinstance(answer, list):
        return [clean_up_latex_answer(_) for _ in answer]

    answer
    answer = remove_starting_text(answer)
    answer = _fix_fracs(answer)

    for template in [
        "\n",
        r"\\text\{(.*?)\}",
        r"\\left",
        r"\\right",
        r"\\\$",
        r"\\\%",
        r"\$",
        r"\\(?:\(|\[)",
        r"\\(?:\)|\])",
    ]:
        answer = re.sub(template, "", answer)

    answer = re.sub(r"\\dfrac", r"\\frac", answer)  # \dfrac -> \frac
    answer = re.sub(r"\\tfrac", r"\\frac", answer)  # \dfrac -> \frac
    answer = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", answer)  # \frac12 -> \frac{1}{2}
    answer = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", answer)  # \frac1{ -> \frac{1}{}
    # answer = re.sub(r"\\frac{}{", r"\\frac{1}{", answer)  # \frac{}{2} -> \frac{1}{2}

    answer = re.sub(r"\\\\tfrac", r"\\frac", answer)  # \dfrac -> \frac

    answer = re.sub(r"sqrt\((.*?)\)", r"\\sqrt{\1}", answer)  # sqrt(1) -> \sqrt(1)

    answer = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", answer)  # \sqrt2 -> \sqrt{2}
    answer = re.sub(r"(\d*),\\!(\d*)", r"\1\2", answer)  # 1,\!000 -> 1000

    # answer = re.sub(r"(\d*)\s*\^*(?:\\circ|\{\\circ\})", r"\\frac{\1 * \\pi}{180}", answer)  # 20^\circ -> \frac{20*\pi}{180}
    answer = re.sub(r"(\d*)\s*\^*(?:\\circ|\{\\circ\})", r"\1", answer)  # 20^\circ -> 20

    answer = re.sub(r"\\mathbb{R}", r"(-\\infty, \\infty)", answer)  # R -> (-\infty, \infty)
    answer = re.sub(r"\\mathbb{R}\^\+", r"(0, \\infty)", answer)  # R^+ -> (0, \infty)
    answer = re.sub(r"\\mathbb{R}\^-", r"(-\\infty, 0)", answer)  # R^- -> (-\infty, 0)

    # Process base-x numbering system
    if re.search(r"\d_{\d}", answer) is not None:
        answer = re.sub(r"(\d)_{(\d)}", r"\1_\2", answer)
    if re.search(r"\d_\d", answer) is not None:
        answer = answer.lstrip("0")

    answer = answer.replace("\\cdot", "*")
    answer = answer.rstrip("%")
    answer = answer.replace("√", "sqrt")
    answer = answer.replace("π", "pi")
    answer = answer.replace("∞", "inf")
    answer = answer.replace("∪", "U")
    answer = answer.replace(") U (", ") \\cup (")
    answer = answer.replace("·", "*")
    answer = answer.replace("×", "*")
    answer = answer.replace(" or ", " , ")
    answer = answer.replace(" and ", " , ")
    answer = answer.strip("$")
    answer = answer.rstrip("\\")

    answer = answer.replace("\displaystyle", "")
    answer = answer.replace("{pmatrix}", "{matrix}")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
    ]:
        answer = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", answer)

    if re.search("\d", answer) is None:
        # For pure text answer in MATH
        if len(answer) == 3 and answer[0] == "(" and answer[2] == ")":  # (C) -> C
            return answer[1]
        return answer.strip()
    answer = normalize_numbers(answer.strip())
    return answer


def all_nums(text):
    if isinstance(text, Number):
        return True
    if isinstance(text, (list, tuple)):
        return all([all_nums(_) for _ in text])
    return False


def is_sympy(item):
    return "sympy" in str(type(item)).lower()


def is_set(item):
    type_str = str(type(item)).lower()
    return "sympy" in type_str and "set" in type_str


def is_point(item, dim=2):
    type_str = str(type(item)).lower()
    return "sympy" in type_str and "point" in type_str and str(dim) in type_str


def is_mat(item):
    type_str = str(type(item)).lower()
    return "sympy" in type_str and "mat" in type_str


def is_relation(item):
    type_str = str(type(item)).lower()
    return "sympy" in type_str and "relation" in type_str


def to_set(point):  # (x, y) can be a point or a open interval
    if is_point(point, dim=2):
        return Interval.open(point[0], point[1])
    elif isinstance(point, Number):
        return FiniteSet(point)
    elif isinstance(point, (list, tuple)):
        return FiniteSet(*point)
    else:
        return point


def is_constant(item):
    if isinstance(item, Number):
        return True
    elif hasattr(item, "is_constant") and item.is_constant():
        return True
    else:
        return False


__in_memory_cache__ = {}


def clean_in_memory_cache():
    global __in_memory_cache__
    __in_memory_cache__ = {}


class PMSet(FiniteSet):
    pass


def advance_parse_latex_core(text, item_type=None):
    def check_condition(sign):
        if not sign:
            raise Exception("Cannot pass the condition check")

    check_condition(text is not None)
    text = text.strip()
    check_condition(len(text) > 0)

    inf = ["inf", "\\infty", "infty"]
    if text in inf:
        return oo

    def be_text_set(text):
        return (text[0] in ["(", "["] or text[:2] in ["\{"]) and text[-1] in [")", "]", "}"] and ("," in text or "\{" in text)

    if "," in text and not be_text_set(text):  # Fix case 1, 2 => (1, 2)
        text = "\{" + text + "\}"

    global __in_memory_cache__
    query_key = ("to_latex", text)

    if query_key in __in_memory_cache__:
        return __in_memory_cache__[query_key]

    set_ops = ["\cup", "\cap", "\setminus", "\\backslash"]
    if any([_ in text for _ in set_ops]):
        #  Set operation cannot be handled by sympy
        sets = re.split(r"(\\cup|\\cap|\\setminus|\\backslash)", text)
        ret = last_op = None
        # print(sets)

        for set_i in sets:
            set_i = set_i.strip()
            if set_i in set_ops:
                last_op = set_i
                check_condition(ret is not None)
                continue
            # print(ret, last_op, set_i)

            item = advance_parse_latex_core(set_i, "latex-set")
            # print(item)
            check_condition(item is not None)

            if ret is None:
                ret = item
                continue
            check_condition(last_op is not None)
            if last_op == "\cup":
                ret = ret = ret.union(item)
            elif last_op == "\cap":
                ret.intersect(item)
            elif last_op in ["\setminus", "\\backslash"]:
                ret = Complement(ret, item)

            last_op = None
        check_condition(last_op is None)
    elif "matrix}" in text:
        # Compare matrices
        matrices = re.findall(r"\\begin{(?:.*?)}(.*?)\\end{(?:.*?)}", text)
        check_condition(len(matrices) == 1)

        rows = matrices[0].split("\\\\")

        ret = []
        for row in rows:
            row = row.strip().split("&")
            if len(row) == 0:
                continue
            row = [advance_parse_latex_core(_.strip(), "latex") for _ in row]

            check_condition(all([_ is not None for _ in row]))
            ret.append(row)
        check_condition(len(ret) > 0)
        ret = ImmutableDenseMatrix(ret)
    elif be_text_set(text):
        # print("set", text)
        # Process sets, intervals and points
        cnt = i = 0
        last_i = 0
        items, last_bracket = [], []
        for i in range(len(text)):
            if text[i] in ["(", "[", "{"]:
                cnt += 1
                last_bracket.append(text[i])
            elif text[i] in [")", "]", "}"]:
                cnt -= 1
                pair = (last_bracket[-1], text[i])
                last_bracket = last_bracket[:-1]
                if pair == "(":
                    check_condition(text[i] in [")", "]"])
                elif pair == "[":
                    check_condition(text[i] in ["]"])
                elif pair == "{":
                    check_condition(text[i] in ["}"])

            check_condition(cnt >= 0)
            if (cnt == 1 and text[i] == ",") or (cnt == 0 and i == len(text) - 1):
                item_text = text[last_i : i + 1].strip()
                # print(text, item_text)
                if len(items) == 0:
                    item_text = item_text[1:] if item_text[0] in ["(", "["] else item_text[2:]

                item_text = item_text[:-2] if item_text[-1] == "}" else item_text[:-1]
                item_text = item_text.strip()

                item_text = advance_parse_latex_core(item_text)
                last_i = i + 1
                items.append(item_text)
        check_condition(len(items) > 0)
        has_pm = any([isinstance(_, PMSet) for _ in items])
        if text[0] == "(" and text[-1] == ")" and item_type != "latex-set" and not has_pm:
            ret = Point(*items)
        elif text[:2] == "\{" and text[-2:] == "\}" or len(items) > 2:
            pm_sets = [_ for _ in items if isinstance(_, PMSet)]
            ret = FiniteSet(*[_ for _ in items if not isinstance(_, PMSet)])
            # print(pm_sets, ret)
            if len(pm_sets) > 0:
                for _ in pm_sets:
                    ret = ret.union(_)
        else:
            try:
                ret = Interval(*items)
            except:
                ret = FiniteSet(*items)
    elif "\\pm" in text:
        parts = re.split(r"\\pm", text)
        # print(parts)
        check_condition(len(parts) == 2)
        # print(text, pair)
        number1 = advance_parse_latex_core(f"{parts[0]} + {parts[1]}")
        number2 = advance_parse_latex_core(f"{parts[0]} - {parts[1]}")
        ret = PMSet(number1, number2)
    else:
        # print("!!", text)
        ret = parse_latex(text)
        if item_type == "latex-set":
            check_condition(is_set(ret))

        from .misc import timeout_call

        simplified_version = timeout_call(lambda: ret.simplify())
        check_condition(simplified_version is not None)

        ret_str = str(ret)

        if "i" in text and hasattr(ret, "free_symbols"):
            im_var = None
            for var in ret.free_symbols:
                if str(var) == "i":
                    im_var = var
                    break
            # print(">>", im_var)
            if im_var is not None:
                from sympy import I

                ret = ret.subs({im_var: I})
    __in_memory_cache__[query_key] = ret
    return ret


def advance_parse_latex(text):
    try:
        return advance_parse_latex_core(text)
    except:
        return None


def normalize_answer_core(text, answer_type="text"):
    if not isinstance(text, str):
        return text

    # Process formula like x \in ....
    if re.match(r"\\in(?!f)", text):
        text = re.split(r"\\in", text)[-1]

    from hlm.utils.answer_utils import NO_ANSWER_TEMPLATE

    if text == "No answer!" or len(text) == 0 or any([len(re.findall(template, text, flags=re.IGNORECASE)) > 0 for template in NO_ANSWER_TEMPLATE]):
        return None

    if answer_type in ["text", "date", "bool"]:
        return text.lower()

    try:
        number = eval_text(text)
    except:
        number = None

    if answer_type == "number":
        return number if isinstance(number, Number) else None

    # For base x numbering system, we do not need to parse it
    if re.search(r"\d_\d", text) is not None:
        return text

    if "/" not in text and number is not None:  # If there is no fraction, we do not need to parse it
        return number if isinstance(number, Number) else str(number)

    assert answer_type in ["latex"]

    for var in ["a", "b", "c", "d", "k", "x", "y", "z"]:
        if f"{var}(" in text:
            text = text.replace(f"{var}(", f"{var} * (")
    ret = advance_parse_latex(text)
    return ret


def normalize_answer(text, answer_type="text"):
    ret = normalize_answer_core(text, answer_type)
    try:
        str(ret)
    except:
        ret = None
    return "No answer!" if ret is None else ret
