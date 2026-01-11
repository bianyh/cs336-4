from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding


import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType

import fasttext
from typing import Tuple

import re






# 文本语言识别模型路径
FASTTEXT_LID_MODEL_PATH = "/home/bianyuhan/LLM Learning/cs336/data/lid.176.bin"

# 全局加载模型（只加载一次）
_lid_model = fasttext.load_model(FASTTEXT_LID_MODEL_PATH)


# NSFW不良内容
NSFW_MODEL_PATH = "/home/bianyuhan/LLM Learning/cs336/data/jigsaw_fasttext_bigrams_nsfw_final.bin"
_nsfw_model = None
def get_nsfw_model():
    global _nsfw_model
    if _nsfw_model is None:
        _nsfw_model = fasttext.load_model(NSFW_MODEL_PATH)
    return _nsfw_model
def classify_nsfw(text: str):
    model = get_nsfw_model()
    labels, scores = model.predict(text, k=1)

    label = labels[0]
    score = scores[0]

    if label == "__label__nsfw":
        return "nsfw", score
    else:
        return "non-nsfw", score

# hatespeech有毒言论
TOXIC_MODEL_PATH = "/home/bianyuhan/LLM Learning/cs336/data/jigsaw_fasttext_bigrams_hatespeech_final.bin"
_toxic_model = None
def get_toxic_model():
    global _toxic_model
    if _toxic_model is None:
        _toxic_model = fasttext.load_model(TOXIC_MODEL_PATH)
    return _toxic_model
def classify_toxic_speech(text: str):
    model = get_toxic_model()
    labels, scores = model.predict(text, k=1)

    label = labels[0]
    score = scores[0]

    if "toxic" in label or "hate" in label:
        return "toxic", score
    else:
        return "non-toxic", score




# 从warc文件中取出一些用于测试的内容
def test_read_warc(path: str, max_records: int = 3):
    """
    Read and print contents from a .warc.gz or .warc.wet.gz file.
    """
    res = []
    with gzip.open(path, "rb") as f:
        for i, record in enumerate(ArchiveIterator(f)):
            if i >= max_records:
                break

            # 只关心 response / conversion 记录
            if record.record_type not in (
                WarcRecordType.response,
                WarcRecordType.conversion,
            ):
                continue

            print("=" * 80)
            print(f"Record {i}")
            print("URL:", record.headers.get("WARC-Target-URI"))
            print("Content-Type:", record.headers.get("Content-Type"))

            payload = record.reader.read()

            res.append(payload)

    return res


# html字节转换成可读文本
def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    """
    Extract visible text from raw HTML bytes.
    """

    # 自动检测编码
    encoding = detect_encoding(html_bytes)
    html_str = html_bytes.decode(encoding, errors="replace")

    # HTML → 可见文本
    text = extract_plain_text(
        html_str,
        main_content=False,   # 不强制主内容（更通用）
        preserve_formatting=True
    )

    return text

# 文本语言识别
def identify_language(text: str) -> Tuple[str, float]:
    """
    Identify the main language of a text using fastText.

    Returns:
        (language_code, confidence_score)
    """
    if not text.strip():
        return "unknown", 0.0

    # fastText 只能处理单行 → 压平文本
    text = text.replace("\n", "")

    labels, probs = _lid_model.predict(text, k=1)

    label = labels[0]          # e.g. "__label__en"
    score = float(probs[0])    # ∈ [0, 1]

    # fastText label → language code
    lang = label.replace("__label__", "")

    # if lang.startswith("zh"):
    #     lang = "zh"
    # elif lang.startswith("en"):
    #     lang = "en"

    return lang, score


# 掩码邮件地址
EMAIL_PATTERN = re.compile(
    r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)
def mask_emails(text: str) -> Tuple[str, int]:
    matches = EMAIL_PATTERN.findall(text)
    masked_text = EMAIL_PATTERN.sub("|||EMAIL_ADDRESS|||", text)
    return masked_text, len(matches)

# 掩码手机号码
PHONE_PATTERN = re.compile(
    r"(?:\+1[\s\-\.]?)?"
    r"(?:\(?\d{3}\)?[\s\-\.]?)"
    r"\d{3}[\s\-\.]?\d{4}"
)

# 掩码手机号
def mask_phone_numbers(text: str) -> Tuple[str, int]:
    matches = PHONE_PATTERN.findall(text)
    masked_text = PHONE_PATTERN.sub("|||PHONE_NUMBER|||", text)
    return masked_text, len(matches)


IP_PATTERN = re.compile(
    r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}"
)


def mask_ips(text: str) -> Tuple[str, int]:
    matches = IP_PATTERN.findall(text)
    masked_text = IP_PATTERN.sub("|||IP_ADDRESS|||", text)
    return masked_text, len(matches)


# 过滤数据质量
def gopher_quality_filter(text: str) -> bool:
    if not text or not text.strip():
        return False

    # ---- Word tokenization ----
    words = text.split()
    num_words = len(words)

    # Rule 1: document length
    if num_words < 50 or num_words > 100_000:
        return False

    # ---- Rule 2: mean word length ----
    word_lengths = [len(w) for w in words]
    mean_word_length = sum(word_lengths) / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # ---- Rule 3: ellipsis lines ----
    lines = text.splitlines()
    if lines:
        ellipsis_lines = sum(
            1 for line in lines if line.rstrip().endswith("...")
        )
        if ellipsis_lines / len(lines) > 0.3:
            return False

    # ---- Rule 4: alphabetic words ratio ----
    alphabetic_words = sum(
        1 for w in words if any(c.isalpha() for c in w)
    )
    if alphabetic_words / num_words < 0.8:
        return False

    return True




if __name__=='__main__':
    data = test_read_warc('/home/bianyuhan/LLM Learning/cs336/data/warc.gz')
    text = extract_text_from_html_bytes(data[0])
    print(text)
    lang, score = identify_language(text=text)
    print(lang, score)

