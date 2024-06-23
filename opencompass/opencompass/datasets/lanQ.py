# flake8: noqa
# yapf: disable
import csv
import json
import re
import string
import unicodedata
from collections import Counter
from typing import List

import jieba
import numpy
from datasets import Dataset

try:
    import langid
except ImportError:
    langid = None

try:
    import textstat
except ImportError:
    textstat = None

try:
    import zhon.hanzi
except ImportError:
    zhon = None

try:
    from hanziconv import HanziConv
except ImportError:
    HanziConv = None

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from .base import BaseDataset

TRANSLATION_TABLE_PUNCTUATION_EN = str.maketrans('', '', string.punctuation)
try:
    TRANSLATION_TABLE_PUNCTUATION_ZH = str.maketrans('', '', zhon.hanzi.punctuation)
except Exception:
    TRANSLATION_TABLE_PUNCTUATION_ZH = None

@LOAD_DATASET.register_module()
class lanQDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        raw_data = []
        with open(path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row) < 1:
                    row = ['']
                raw_data.append({'input': row[0]})
        return Dataset.from_list(raw_data)


@LOAD_DATASET.register_module()
class lanQLongDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        raw_data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                raw_data.append({'input': json.loads(line).get('input')})
        return Dataset.from_list(raw_data)


@ICL_EVALUATORS.register_module()
class lanQEvaluator(BaseEvaluator):

    def score(self, origin_prompt: List, predictions: List) -> dict:
        num = len(predictions)
        res = {}
        res['ALL'] = [1] * num
        res['num_good'] = 0
        res['num_bad'] = 0
        res['total'] = num
        res['score'] =0

        detect_colon_end(res, origin_prompt, predictions)
        detect_bracket_unmatch(res, origin_prompt, predictions)
        detect_doc_repeat(res, origin_prompt, predictions)
        detect_no_punc(res, origin_prompt, predictions)
        detect_chaos_zh(res, origin_prompt, predictions)
        detect_chaos_en(res, origin_prompt, predictions)
        detect_chaos_symbol(res, origin_prompt, predictions)
        detect_language_mixed(res, origin_prompt, predictions)
        detect_chinese_produce_english(res, origin_prompt, predictions)
        detect_english_produce_chinese(res, origin_prompt, predictions)
        detect_advertisement(res, origin_prompt, predictions)
        detect_enter_continuous(res, origin_prompt, predictions)
        detect_content_null(res, origin_prompt, predictions)
        detect_space_more(res, origin_prompt, predictions)
        detect_url_only(res, origin_prompt, predictions)
        detect_enter_more(res, origin_prompt, predictions)
        detect_special_character(res, origin_prompt, predictions)
        detect_word_stuck(res, origin_prompt, predictions)
        detect_watermark(res, origin_prompt, predictions)

        res['num_good'] = res['ALL'].count(1)
        res['num_bad'] = res['ALL'].count(0)
        res['score'] = res['num_good'] / num * 100
        res.pop('ALL')
        # print(res)
        return res


def is_match(s):
    # 定义需要匹配的符号列表
    symbol_list = ['»', '!', '>', '\\\\', '›', '|', '→', '/', '！']
    # 遍历符号列表，检查每个符号是否出现两次以上
    for symbol in symbol_list:
        if s.count(symbol) >= 2:
            return True
    # 使用正则表达式，检查除了定义的符号和英文字母、数字、空格之外，字符串中是否还有其他字符
    pattern = r'[^a-zA-Z0-9\s' + ''.join(symbol_list) + ']'
    if re.search(pattern, s):
        return False
    return False


def form_ngrams(sequence, n):
    history = []
    # build the first ngram, yielding only when we have a full ngram
    while n > 1:
        try:
            next_item = next(sequence)
        except StopIteration:
            # no more data, terminate the generator
            return
        history.append(next_item)
        n -= 1

    # yield each ngram we have, then add the next item and repeat
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def normalize(
        text: str,
        remove_punct: bool = True,
        lowercase: bool = True,
        nfd_unicode: bool = True,
        white_space: bool = True
) -> str:
    """Normalize the text by lowercasing and removing punctuation."""
    # remove punctuation
    if remove_punct:
        text = text.translate(TRANSLATION_TABLE_PUNCTUATION_EN)
        text = text.translate(TRANSLATION_TABLE_PUNCTUATION_ZH)

    # lowercase
    if lowercase:
        text = text.lower()

    if white_space:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)

    # NFD unicode normalization
    if nfd_unicode:
        text = unicodedata.normalize('NFD', text)

    return text


def split_words(content: str):
    res = []
    for i in content.split():
        en_word = ''
        for j in i:
            if re.match(r'[\u4e00-\u9fff]', j):
                if en_word != '':
                    res.append(en_word)
                    en_word = ''
                res.append(j)
            else:
                en_word = en_word + j
        if en_word == i:
            res.append(i)
    return tuple(res)


def base_rps_frac_chars_in_dupe_ngrams(NGRAM_SIZE, content):
    """Base class for calculating the fraction of characters in duplicate word
    N-grams.

    This operates on the lower-cased, punctuation removed content. The function
    also ensures that characters in overlapping ngrams are only counted once.
    """
    normalized_content = normalize(content)
    normalized_words = split_words(normalized_content)

    if len(normalized_words) < NGRAM_SIZE:
        return 0

    # fetch the ngrams from the document if they exist, otherwise
    # compute them
    doc_n_grams = tuple(form_ngrams(iter(normalized_words), NGRAM_SIZE))

    # keep only ngrams which occur at least twice
    ngram_dupes = {
        ngram for ngram, count in Counter(doc_n_grams).items() if count > 1
    }

    duplicated_grams = numpy.zeros(len(normalized_words), dtype=int)
    i = 0
    for ngram in doc_n_grams:
        if ngram in ngram_dupes:
            duplicated_grams[i: i + NGRAM_SIZE] = 1

        i += 1

    word_lengths = numpy.array(list(map(len, normalized_words)))
    chars_duped = numpy.sum(word_lengths * duplicated_grams)
    total_chars = numpy.sum(word_lengths)

    if total_chars == 0:
        return 0

    score = float(chars_duped / total_chars) * 100
    return score


def detect_colon_end(res, origin_prompt: List, predictions: List):
    """content最后一个字符是冒号."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_COLON_END'] = 0
    res['ERROR_COLON_END_LIST'] = []
    for i in range(num):
        content = predictions[i]
        if len(content) < 1:
            continue
        if content[-1] == ':':
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_COLON_END_LIST'].append(i)
    res['ERROR_COLON_END'] = good_list.count(0)


def detect_bracket_unmatch(res, origin_prompt: List, predictions: List):
    """检查开闭括号数量是否一致."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_BRACKET_UNMATCH'] = 0
    res['ERROR_BRACKET_UNMATCH_LIST'] = []
    for i in range(num):
        flag = True
        content = predictions[i]
        if content.count('[') != content.count(']'):
            flag = False
        if content.count('{') != content.count('}'):
            flag = False
        if content.count('【') != content.count('】'):
            flag = False
        if content.count('《') != content.count('》'):
            flag = False
        if flag != True :
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_BRACKET_UNMATCH_LIST'].append(i)
    res['ERROR_BRACKET_UNMATCH'] = good_list.count(0)


# def detect_doc_repeat(predictions: List, good_list: List) -> List:
#     """检查content内是否有连续重复."""
#     num = len(predictions)
#     good_list[ERROR_DOC_REPEAT] = [1] * num
#     good_list[ERROR_DOC_REPEAT_LIST] = []
#     for i in range(num):
#         content = predictions[i]
#         start_pattern = re.compile(r'(?s)(.+?)\1+')
#         end_pattern = re.compile(r'(.+?)\1+')
#         start_matches = start_pattern.findall(content)
#         end_matches = end_pattern.findall(content)
#         combined_matches = start_matches + end_matches
#         if combined_matches:
#             longest_string = max(combined_matches, key=len)
#             if len(longest_string) > 15:
#                 if not is_match(longest_string):
#                     good_list['ALL'][i] = 0
#                     good_list[ERROR_DOC_REPEAT][i] = 0
#                     good_list[ERROR_DOC_REPEAT_LIST].append(i)
#     return good_list


def detect_doc_repeat(res, origin_prompt: List, predictions: List):
    """检查content内是否有连续重复."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_DOC_REPEAT'] = 0
    res['ERROR_DOC_REPEAT_LIST'] = []
    for i in range(num):
        content = predictions[i]
        repeat_score = base_rps_frac_chars_in_dupe_ngrams(6, content)
        if repeat_score >= 80:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_DOC_REPEAT_LIST'].append(i)
    res['ERROR_DOC_REPEAT'] = good_list.count(0)


def detect_no_punc(res, origin_prompt: List, predictions: List):
    """检查content内是否有大段无标点."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_NO_PUNC'] = 0
    res['ERROR_NO_PUNC_LIST'] = []
    for i in range(num):
        content = predictions[i]
        paragraphs = content.split('\n')
        max_word_count = 0
        for paragraph in paragraphs:
            if len(paragraph) == 0:
                continue
            sentences = re.split(r'[-–.!?,;•、。！？，；·]', paragraph)
            for sentence in sentences:
                words = sentence.split()
                word_count = len(words)
                if word_count > max_word_count:
                    max_word_count = word_count
        text_stat_res = textstat.flesch_reading_ease(content)
        if int(max_word_count) > 56 and text_stat_res < 20:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_NO_PUNC_LIST'].append(i)
    res['ERROR_NO_PUNC'] = good_list.count(0)


def detect_chaos_zh(res, origin_prompt: List, predictions: List):
    """检查content内是否有中文乱码."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_CHAOS_ZH'] = 0
    res['ERROR_CHAOS_ZH_LIST'] = []
    for i in range(num):
        content = predictions[i]
        lan = langid.classify(content)[0]
        if lan != 'zh':
            continue
        punctuation_en = string.punctuation
        punctuation_zh = zhon.hanzi.punctuation
        for p in punctuation_en:
            content = content.replace(p, '')
        for p in punctuation_zh:
            content = content.replace(p, '')
        pattern = r'[a-zA-Zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ\n\s]'
        content = re.sub(pattern, '', content)
        content_simplified = HanziConv.toSimplified(content)
        str_len = len(content)
        seg_len = len(list(jieba.cut(content_simplified)))
        num_bytes = len(predictions[i].encode('utf-8'))
        tokens_len = int(num_bytes * 0.248)
        if str_len == 0 or seg_len == 0 or tokens_len < 50:
            continue
        if str_len / seg_len <= 1.1:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_CHAOS_ZH_LIST'].append(i)
    res['ERROR_CHAOS_ZH'] = good_list.count(0)


def detect_chaos_en(res, origin_prompt: List, predictions: List):
    """检查content内是否有英文乱码."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_CHAOS_EN'] = 0
    res['ERROR_CHAOS_EN_LIST'] = []
    for i in range(num):
        content = predictions[i]
        lan = langid.classify(content)[0]
        if lan != 'en':
            continue
        punctuation_en = string.punctuation
        punctuation_zh = zhon.hanzi.punctuation
        for p in punctuation_en:
            content = content.replace(p, '')
        for p in punctuation_zh:
            content = content.replace(p, '')
        pattern = r'[\n\s]'
        content = re.sub(pattern, '', content)
        str_len = len(content)
        seg_len = len(list(jieba.cut(content)))
        num_bytes = len(predictions[i].encode('utf-8'))
        tokens_len = int(num_bytes * 0.248)
        if str_len == 0 or seg_len == 0 or tokens_len < 50:
            continue
        if str_len / seg_len <= 1.2:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_CHAOS_EN_LIST'].append(i)
    res['ERROR_CHAOS_EN'] = good_list.count(0)


def detect_chaos_symbol(res, origin_prompt: List, predictions: List):
    """检查content内是否有大量非正文内容."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_CHAOS_SYMBOL'] = 0
    res['ERROR_CHAOS_SYMBOL_LIST'] = []
    for i in range(num):
        content = predictions[i]
        pattern = r'[0-9a-zA-Z\u4e00-\u9fa5]'
        content = re.sub(pattern, '', content)
        str_len = len(predictions[i])
        symbol_len = len(content)
        if str_len == 0 or symbol_len == 0:
            continue
        if symbol_len / str_len > 0.5:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_CHAOS_SYMBOL_LIST'].append(i)
    res['ERROR_CHAOS_SYMBOL'] = good_list.count(0)


def detect_language_mixed(res, origin_prompt: List, predictions: List):
    """检查content内是否有中英文混杂."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_LANGUAGE_MIXED'] = 0
    res['ERROR_LANGUAGE_MIXED_LIST'] = []
    for i in range(num):
        content = predictions[i]
        punctuation_en = string.punctuation
        punctuation_zh = zhon.hanzi.punctuation
        for p in punctuation_en:
            content = content.replace(p, '')
        for p in punctuation_zh:
            content = content.replace(p, '')
        en_len = len(re.findall(r'[a-zA-Z]', content))
        zh_len = len(re.findall(r'[\u4e00-\u9fa5]', content))
        count_len = len(content)
        if count_len != 0:
            if en_len / count_len >= 0.5 and zh_len / count_len >= 0.1:
                res['ALL'][i] = 0
                good_list[i] = 0
                res['ERROR_LANGUAGE_MIXED_LIST'].append(i)
    res['ERROR_LANGUAGE_MIXED'] = good_list.count(0)


def detect_chinese_produce_english(res, origin_prompt: List, predictions: List):
    """检查中文promt生成英文content."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_CHINESE_PRODUCE_ENGLISH'] = 0
    res['ERROR_CHINESE_PRODUCE_ENGLISH_LIST'] = []
    for i in range(num):
        prompt = origin_prompt[i]
        content = predictions[i]
        punctuation_en = string.punctuation
        punctuation_zh = zhon.hanzi.punctuation
        for p in punctuation_en:
            prompt = prompt.replace(p, '')
            content = content.replace(p, '')
        for p in punctuation_zh:
            prompt = prompt.replace(p, '')
            content = content.replace(p, '')
        zh_len_prompt = len(re.findall(r'[\u4e00-\u9fa5]', prompt))
        en_len_content = len(re.findall(r'[a-zA-Z]', content))
        prompt_len = len(prompt)
        content_len = len(content)
        if prompt_len == 0 or content_len == 0:
            continue
        #prompt非中文
        if zh_len_prompt / prompt_len < 0.6:
            continue
        if en_len_content / content_len >= 0.6 :
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_CHINESE_PRODUCE_ENGLISH_LIST'].append(i)
    res['ERROR_CHINESE_PRODUCE_ENGLISH'] = good_list.count(0)


def detect_english_produce_chinese(res, origin_prompt: List, predictions: List):
    """检查英文promt生成中文content."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_ENGLISH_PRODUCE_CHINESE'] = 0
    res['ERROR_ENGLISH_PRODUCE_CHINESE_LIST'] = []
    for i in range(num):
        prompt = origin_prompt[i]
        content = predictions[i]
        punctuation_en = string.punctuation
        punctuation_zh = zhon.hanzi.punctuation
        for p in punctuation_en:
            prompt = prompt.replace(p, '')
            content = content.replace(p, '')
        for p in punctuation_zh:
            prompt = prompt.replace(p, '')
            content = content.replace(p, '')
        en_len_prompt = len(re.findall(r'[a-zA-Z]', prompt))
        zh_len_content = len(re.findall(r'[\u4e00-\u9fa5]', content))
        prompt_len = len(prompt)
        content_len = len(content)
        if prompt_len == 0 or content_len == 0:
            continue
        #prompt非英文
        if en_len_prompt / prompt_len < 0.6:
            continue
        if zh_len_content / content_len >= 0.6 :
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_ENGLISH_PRODUCE_CHINESE_LIST'].append(i)
    res['ERROR_ENGLISH_PRODUCE_CHINESE'] = good_list.count(0)


def detect_advertisement(res, origin_prompt: List, predictions: List):
    """检查content内是否有广告."""
    AdList_en = ['deadlinesOrder', 'Kindly click on ORDER NOW to receive an']
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_ADVERTISEMENT'] = 0
    res['ERROR_ADVERTISEMENT_LIST'] = []
    for i in range(num):
        content = predictions[i]
        ad_en_count = len(re.findall('|'.join(AdList_en), content))
        if ad_en_count > 0:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_ADVERTISEMENT_LIST'].append(i)
    res['ERROR_ADVERTISEMENT'] = good_list.count(0)


def detect_enter_continuous(res, origin_prompt: List, predictions: List):
    """检查content内是否有连续大于8个的回车."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_ENTER_CONTINUOUS'] = 0
    res['ERROR_ENTER_CONTINUOUS_LIST'] = []
    for i in range(num):
        content = predictions[i]
        count = len(re.findall(r'\n{8,}|\r{8,}', content))
        if count > 0:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_ENTER_CONTINUOUS_LIST'].append(i)
    res['ERROR_ENTER_CONTINUOUS'] = good_list.count(0)


def detect_content_null(res, origin_prompt: List, predictions: List):
    """检查content内是否为空."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_CONTENT_NULL'] = 0
    res['ERROR_CONTENT_NULL_LIST'] = []
    for i in range(num):
        content = predictions[i]
        count = len(content.strip())
        if count == 0:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_CONTENT_NULL_LIST'].append(i)
    res['ERROR_CONTENT_NULL'] = good_list.count(0)


def detect_space_more(res, origin_prompt: List, predictions: List):
    """检查content内是否有连续500个以上的空格."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_SPACE_MORE'] = 0
    res['ERROR_SPACE_MORE_LIST'] = []
    for i in range(num):
        content = predictions[i]
        count = len(re.findall(r' {500,}', content))
        if count > 0:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_SPACE_MORE_LIST'].append(i)
    res['ERROR_SPACE_MORE'] = good_list.count(0)


def detect_url_only(res, origin_prompt: List, predictions: List):
    """检查content内是否只有url."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_URL_ONLY'] = 0
    res['ERROR_URL_ONLY_LIST'] = []
    for i in range(num):
        content = predictions[i]
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'  # noqa
        content = re.sub(pattern, '', content)
        count = len(content.strip())
        if count == 0:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_URL_ONLY_LIST'].append(i)
    res['ERROR_URL_ONLY'] = good_list.count(0)


def detect_enter_more(res, origin_prompt: List, predictions: List):
    """检查content内是否有超过25%正文占比的回车."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_ENTER_MORE'] = 0
    res['ERROR_ENTER_MORE_LIST'] = []
    for i in range(num):
        content = predictions[i]
        enter_count = content.count('\n')
        if enter_count < 5:
            continue
        content = normalize(content)
        count = len(split_words(content))
        if count != 0:
            ratio = enter_count / count * 100
            if ratio >= 35:
                res['ALL'][i] = 0
                good_list[i] = 0
                res['ERROR_ENTER_MORE_LIST'].append(i)
    res['ERROR_ENTER_MORE'] = good_list.count(0)


def detect_special_character(res, origin_prompt: List, predictions: List):
    """检查content内是否有特殊字符."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_SPECIAL_CHARACTER'] = 0
    res['ERROR_SPECIAL_CHARACTER_LIST'] = []
    for i in range(num):
        content = predictions[i]
        count = len(re.findall(r'[�]', content))
        if count > 0:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_SPECIAL_CHARACTER_LIST'].append(i)
    res['ERROR_SPECIAL_CHARACTER'] = good_list.count(0)


def detect_word_stuck(res, origin_prompt: List, predictions: List):
    """检查content内是否有英文单词黏连."""
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_WORD_STUCK'] = 0
    res['ERROR_WORD_STUCK_LIST'] = []
    for i in range(num):
        content = predictions[i]
        words = re.findall(r'[a-zA-Z]+', content)
        max_word_len = 0
        for word in words:
            if len(word) > max_word_len:
                max_word_len = len(word)
        if max_word_len > 45:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_WORD_STUCK_LIST'].append(i)
    res['ERROR_WORD_STUCK'] = good_list.count(0)


def detect_watermark(res, origin_prompt: List, predictions: List):
    """检查content内是否有水印."""
    WatermarkList = ['仩嗨亾笁潪能', '上s海h人r工g智z能n实s验y室s']
    num = len(predictions)
    good_list = [1] * num
    res['ERROR_DETECT_WATERMARK'] = 0
    res['ERROR_DETECT_WATERMARK_LIST'] = []
    for i in range(num):
        content = predictions[i]
        count = len(re.findall('|'.join(WatermarkList), content))
        if count > 0:
            res['ALL'][i] = 0
            good_list[i] = 0
            res['ERROR_DETECT_WATERMARK_LIST'].append(i)
    res['ERROR_DETECT_WATERMARK'] = good_list.count(0)
