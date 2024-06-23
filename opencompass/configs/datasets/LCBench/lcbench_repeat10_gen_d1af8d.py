from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import LCDataset, LCPassKEvaluator

LC_reader_cfg = dict(
    input_columns=['text', 'test_list'], output_column='test_column')


LC_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="You are an expert Python programmer, and here is your task: You are given three positive integers n, x, and y.\nIn a city, there exist houses numbered 1 to n connected by n streets. There is a street connecting the house numbered i with the house numbered i + 1 for all 1 <= i <= n - 1 . An additional street connects the house numbered x with the house numbered y.\nFor each k, such that 1 <= k <= n, you need to find the number of pairs of houses (house1, house2) such that the minimum number of streets that need to be traveled to reach house2 from house1 is k.\nReturn a 1-indexed array result of length n where result[k] represents the total number of pairs of houses such that the minimum streets required to reach one house from the other is k.\nNote that x and y can be equal. Your code should pass these tests:\n\n assert countOfPairs(n = 3, x = 1, y = 3) == [6,0,0]\n assert countOfPairs(n = 5, x = 2, y = 4) == [10,8,2,0,0] \n assert countOfPairs(n = 4, x = 1, y = 1) == [6,4,2,0] \n"
                ),
                dict(
                    role="BOT",
                    prompt="[BEGIN]\n 'from itertools import accumulate\ndef countOfPairs(n, x, y):\n        x, y = min(x, y), max(x, y)\n        A = [0] * n\n        for i in range(1, n + 1):\n            A[0] += 2                                   \n            A[min(i - 1, abs(i - y) + x)] -= 1          \n            A[min(n - i, abs(i - x) + 1 + n - y)] -= 1  \n            A[min(abs(i - x), abs(y - i) + 1)] += 1     \n            A[min(abs(i - x) + 1, abs(y - i))] += 1     \n            r = max(x - i, 0) + max(i - y, 0)\n            A[r + (y - x + 0) // 2] -= 1                \n            A[r + (y - x + 1) // 2] -= 1                \n        return list(accumulate(A))' \n[DONE] \n\n "
                ),
                dict(
                    role="HUMAN",
                    prompt="You are an expert Python programmer, and here is your task: You are given a string word containing lowercase English letters.\nTelephone keypads have keys mapped with distinct collections of lowercase English letters, which can be used to form words by pushing them. For example, the key 2 is mapped with [\"a\",\"b\",\"c\"], we need to push the key one time to type \"a\", two times to type \"b\", and three times to type \"c\" .\nIt is allowed to remap the keys numbered 2 to 9 to distinct collections of letters. The keys can be remapped to any amount of letters, but each letter must be mapped to exactly one key. You need to find the minimum number of times the keys will be pushed to type the string word.\nReturn the minimum number of pushes needed to type word after remapping the keys.\nAn example mapping of letters to keys on a telephone keypad is given below. Note that 1, *, #, and 0 do not map to any letters. Your code should pass these tests:\n\n assert minimumPushes(\"abcde\") == 5 \n assert minimumPushes(\"xyzxyzxyzxyz\") == 12 \n assert minimumPushes(\"aabbccddeeffgghhiiiiii\") == 24 \n"
                ),
                dict(
                    role="BOT",
                    prompt="[BEGIN]\n 'def minimumPushes(word):\n        letter_counts = {}\n        for c in word:\n            letter_counts[c] = letter_counts.get(c, 0) + 1\n        counts = list(letter_counts.values())\n        counts.sort(reverse=True)\n        ans, row = 0, 1\n        for i in range(len(counts)):\n            if i > 7 and i % 8 == 0:\n                row += 1\n            ans += row * counts[i]\n        return ans' \n[DONE] \n\n "
                ),
                dict(
                    role="HUMAN",
                    prompt="You are an expert Python programmer, and here is your task: You are given three positive integers n, x, and y.\nIn a city, there exist houses numbered 1 to n connected by n streets. There is a street connecting the house numbered i with the house numbered i + 1 for all 1 <= i <= n - 1 . An additional street connects the house numbered x with the house numbered y.\nFor each k, such that 1 <= k <= n, you need to find the number of pairs of houses (house1, house2) such that the minimum number of streets that need to be traveled to reach house2 from house1 is k.\nReturn a 1-indexed array result of length n where result[k] represents the total number of pairs of houses such that the minimum streets required to reach one house from the other is k.\nNote that x and y can be equal. Your code should pass these tests:\n\n assert countOfPairs(n = 3, x = 1, y = 3) == [6,0,0] \n assert countOfPairs(n = 5, x = 2, y = 4) == [10,8,2,0,0] \n assert countOfPairs(n = 4, x = 1, y = 1) == [6,4,2,0] \n"
                ),
                dict(
                    role="BOT",
                    prompt="[BEGIN]\n 'def countOfPairs(n, x, y):\n        ans = [0] * n\n        for i in range(1, n + 1):\n            for j in range(i + 1, n + 1):\n                minimum = min(abs(x - i) + abs(y - j) + 1, min(abs(y - i) + abs(x - j) + 1, abs(i - j)))\n                ans[minimum - 1] += 2\n        return ans' \n[DONE] \n\n "
                ),
                dict(
                    role="HUMAN",
                    prompt="You are an expert Python programmer, and here is your task: {text} Your code should pass these tests:\n\n {test_list}  \n"
                ),
                dict(role="BOT", prompt="[BEGIN]\n"),
            ], )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

LC_eval_cfg = dict(evaluator=dict(type=LCPassKEvaluator), pred_role="BOT")

LC_datasets_pass10 = [
    dict(
        type=LCDataset,
        abbr='lc_pass10_eng',
        path='./data/LCBench2023/LCBench2023.jsonl',
        num_repeats=10,
        reader_cfg=LC_reader_cfg,
        infer_cfg=LC_infer_cfg,
        eval_cfg=LC_eval_cfg)
]

LC_cn_datasets_pass10 = [
    dict(
        type=LCDataset,
        abbr='lc_pass10_cn',
        path='./data/LCBench2023/LCBench2023_cn.jsonl',
        num_repeats=10,
        reader_cfg=LC_reader_cfg,
        infer_cfg=LC_infer_cfg,
        eval_cfg=LC_eval_cfg)
]
