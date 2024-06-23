# 主观数据集情况说明

|    文件   |  说明  | 
| :------: | :-----: |
|   subjectiveqa_gen               | 630 之前的主观评测题，已舍弃 |
|   subjectiveqav2_gen             | 630 之前的手工筛选的中英文多类型评测题，已舍弃 |
|   subjectiveqav3_gen             | 浩东提供的 360 道题，用于 gpt4 打分（客观推理是默认一起推理不带 meta instruction 的） |
|   subjectiveqav4_gen             |  商汤500题，有需求再跑            |
|   subjectiveqav4_qishi           |  歧视题，交付的时候帮忙跑过一次   |
|   subjectiveqav4_yinsi           |  隐私题，交付的时候帮忙跑过一次   |
|   subjectiveqav5_1_gen           |  袁畅提供的主观题，筛选出了单论对话内容  |
|   subjectiveqav5_2_gen           |  浩东提供的判断中英文切换问题的主观题    |
|   q60_gen                        |  丁昱菲的数据 |  
|   subjectiveqav6_safety           | 安全敏感词prompt数据 |
|   val_examples200_subjective      | 杨超提供的200道安全测试题 |
|   val_examples500_subjective      | 滕妍提供的500道安全测试题 |
|   chat_enhance                    | 情感倾诉类问题 |
|   zh_en_enhance                    | 中英文切换混乱的测试题 |