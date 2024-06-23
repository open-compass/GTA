feedback_format_mapping = {
    'translate':
    '''### Source:\n{question}\n### Translation:\n{generation}''',
    'chat':
    '''### Question:\n{question}\n### Answer:\n{generation}''',
    'qa':
    '''### Question:\n{question}\n### Answer:\n{generation}''',
    'harmlessness':
    '''### Question:\n{question}\n### Answer:\n{generation}''',
    'summary':
    '''### Article:\n{article}\n### Question:
    {question}\n### Answer:\n{generation}''',
    'math_cot':
    '''### Math Question:\n{question}
    ### Rationale:\n{generation}\n''',
    'math_pot':
    '''### Math Question:\n{question}\n### Code:\n{generation}\n''',
    'code_exec':
    '''### Code Question Context:
    {question}\n### Unit Test: {unit_test}\n### Code:
    {generation}\n### Execution Result:\n{exec_rest}''',
    'code_not_exec':
    '''### Code Question Context:\n{question}
    ### Unit Test: {unit_test}\n### Code:\n{generation}'''
}


correction_format_mapping = {
    'translate':
    '''### Source:
{question}
### Translation:
{generation}
### Feedback:
{feedback}''',
    'chat':
    '''### Question:
{question}
### Answer:
{generation}
### Feedback:
{feedback}''',
    'qa':
    '''### Question:
{question}
### Answer:
{generation}
### Feedback:
{feedback}''',
    'harmlessness':
    '''### Question:
{question}
### Answer:
{generation}
### Feedback:
{feedback}''',
    'summary':
    '''### Article:
{article}
### Question:
{question}
### Answer:
{generation}
### Feedback:
{feedback}''',
    'math_cot':
    '''### Math Question:
{question}
### Rationale:
{generation}
### Feedback:
{feedback}''',
    'math_pot':
    '''### Math Question:
{question}
### Code:
{generation}
### Feedback:
{feedback}''',
    'code_exec':
    '''### Code Question Context:
{question}
### Unit Test:
{unit_test}
### Code:
{generation}
### Execution Result:
{exec_rest}
### Feedback:
{feedback}''',
    'code_not_exec':
    '''### Code Question Context:
{question}
### Unit Test:
{unit_test}
### Code:
{generation}
### Feedback:
{feedback}'''
}

comp_feedback_format_mapping = {
    'translate':
    '''### Source:
{question}
### Translation A:
{generation_a}
### Translation B:
{generation_b}''',
    'chat':
    '''### Question:
{question}
### Answer A:
{generation_a}
### Answer B:
{generation_b}''',
    'qa':
    '''### Question:
{question}
### Answer A:
{generation_a}
### Answer B:
{generation_b}''',
    'harmlessness':
    '''### Question:
{question}
### Answer A:
{generation_a}
### Answer B:
{generation_b}''',
    'summary':
    '''### Article:
{article}
### Question:
{question}
### Answer A:
{generation_a}
### Answer B:
{generation_b}''',
    'math_cot':
    '''### Math Question:
{question}
### Rationale A:
{generation_a}
### Rationale B:
{generation_b}''',
    'math_pot':
    '''### Math Question:
{question}
### Code A:
{generation_a}
### Code B:
{generation_b}''',
    'code_exec':
    '''### Code Question Context:
{question}
### Unit Test:
{unit_test}
### Code A:
{generation_a}
### Execution Result A:
{exec_rest_a}
### Code B:
{generation_b}
### Execution Result B:
{exec_rest_b}''',
    'code_not_exec':
    '''### Code Question Context:
{question}
### Unit Test:
{unit_test}
### Code A:
{generation_a}
### Code B:
{generation_b}'''
}




meta_feedback_format_mapping = {
    'translate':
    '''### Source:
{question}
### Translation:
{generation}
### Reference high-quality Feedback
{ground_truth_feedback}
### Feedback To be Evaluated
{evaluated_feedback}
''',
    'chat':
    '''### Question:
{question}
### Answer:
{generation}
### Reference high-quality Feedback
{ground_truth_feedback}
### Feedback To be Evaluated
{evaluated_feedback}
''',
    'qa':
    '''### Question:
{question}
### Answer:
{generation}
### Reference high-quality Feedback
{ground_truth_feedback}
### Feedback To be Evaluated
{evaluated_feedback}
''',
    'harmlessness':
    '''### Question:
{question}
### Answer:
{generation}
### Reference high-quality Feedback
{ground_truth_feedback}
### Feedback To be Evaluated
{evaluated_feedback}
''',
    'summary':
    '''### Article:
{article}
### Question:
{question}
### Answer:
{generation}
### Reference high-quality Feedback
{ground_truth_feedback}
### Feedback To be Evaluated
{evaluated_feedback}
''',
    'math_cot':
    '''### Math Question:
{question}
### Rationale:
{generation}
### Reference high-quality Feedback
{ground_truth_feedback}
### Feedback To be Evaluated
{evaluated_feedback}''',
    'math_pot':
    '''### Math Question:
{question}
### Code:
{generation}
### Reference high-quality Feedback
{ground_truth_feedback}
### Feedback To be Evaluated
{evaluated_feedback}''',
    'code_exec':
    '''### Code Question Context:
{question}
### Unit Test: 
{unit_test}
### Code:
{generation}
### Execution Result:
{exec_rest}
### Reference high-quality Feedback
{ground_truth_feedback}
### Feedback To be Evaluated
{evaluated_feedback}''',
    'code_not_exec':
    '''### Code Question Context:
{question}
### Unit Test:
{unit_test}
### Code:
{generation}
### Reference high-quality Feedback
{ground_truth_feedback}
### Feedback To be Evaluated
{evaluated_feedback}'''
}



mappings = {
    'feedback': feedback_format_mapping,
    'comp_feedback': comp_feedback_format_mapping,
    'correction': correction_format_mapping,
    'meta_feedback': meta_feedback_format_mapping
}

