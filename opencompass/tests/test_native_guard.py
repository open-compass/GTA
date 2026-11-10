import unittest

from opencompass.models.native_workflow.native_guard import final_guard, pseudo_tool


TOOLS = ['ReadPDF', 'ReadCSV', 'GoogleSearch', 'Calculator', 'HtmlFileGenerator', 'CsvFileGenerator']
INPUT = '/benchmark/resources/files/2-1.pdf'
READ = {'role': 'assistant', 'tool_calls': [{'function': {
    'name': 'ReadPDF', 'arguments': {'file': INPUT}}}]}


def guard(prompt, text, **evidence):
    return final_guard(prompt=prompt, text=text, **{'used_tools': [], 'files': [],
                       'available_tools': TOOLS, **evidence})


class NativeGuardTests(unittest.TestCase):
    def test_scout_printed_a_call_instead_of_reading_policies(self):
        task = 'Analyze the four PDF documents located at resources/files/2-1.pdf and other paths.'
        answer = 'First, I will read the policies.\n[ReadPDF(file="resources/files/2-1.pdf")]'
        self.assertIn('printed ReadPDF', guard(task, answer, input_files=[INPUT]))
        self.assertIn('inspect', guard(task, 'The premium is $10.', input_files=[INPUT],
                                      used_tools=['GoogleSearch']))
        self.assertIsNone(guard(task, 'The premium is $10.', input_files=[INPUT],
                                used_tools=['ReadPDF'], tool_steps=[READ]))

    def test_passing_a_path_to_a_writer_or_mentioning_it_is_not_reading_it(self):
        fake_read = {'role': 'assistant', 'tool_calls': [{'function': {
            'name': 'HtmlFileGenerator', 'arguments': {'html_content': INPUT}}}]}
        other = {'role': 'assistant', 'tool_calls': [{'function': {
            'name': 'ReadPDF', 'arguments': {'file': '/different/other.pdf'}}}]}
        for step, name in [(fake_read, 'HtmlFileGenerator'), (other, 'ReadPDF')]:
            self.assertIsNotNone(guard('Summarize the attached file.', 'Here is the summary.',
                                  input_files=[INPUT], tool_steps=[step], used_tools=[name]))

    def test_additional_resources_alone_never_force_tools(self):
        self.assertIsNone(guard('What does insurance mean?', 'Protection against risk.', input_files=[INPUT]))
        self.assertIsNone(guard('Write a script to read the attached PDF.', 'import pymupdf',
                               input_files=[INPUT]))

    def test_required_web_search_and_explicit_calculator(self):
        for prompt, tool in (('Verify the opening hours by searching the web.', 'GoogleSearch'),
                             ('Use Calculator to square 32.', 'Calculator')):
            self.assertIn(tool, guard(prompt, 'Here is the answer.'))
            self.assertIsNone(guard(prompt, 'Here is the answer.', used_tools=[tool]))
        for prompt in ('Explain how to search the web.', 'Give the definition without using an online search.',
                       'Do not search the web. Give the definition of a circle.'):
            self.assertIsNone(guard(prompt, 'A circle is the set of points equidistant from its center.'))

    def test_output_formats_and_generic_downloadable_deliverables(self):
        for prompt in ('The PDF report should be saved.', 'The deliverable is a polished HTML handbook.',
                       'A standalone report must be submitted.'):
            self.assertIsNotNone(guard(prompt, 'Here is a summary.'))
        task = ('Two deliverable files are to be produced: an HTML file named report.html and a CSV file '
                'named comparison.csv.')
        self.assertIn('csv', guard(task, 'Done.', files=['/out/report.html']))
        self.assertIsNone(guard(task, 'Done.', files=['/out/report.html', '/out/comparison.csv']))
        self.assertIsNone(guard('Create a PDF or Word document.', 'Done.', files=['/out/report.docx']))
        self.assertIsNone(guard('Provide a table in your reply.', 'A | B'))
        self.assertIsNone(guard('Show HTML code inline.', '<p>Hi.</p>'))

    def test_genuine_limitation_with_failed_real_attempt(self):
        failed = dict(READ, error={'type':'tool_error','msg':'file is encrypted'})
        self.assertIsNone(guard('Read the attached PDF.', 'I could not read the encrypted document.',
                               input_files=[INPUT], tool_steps=[failed]))
        self.assertIsNotNone(guard('Read the attached PDF.', 'I could not read the document.',
                                   input_files=[INPUT]))
        self.assertIsNone(guard('Create a PDF.', 'I cannot create files in this environment.',
                                available_tools=[]))

    def test_scoped_pseudo_call_detector(self):
        for answer in ('[ReadPDF(file="x.pdf")]',
                       'First, I will read the policy. [ReadPDF(file="x.pdf")]',
                       '{"tool_calls":[{"function":{"name":"ReadPDF","arguments":"{}"}}]}',
                       '{"name":"ReadPDF","arguments":{"file":"x.pdf"}}',
                       'Tool: ReadPDF\nTool Input: {"file":"x.pdf"}'):
            self.assertEqual('ReadPDF', pseudo_tool(answer, TOOLS), answer)
        for answer in ('[Nonexistent(file="x.pdf")]', 'A bracketed call looks like [ReadPDF(file="x.pdf")].',
                       '> [ReadPDF(file="x.pdf")]', '"[ReadPDF(file=path)]"',
                       '\x60\x60\x60json\n{"name":"ReadPDF","arguments":{}}\n\x60\x60\x60',
                       'The syntax looks like:\nTool: ReadPDF\nTool Input: {"file":"x.pdf"}'):
            self.assertIsNone(pseudo_tool(answer, TOOLS), answer)
        example = '[ReadPDF(file="x.pdf")]'
        self.assertIsNone(guard('Show an example of a tool call.', example))

    def test_action_after_prose_or_earlier_code_needs_real_execution(self):
        prose = ('I have outlined the stages and the \x60tools\x60 below. ' * 18
                 + '\n\x60\x60\x60text\n[Calculator(a=4)]\n\x60\x60\x60\n\n')
        for body in ('[GoogleSearch(query="top funds", k=10)]',
                     '[HtmlFileGenerator(name="r.html"); CsvFileGenerator(name="r.csv")]'):
            self.assertIn('did not execute', guard('Identify ten firms.', prose + body))
        self.assertIsNone(guard('Outline a study plan.', 'First, I will set a daily study schedule.'))
        self.assertIsNone(guard('Explain how to use the search client.',
                                prose + 'For example:\n[GoogleSearch(query="top funds")]'))

    def test_placeholders_and_future_archives_do_not_count_as_files(self):
        prompt = 'The deliverables are two files, a PDF and a Word document.'
        self.assertIsNotNone(guard(prompt, 'Output: [Insert DOCX file]\n[Insert PDF file]'))
        prompt = 'Compress the page files into a zip archive named "site.zip".'
        answer = 'Here is the command.\n\x60\x60\x60sh\nzip site.zip index.html\n\x60\x60\x60\nThis will create a zip archive.'
        self.assertIsNotNone(guard(prompt, answer))
        self.assertIsNone(guard('Explain how to create a zip archive.', answer))


if __name__ == '__main__':
    unittest.main()
