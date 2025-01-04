from pathlib import Path


class ExampleRecorder:
    def __title_row(self, title: str):
        return '\n' + '*' * 40 + f'{title:^20}' + '*' * 40 + '\n'

    def __init__(self, save_dir: Path, file_name: str, prompt_template: str, record_examples=200):
        save_dir.mkdir(parents=True, exist_ok=True)
        f = open(save_dir / file_name, mode='w')
        f.write(self.__title_row("Prompt Template"))
        f.write(prompt_template)
        f.write('\n\n')

        self.f = f
        self.example_counter = 0
        self.record_examples = record_examples

    def record(self, prompt: str, decode_str: str):
        if self.example_counter <= self.record_examples:
            generate_str = decode_str[len(prompt):]
            self.f.write(self.__title_row(f'Example {self.example_counter}'))
            self.f.write(prompt)
            self.f.write(self.__title_row('Generate'))
            self.f.write(generate_str)
            self.f.write('\n\n')
            self.example_counter += 1
            self.f.flush()
        else:
            self.f.close()

