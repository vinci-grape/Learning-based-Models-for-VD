import dataclasses
from pathlib import Path
from tree_sitter import Language,Parser
import re
from typing import Union
from utils import write_dataclasses_to_json, debug


class ParserCLike:
    def __init__(self,file_path: str,language) -> None:
        self.file_path = file_path
        # file_b = open(file_path, mode='rb').read()

        self.file_lines = open(file_path, mode='r').readlines()
        # remove comment
        remove_comment_src = self.replace_comments_with_whitespace(''.join(self.file_lines)).splitlines(True)
        # remove preproc
        file_b = self.remove_preproc(remove_comment_src)
        parser = Parser()
        parser.set_language(language)
        self.parser = parser
        self.tree = self.parser.parse(bytes(file_b,encoding='utf-8'))

    def remove_preproc(self,code):
        """  remove preprocessor to temporarily solve https://github.com/tree-sitter/tree-sitter-c/issues/70 """
        res_line = []

        new_code = []
        if_nif_stack = []
        nested_level = 0
        encounter_else_stack = []
        # remove `else`,`elif` block content
        for line in code:
            # print(line,end='')
            strip_line = line.strip()
            if  re_res := re.match(r'^#[ \t]*(?P<macro>ifdef|ifndef|else|elif|endif|if)', strip_line):
                macro = re_res.group('macro')
                if macro in ['ifdef','ifndef','if']:
                    nested_level += 1
                    if_nif_stack.append(nested_level)
                elif macro in ['else','elif']:
                    if (len(encounter_else_stack) > 0 and encounter_else_stack[-1] != nested_level) or len(encounter_else_stack) == 0 :
                        encounter_else_stack.append(nested_level)
                elif macro in ['endif']:
                    previous_level =  if_nif_stack.pop()
                    nested_level -= 1
                    if len(encounter_else_stack) > 0 and encounter_else_stack[-1] == previous_level:
                        encounter_else_stack.pop()

            if len(encounter_else_stack) > 0 and encounter_else_stack[-1] == nested_level:
                new_code.append(' ' * (len(line) - 1) + '\n')
            else:
                new_code.append(line)

        debug('**********remove else block********* ')
        for line in new_code:
            debug(line,end='')

        code = new_code
        # remove macro
        for line in code:
            debug(line,end='')
            strip_line = line.strip()
            if re_res:=re.match(r'^#[ \t]*(ifdef|ifndef|else|elif|endif|if)',strip_line):
                res_line.append(' ' * (len(line) - 1) + '\n')
            else:
                res_line.append(line)
        assert len(res_line[-1]) == len(line)
        debug('\n')
        debug('*'*20)
        for line in res_line:
            debug(line,end='')
        debug('\n')
        return ''.join(res_line)


    def split_function_from_file(self,start_line, end_line):
        func = self.file_lines[start_line:end_line + 1]
        return func

    def split_from_file(self,start_point: tuple, end_point: tuple):
        content = self.file_lines[start_point[0]:end_point[0] + 1]
        if len(content) == 1:
            content[0] = content[0][start_point[1]:end_point[1]]
        else:
            content[0] = content[0][start_point[1]:]
            content[-1] = content[-1][:end_point[1]]
        return content

    def split_from_file_maybe_flatten(self,start_point: tuple, end_point: tuple):
        res = self.split_from_file(start_point, end_point)
        return res[0] if len(res) == 1 else res

    def node_split_from_file(self,node) -> Union[str, list[str]]:
        res = self.split_from_file(node.start_point, node.end_point)
        return res[0] if len(res) == 1 else res

    def traverse_cursor(self,cursor):
        reached_root = False
        while reached_root == False:
            yield cursor.node

            if cursor.goto_first_child():
                continue

            if cursor.goto_next_sibling():
                continue

            retracing = True
            while retracing:
                if not cursor.goto_parent():
                    retracing = False
                    reached_root = True

                if cursor.goto_next_sibling():
                    retracing = False

    def traverse_tree(self,tree):
        cursor = tree.walk()
        return self.traverse_cursor(cursor)

    @staticmethod
    def cal_relative_point(start_point,point):
        if start_point[0] == point[0]: # save row
            return point[0]-start_point[0] ,point[1] - start_point[1]
        else:
            return point[0]-start_point[0] , point[1]


    SPECIAL_IDENTIFIER = '$$$$$ID$$$$$'

    @staticmethod
    def multiline_replace(lines:list[str], start_point:tuple, replace_id:str):
        prefix_line = lines[:start_point[0]]
        postfix_line = lines[start_point[0] + 1:]
        modify_line = lines[start_point[0]]
        modify_line = modify_line[:start_point[1]] + modify_line[start_point[1]:].replace(replace_id,ParserCLike.SPECIAL_IDENTIFIER,1)  # replace one time
        lines = prefix_line + [modify_line] + postfix_line
        return lines

    @staticmethod
    def replace_comments_with_whitespace(csrc):
        regex = r'(\".*?\"|\'.*?\')|(/\*.*?\*/|//[^\r\n]*$)'
        old_len = len(csrc)
        def replace_block_comment(x:re.Match) -> str:
            if x.group(2) is not None:
                match = x.group()
                replace_str = list(' ' * len(match))
                for idx, c in enumerate(match):
                    if c == '\n':
                        replace_str[idx] = '\n'
                replace_str = ''.join(replace_str)
                return replace_str
            else:
                return x.group(1)

        csrc = re.sub(regex, replace_block_comment,csrc, flags=re.MULTILINE|re.DOTALL)

        assert old_len == len(csrc)

        return csrc

    @staticmethod
    def remove_block_comments(csrc):
        '''Remove block comments from a c source string - /* */'''
        regex = r'/\*.*?\*/'
        matches = re.findall(regex, csrc, re.DOTALL)
        for match in matches:
            csrc = csrc.replace(match, '')

        return csrc

    @staticmethod
    def remove_singleline_comments(csrc):
        '''Remove single line comments from a c source string - //'''
        regex = r'//.*$'
        csrc = re.sub(regex, '', csrc, flags=re.MULTILINE)
        return csrc

    @staticmethod
    def remove_comments(csrc):
        '''Remove comments from a c source file'''
        content = csrc[:]
        content = ParserCLike.remove_block_comments(content)
        content = ParserCLike.remove_singleline_comments(content)
        return content

    def save_extracted_func(self,extracted_funcs:list):
        # print(self.file_path)
        # save in processed directory
        func_save_dir = f'processed/{"/".join(self.file_path.split("/")[1:]) }'
        # print(func_save_dir)
        # print(Path(func_save_dir).parent)
        write_dataclasses_to_json(extracted_funcs,Path(func_save_dir))


    def debug(self):
        for node in self.traverse_tree(self.tree):
            debug(node.type, node.text)
