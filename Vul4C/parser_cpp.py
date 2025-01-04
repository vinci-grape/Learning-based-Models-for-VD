from tree_sitter import Language
from typing import Union, Tuple
from termcolor import colored
from utils import debug, FunctionSignatureError, write_dataclasses_to_json, ExtractedFunction
import os
import subprocess
import re
from parser_c_like import ParserCLike

os.environ['PATH'] += os.pathsep + '/usr/local/node/bin' + os.pathsep + '/usr/local/tree-sitter'
os.system('cd tree-sitter-cpp && tree-sitter generate')

Language.build_library(
    "build/my-languages.so",
    [
        "tree-sitter-c",
        "tree-sitter-cpp",
    ]
)

CPP_LANGUAGE = Language("build/my-languages.so", "cpp")


class ParserCPP(ParserCLike):

    def __init__(self, file_path: str) -> None:
        super().__init__(file_path, CPP_LANGUAGE)

    def traverse_function_definition(self, func_node):
        cursor = func_node.walk()
        debug(func_node.text)
        debug(func_node.start_point, func_node.end_point)

        # in CPP the function return type may be `null` in class constructor
        return_type_node = cursor.node.child_by_field_name('type')
        return_type = None
        if return_type_node is not None:
            return_type = self.node_split_from_file(return_type_node)
        cursor.goto_first_child()

        func_name = None
        parameter_list_signature = None
        destruction_ptr_return_type_level = 0
        parameter_list_node = None

        # 'child_by_field_id', 'child_by_field_name', 'child_count', 'children',
        # 'children_by_field_id', 'children_by_field_name', 'end_byte', 'end_point',
        # 'field_name_for_child', 'has_changes', 'has_error', 'id', 'is_missing',
        # 'is_named', 'named_child_count', 'named_children', 'next_named_sibling',
        # 'next_sibling', 'parent', 'prev_named_sibling', 'prev_sibling', 'sexp',
        # 'start_byte', 'start_point', 'text', 'type', 'walk']

        while True:
            node_type = cursor.node.type

            if node_type == 'pointer_declarator':
                destruction_ptr_return_type_level += 1
                while True:
                    cursor.goto_first_child()
                    if cursor.node.type != 'pointer_declarator':
                        break
                    destruction_ptr_return_type_level += 1

            if node_type == 'function_declarator':
                cursor.goto_first_child()
                while True:
                    declarator_type = cursor.node.type
                    if declarator_type == 'identifier':
                        func_name = self.node_split_from_file(cursor.node)
                    elif declarator_type in ['qualified_identifier', 'destructor_name', 'field_identifier',
                                             'operator_name']:
                        # e.g. NameSpace::Scanner::foo
                        func_name = self.node_split_from_file(cursor.node)
                    elif declarator_type == 'template_function':
                        # e.g. bool foo<A>
                        func_name = self.node_split_from_file(cursor.node.child_by_field_name('name'))
                    elif declarator_type == 'parameter_list':
                        parameter_list_node = cursor.node
                        parameter_list_signature = self.node_split_from_file(cursor.node)
                    if not cursor.goto_next_sibling():
                        break
                cursor.goto_parent()

            if not cursor.goto_next_sibling():
                break

        if parameter_list_node is None:
            debug('[ERROR] this function missing parameter list [SKIP]')
            return

        parameter_list = []
        debug(f'begin process {func_name}')
        parameter_list_cursor = parameter_list_node.walk()
        parameter_list_cursor.goto_first_child()
        while True:
            # print('=================')
            # print(parameter_list_cursor.node.type,parameter_list_cursor.node.text)
            if parameter_list_cursor.node.type == 'parameter_declaration':
                # extract all parameter
                parameter_list.append(self.traverse_parameter_declaration(parameter_list_cursor.node))
            elif parameter_list_cursor.node.type == 'variadic_parameter_declaration':
                # int main(...)   int main(Tail...)
                parameter_list.append(self.traverse_variadic_parameter_declaration(parameter_list_cursor.node))
            elif parameter_list_cursor.node.type == 'optional_parameter_declaration':
                parameter_list.append(self.traverse_optional_parameter_declaration(parameter_list_cursor.node))

            if not parameter_list_cursor.goto_next_sibling():
                break

        compose_signature = ''
        for i, p in enumerate(parameter_list):
            if i > 0:
                compose_signature += ','
            if p[1] is None:
                compose_signature += p[0]
            else:
                insert_idx = p[2]
                compose_signature += p[0][:insert_idx] + p[1] + p[0][insert_idx:]
            if len(p) == 4:  # default parameter default value
                compose_signature += p[3]
        compose_signature = f'({compose_signature})'

        if destruction_ptr_return_type_level != 0:
            return_type += '*' * destruction_ptr_return_type_level

        if type(parameter_list_signature) is str:
            parameter_list_signature = [parameter_list_signature]

        if type(parameter_list_signature) is list:  # the parameter_list_signature may be span multilines, so the returned parameter_list_signature is `list` type
            parameter_list_signature = list(map(lambda x: ParserCPP.remove_comments(x), parameter_list_signature))
            parameter_list_signature = ''.join(parameter_list_signature)

            # remove multiline comment again
            parameter_list_signature = ParserCPP.remove_comments(parameter_list_signature)

        parameter_list_signature = parameter_list_signature.replace('\n', '').replace('\t', ' ').replace('\\', '')
        parameter_list_signature = ' '.join(
            parameter_list_signature.split())  # remove multiple whitespace with single whitespace

        if func_name is None:
            # raise FunctionSignatureError(f'[Function Name is None] {func_node.text}')
            debug('[Function Name is None] this function missing function name')
            return
        elif type(func_name) is list:
            func_name = ''.join(func_name)

        if return_type is None:
            return_type = func_name
            if return_type.startswith('~'):  # destructor name
                return_type = ''

        func = self.node_split_from_file(func_node)

        # parameter_list_signature = parameter_list_signature[1:-1] # remove bracket
        debug('=' * 80)
        debug(f'function name  : {func_name}')
        debug(f'parameters sig : {parameter_list_signature}')
        debug(f'parameters     : {parameter_list}')
        debug(f'return type    : {return_type}')
        debug('=' * 80)
        assert func_name is not None and parameter_list_signature is not None and return_type is not None

        if compose_signature.replace(' ', '') != parameter_list_signature.replace(' ', ''):
            # raise FunctionSignatureError(
            #     f'[Signature check error] [{compose_signature}] != [{parameter_list_signature}]')
            debug("f'[Signature check error] [{compose_signature}] != [{parameter_list_signature}]')")
            return

        return ExtractedFunction(func_name, parameter_list_signature, parameter_list, return_type,
                                 func)

    def traverse_optional_parameter_declaration(self, pd_cursor_node) -> Tuple[str, str, int, str]:
        pd_cursor = pd_cursor_node.walk()
        pd_start_point, pd_end_point = pd_cursor.node.start_point, pd_cursor.node.end_point
        pd_lines = self.split_from_file(pd_start_point, pd_end_point)
        debug(pd_lines)
        id_node = pd_cursor_node.child_by_field_name('declarator')
        id = None
        if id_node is not None:
            while id_node.type == 'pointer_declarator':
                id_node = id_node.children[1]
            id = self.node_split_from_file(id_node)
            pd_lines = self.multiline_replace(pd_lines,
                                              ParserCPP.cal_relative_point(pd_start_point, id_node.start_point), id)

        pd_lines = ''.join(pd_lines).replace('\n', '').replace('\t', ' ')
        pd_lines = ParserCPP.remove_comments(pd_lines)
        default_value = None
        if (re_res := re.search(r'=.*', pd_lines)) is not None:
            default_value = re_res.group()
        type = pd_lines.replace(self.SPECIAL_IDENTIFIER, '').replace(default_value, '')
        type = type.strip()
        debug('default parameter', type, id)
        assert default_value is not None
        return type, id, pd_lines.index(self.SPECIAL_IDENTIFIER) if id is not None else -1, default_value

    def traverse_variadic_parameter_declaration(self, pd_cursor_node) -> Tuple[str, str, int]:
        if ''.join(self.node_split_from_file(pd_cursor_node)).replace(' ', '') == '...':
            return '...', None, -1
        pd_cursor = pd_cursor_node.walk()
        pd_start_point, pd_end_point = pd_cursor.node.start_point, pd_cursor.node.end_point
        pd_lines = self.split_from_file(pd_start_point, pd_end_point)

        id = None
        found_identifier = False
        for node in self.traverse_cursor(pd_cursor):
            if node.type == 'identifier':
                id = self.split_from_file_maybe_flatten(node.start_point, node.end_point)  # must be str ,not list
                id_start = ParserCPP.cal_relative_point(pd_start_point, node.start_point)
                found_identifier = True

        if found_identifier:
            pd_lines = ParserCPP.multiline_replace(pd_lines, id_start, id)
        else:
            pass

        pd_lines = ''.join(pd_lines).replace('\n', '').replace('\t', ' ')
        insert_index = -1
        if found_identifier:
            insert_index = pd_lines.index(ParserCPP.SPECIAL_IDENTIFIER)
            type = pd_lines.replace(ParserCPP.SPECIAL_IDENTIFIER, '', 1)
        else:
            type = pd_lines

        type = type.strip()
        return type, id, insert_index

    def traverse_parameter_declaration(self, pd_cursor_node) -> Tuple[str, str, int]:
        pd_cursor = pd_cursor_node.walk()
        pd_start_point, pd_end_point = pd_cursor.node.start_point, pd_cursor.node.end_point
        pd_lines = self.split_from_file(pd_start_point, pd_end_point)
        debug(pd_lines)
        debug('ggg', pd_start_point, pd_end_point)
        id = None
        found_identifier = False
        for node in self.traverse_cursor(pd_cursor):
            if node.type == 'identifier':
                # find the last identifier.  we me face `char __user *optval` , __user will be recognized as identifier
                assert node.start_point[0] == node.end_point[0]
                id = self.split_from_file_maybe_flatten(node.start_point, node.end_point)  # must be str ,not list
                id_start = ParserCPP.cal_relative_point(pd_start_point, node.start_point)
                debug('id_start', id_start)
                found_identifier = True
            elif node.type == 'parameter_list':
                break

        if found_identifier:
            pd_lines = ParserCPP.multiline_replace(pd_lines, id_start, id)
        else:
            pass

        # merge pd_lines to one line
        pd_lines = ''.join(pd_lines).replace('\n', '').replace('\t', ' ')
        pd_lines = ParserCPP.remove_comments(pd_lines)
        pd_lines = ' '.join(pd_lines.split())
        debug(pd_lines)

        insert_index = -1
        if found_identifier:
            insert_index = pd_lines.index(ParserCPP.SPECIAL_IDENTIFIER)
            type = pd_lines.replace(ParserCPP.SPECIAL_IDENTIFIER, '', 1)
        else:
            type = pd_lines

        type = type.strip()
        debug(type, id, insert_index)
        assert type is not None
        return type, id, insert_index

    def _extract_function(self, tree):
        func_nodes = []
        for i in self.traverse_tree(tree):
            if i.type == 'function_definition':
                # start_line = i.start_point[0]
                # end_line = i.end_point[0]
                # func_anchors.append((start_line,end_line))
                func_nodes.append(i)

        # if a function nested in a function definition , filter it
        result_func_nodes = []
        extracted_funcs = []
        for i, s in enumerate(func_nodes):
            need_add = True
            for j, other_s in enumerate(func_nodes):
                if i != j:
                    if other_s.start_point[0] <= s.start_point[0] and other_s.end_point[0] >= s.end_point[0]:
                        need_add = False
            if need_add:
                result_func_nodes.append(s)
        debug(colored(f'init found:[{len(func_nodes)}] , after filter:[{len(result_func_nodes)}]', 'red'))
        for node in result_func_nodes:
            func = self.traverse_function_definition(node)
            if func is not None:
                extracted_funcs.append(func)
            # result_func_str.append(''.join(split_function_from_file(anchor[0],anchor[1])))

        self.save_extracted_func(extracted_funcs)

        # with open('split_func.txt',mode='w') as f:
        #     f.write('\n\n'.join(result_func_str))

    def begin_extract(self):
        return self._extract_function(self.tree)

    @staticmethod
    def extract_function(file_path: str) -> bool:
        try:
            ParserCPP(file_path).begin_extract()
            return True
        except FunctionSignatureError as e:
            print('[[[FunctionSignatureError]]]!!!!!!', e.msg)
        return False


# ParserCPP.extract_function(
# 'cache/download/michaelrsweet/htmldoc/369b2ea1fd0d0537ba707f20a2f047b6afd2fbdc/369b2ea1fd0d0537ba707f20a2f047b6afd2fbdc/ps-pdf.cxx')
# ParserCPP.extract_function(
#     'cache/download/binutils-gdb/4581a1c7d304ce14e714b27522ebf3d0188d6543/4581a1c7d304ce14e714b27522ebf3d0188d6543/coffcode.h')
