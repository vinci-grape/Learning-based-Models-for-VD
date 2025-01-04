from collections import ChainMap
from typing import Tuple

from tree_sitter import Language,Parser

CPP_LANGUAGE = Language("../../Vul4C/build/my-languages.so", "cpp")
parser = Parser()
parser.set_language(CPP_LANGUAGE)


def __traverse_cursor(cursor):
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

def __traverse_tree(tree):
    cursor = tree.walk()
    return __traverse_cursor(cursor)


def traverse_source_code(source_code:str):
    tree = parser.parse(bytes(source_code, encoding='utf-8'))
    return __traverse_tree(tree)


def get_function_body_pos(source_code:str):
    for i in traverse_source_code(source_code):
        if i.type == 'function_definition':
            body = i.child_by_field_name("body")
            if body is None:
                return None,None
            return body.start_point , body.end_point
    return None , None





class CodeAbstracter:


    def __init__(self, add_prefix_symbol = False):
        """
        :param string_number_literal_need_abstract: when `True`, abstract string and number literal as well.
        :param add_prefix_symbol: when `True`, abstracted symbol will add prefix `$`
        """
        C_LANGUAGE = Language("../../Vul4C/build/my-languages.so", "c")
        CPP_LANGUAGE = Language("../../Vul4C/build/my-languages.so", "cpp")
        self.parser_c = self.get_parser(C_LANGUAGE)
        self.parser_cpp = self.get_parser(CPP_LANGUAGE)
        self.prefix_symbol = '$' if add_prefix_symbol else ''

    def get_parser(self,language):
        parser = Parser()
        parser.set_language(language)
        return parser

    def abstract_func(self,func:str):
        return self.abstract_code(func,'CPP')

    def abstract_code(self, code:str ,language:str) -> Tuple[list[str],dict] :
        assert language in ['C','CPP']
        code_bytes=  bytes(code,encoding='utf-8')
        tree =  self.parser_c.parse(code_bytes)  if language == 'C' else  self.parser_cpp.parse(code_bytes)

        position_map = { }  # row -> list[(col,str,replace_whitespace?)]
        var_map = { }
        id_class_map = {
            'VAR': var_map,
        }
        global_internal_id_cnt = 0   # internal id

        def multiline_str_abstract(node, content_to_id_record_map, multiline_content_map, abstract_type:str):
            nonlocal global_internal_id_cnt,position_map
            assert abstract_type in ['STR', 'COMMENT']
            node_text: str = node.text.decode('utf-8')
            final_abstract_symbol = f"{self.prefix_symbol}{abstract_type}_{len(multiline_content_map)}"
            if abstract_type == 'COMMENT':
                final_abstract_symbol = f"/* {self.prefix_symbol}{abstract_type}_{len(multiline_content_map)} */"
            if len(node_text.splitlines()) > 1:  # multiline string literal
                if node_text in content_to_id_record_map:
                    global_id = content_to_id_record_map[node_text]
                else:
                    global_internal_id_cnt += 1
                    global_id = global_internal_id_cnt
                    content_to_id_record_map[node_text] = global_id

                node_texts = node_text.splitlines(keepends=False)
                start_row: int = node.start_point[0]
                for row in range(start_row, start_row + len(node_texts)):  # row number
                    position_map.setdefault(row, [])
                    col = node.start_point[1] if row == start_row else 0
                    node_text = node_texts[row - start_row]
                    key = f'{node_text}$${global_id}'
                    if row == start_row:
                        # for multiline string literal, we only abstract first line, rest of lines are replaced with whitespace
                        if key not in multiline_content_map:
                            multiline_content_map[key] = final_abstract_symbol
                        position_map[row].append((col, node_text, abstract_type, global_id))
                    else:
                        # rest of lines
                        position_map[row].append((col, node_text, abstract_type, global_id, True))
            else:  # one line string literal
                if node_text in content_to_id_record_map:
                    global_id = content_to_id_record_map[node_text]
                else:
                    global_internal_id_cnt += 1
                    global_id = global_internal_id_cnt
                    content_to_id_record_map[node_text] = global_id

                key = f'{node_text}$${global_id}'
                if key not in multiline_content_map:
                    multiline_content_map[key] = final_abstract_symbol
                row = node.start_point[0]
                col = node.start_point[1]
                position_map.setdefault(row, [])
                position_map[row].append((col, node_text, abstract_type, global_id))

        for node in self.traverse_tree(tree):
            # print(dir(node))
            # print('==================')
            # print(node.text,node.type)
            # if node.parent:
            #     print(node.parent,node.parent.type)
            if node.type in ['identifier']:
                if node.parent and node.parent.type not in ['function_declarator','qualified_identifier']:   # filter funciton name
                    node_text = node.text.decode('utf-8')
                    id_class = 'VAR'
                    if node.parent.type in ['call_expression','preproc_function_def']:   # function id
                        continue

                    save_map = id_class_map[id_class]

                    if (node_text not in save_map ) and ((id_class != 'NUMBER') ):
                        global_internal_id_cnt += 1
                        save_map[node_text] = f"{self.prefix_symbol}{id_class}{len(save_map)}"
                    row , col = node.start_point

                    if id_class != 'NUMBER' or (id_class == 'NUMBER'):
                        position_map.setdefault(row,[])
                        position_map[row].append((col,node_text,id_class,-1))
                    assert node.start_point[0] == node.end_point[0]  , "[ERROR] identifier not in one line"
            # print()

        new_line_of_code = []
        intersect_symbol = var_map.keys()
        # merge_map = ChainMap(var_map,func_map,type_map,string_map,number_map,field_map)
        # print(merge_map)
        # print(id_class_map)
        for line,line_of_code in enumerate(code.splitlines(keepends=True)):
            if line not in position_map:
                new_line_of_code.append(line_of_code)
                continue

            items_in_line = position_map[line]
            for item_idx,item in enumerate(items_in_line):
                replace_with_whitespace = False
                if len(item) == 5 and item[4] == True:
                    replace_with_whitespace = True
                # print(item)
                col,text = item[0],item[1]
                map_type = item[2]
                global_id = item[3]
                key = text if map_type not in ['STR','COMMENT'] else f'{text}$${global_id}'
                if not replace_with_whitespace:
                    save_map = id_class_map[map_type]
                    line_of_code = line_of_code[:col] + line_of_code[col:].replace(text, save_map[key], 1)

                    offset = len(save_map[key]) - len(text)

                    new_tail_items = []
                    for tail_item in items_in_line[item_idx + 1:]:
                        tail_item = tail_item[0] + offset, *tail_item[1:]
                        new_tail_items.append(tail_item)
                    items_in_line[item_idx + 1:] = new_tail_items
                else:
                    line_of_code = line_of_code[:col] + ' ' * len(text) + line_of_code[col + len(text):]

            new_line_of_code.append(line_of_code)



        # print(string_id_2_string_map)
        # print(var_map)
        # print(func_map)
        # print(self.dict_kv_change(merge_map))
        symbol_table = [self.dict_kv_change(x) for x in id_class_map.values() ]
        symbol_table = dict(ChainMap(*symbol_table))
        return  new_line_of_code , symbol_table

    def dict_kv_change(self,dict):
        return { v:k  for k,v in dict.items() }
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



