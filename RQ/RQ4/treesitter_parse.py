from treesitter_utils import traverse_source_code, Test_CPP
from functools import cmp_to_key
from dataclasses import dataclass
from functools import reduce


@dataclass
class StatementAttention:
    type: str
    code: str
    attention: float
    start_pos: tuple
    end_pos: tuple


# (statement_type, code , attention , start_pos, end_pos)

def merge_pos(pos: list):
    def cmp_point(a, b):
        if a[0] == b[0]:
            if a[1] <= b[1]:
                return -1
        if a[0] < b[0]:
            return -1
        return 1

    return list(sorted(pos, key=cmp_to_key(cmp_point)))


def merge_two_node_pos(node_A, node_B):
    pos = merge_pos([node_A.start_point, node_A.end_point, node_B.start_point, node_B.end_point])
    return pos[0], pos[-1]


def merge_nodes_pos(nodes):
    pos_list = []
    for node in nodes:
        pos_list.append(node.start_point)
        pos_list.append(node.end_point)
    pos = merge_pos(pos_list)
    return pos[0], pos[-1]


def get_attention(pos, token_level_attentions: list[tuple]):
    start_pos, end_pos = pos
    start_row , end_row = start_pos[0] , end_pos[0]
    start_col , end_col = start_pos[1] , end_pos[1]
    attention_score = 0.0
    for cur_row in range(start_row,end_row + 1):
        line_attention = token_level_attentions[cur_row]
        if cur_row == start_row and start_row == end_row: # same line
            cur_col = 0
            for token,attention in line_attention:
                cur_end_col = cur_col + len(token)
                if cur_end_col <= start_col or end_col <= cur_col:
                    pass
                else:
                    # print(f'[{cur_col} - {cur_end_col}] {token} , {attention}')
                    attention_score += attention
                cur_col = cur_end_col
        elif cur_row == start_row:
            cur_col = 0
            for token,attention in line_attention:
                cur_end_col = cur_col + len(token)
                if cur_end_col > start_col:
                    # print(f'[{cur_col} - {cur_end_col}] {token} , {attention}')
                    attention_score += attention
                cur_col = cur_end_col
        elif cur_row == end_row:
            cur_col = 0
            for token,attention in line_attention:
                cur_end_col = cur_col + len(token)
                if cur_col < end_col:
                    # print(f'[{cur_col} - {cur_end_col}] {token} , {attention}')
                    attention_score += attention
                cur_col = cur_end_col
        else:   #   start_row < cur_row < end_row
            for token,attention in line_attention:
                attention_score += attention
    return attention_score


def decode_node(node):
    return node.text.decode()


def parse_source_code_statement_type(code: str, token_level_attentions: list[tuple]):
    # 'child_by_field_id', 'child_by_field_name', 'child_count', 'children', 'children_by_field_id', 'children_by_field_name',
    # 'end_byte', 'end_point', 'field_name_for_child', 'has_changes', 'has_error', 'id', 'is_missing', 'is_named', 'named_child_count',
    # 'named_children', 'next_named_sibling', 'next_sibling', 'parent', 'prev_named_sibling', 'prev_sibling', 'sexp', 'start_byte', 'start_point', 'text', 'type', 'walk'
    result: list[StatementAttention] = []
    for node in traverse_source_code(code):
        # print(node.type, node.text)
        type = node.type
        pos = (node.start_point, node.end_point)

        if type == "if_statement":  # If Statement
            # if + condition_clause
            if_node = node.children[0]
            condition_clause_node = node.child_by_field_name('condition')
            pos = merge_two_node_pos(if_node, condition_clause_node)
            result.append(StatementAttention("If Statement", f"if{decode_node(condition_clause_node)}",
                                             get_attention(pos, token_level_attentions), pos[0], pos[1]))
        elif type == "for_statement":  # For Statement
            for_children = list(
                filter(lambda x: x.type != 'compound_statement', node.children))  # filter out `{ xxxx }`
            for_code = reduce(lambda x, y: f'{x}{decode_node(y)}', for_children, "")  # `for(int i =0 ; i<= 10 ; i++)`
            for_pos = merge_nodes_pos(for_children)
            result.append(StatementAttention("For Statement", for_code, get_attention(for_pos, token_level_attentions),
                                             for_pos[0], for_pos[1]))
        elif type in ["while_statement", "do_statement"]:  # While Statement
            print('============================ while_statement ===============')

            while_children = list(filter(lambda x: x.type != 'compound_statement' and x.type != 'do',
                                         node.children))  # filter out `{ xxxx }` and `do`
            while_code = reduce(lambda x, y: f'{x}{decode_node(y)}', while_children,
                                "")  # `for(int i =0 ; i<= 10 ; i++)`
            while_pos = merge_nodes_pos(while_children)
            result.append(
                StatementAttention("While Statement", while_code, get_attention(while_pos, token_level_attentions),
                                   while_pos[0], while_pos[1]))
        elif type in ['break_statement', 'continue_statement', 'goto_statement']:  # Jump Statement
            result.append(
                StatementAttention("Jump Statement", decode_node(node),
                                   get_attention((node.start_point, node.end_point), token_level_attentions),
                                   node.start_point, node.end_point))
        elif type in ['switch_statement']:  # Switch Statement ,

            switch_children = list(
                filter(lambda x: x.type != 'compound_statement', node.children))  # filter out `{ xxxx }`
            switch_code = reduce(lambda x, y: f'{x}{decode_node(y)}', switch_children,
                                 "")  # `for(int i =0 ; i<= 10 ; i++)`
            switch_pos = merge_nodes_pos(switch_children)
            result.append(
                StatementAttention("Switch Statement", switch_code, get_attention(switch_pos, token_level_attentions),
                                   switch_pos[0], switch_pos[1]))
        elif type in ['case_statement']:  # Case Statement
            case_children = list(
                filter(lambda x: x.type in ['case', 'default', 'value'], node.children))  # filter out `{ xxxx }`
            case_value_node = node.child_by_field_name('value')
            if case_value_node is not None:
                case_children.append(case_value_node)
            case_code = reduce(lambda x, y: f'{x}{decode_node(y)}', case_children, "")
            case_pos = merge_nodes_pos(case_children)
            result.append(
                StatementAttention("Case Statement", case_code, get_attention(case_pos, token_level_attentions),
                                   case_pos[0], case_pos[1]))
        elif type in ['return_statement']:  # Return Statement
            result.append(
                StatementAttention("Return Statement", decode_node(node), get_attention(pos, token_level_attentions),
                                   pos[0], pos[1]))
        elif type in ['binary_expression']:
            binary_expression_type = node.child_by_field_name('operator').type
            statement_type : str = None
            if binary_expression_type in ['+' , '-' , '*' , '/' ,'%']:  # Arithmetic Operation
                statement_type = 'Arithmetic Operation'
            elif binary_expression_type in ['==' , '!=' , '>' , '<' ,'>=' ,'<=' ]: # Relational Operation
                statement_type = 'Relational Operation'
            elif binary_expression_type in ['&&' , '||' ]: # Logical Operation
                statement_type = 'Logical Operation'
            elif binary_expression_type in ['&' , '|' , '^' , '<<' , '>>' ] : # Bitwise Operation
                statement_type = 'Bitwise Operation'

            if statement_type is not None:
                result.append(
                    StatementAttention(statement_type, decode_node(node), get_attention(pos, token_level_attentions),
                                       pos[0], pos[1]))
        elif type in ['assignment_expression']: # Assignment Operation
            result.append(
                StatementAttention('Assignment Operation', decode_node(node), get_attention(pos, token_level_attentions),
                                   pos[0], pos[1]))
        elif type in ['call_expression']: # Function Call
            result.append(
                StatementAttention('Function Call', decode_node(node), get_attention(pos, token_level_attentions),
                                   pos[0], pos[1]))
        elif type in ['field_expression']: # Field Expression
            print('============================ field_expression ===============')

            result.append(
                StatementAttention('Field Expression', decode_node(node), get_attention(pos, token_level_attentions),
                                   pos[0], pos[1]))
        elif type in ['declaration']: # Declaration Statement
            result.append(
                StatementAttention('Declaration Statement', decode_node(node), get_attention(pos, token_level_attentions),
                                   pos[0], pos[1]))

    # for r in result:
    #     print(r)
    return result


def filter_statement_on_given_lines(line_numbers:list[int] , statements:list[StatementAttention]):
    res_statements = []
    for statement in statements:
        if len(set(range(statement.start_pos[0] ,statement.end_pos[0] + 1)) & set(line_numbers)) != 0:
            res_statements.append(statement)
    return res_statements

