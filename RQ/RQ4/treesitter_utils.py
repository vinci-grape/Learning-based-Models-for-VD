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



# tree = parser.parse(bytes(Test_CPP,encoding='utf-8'))
# for node in __traverse_tree(tree):
#     # print(dir(node))
#     # print(f"[{node.type}]:{node.text}")
#     if node.type == "binary_expression":
#         # print(node)
#         # print(dir(node))
#         print(node.child_by_field_name("operator"))
#         print(node.children_by_field_name("operator"))
#         print(node.text)

