from tree_sitter import Language, Parser
from .parser_folder import remove_comments_and_docstrings,tree_to_token_index,index_to_code_token, DFG_java


LANGUAGE = Language('/home/liwei/Code-Watermark/variable-watermark/resources/my-languages.so', 'java')
parser = Parser()
parser.set_language(LANGUAGE)
lang = 'java'


def extract_dataflow(code):
    code = code.replace("\\n", "\n")
    try:
        code = remove_comments_and_docstrings(code, 'java')
    except:
        pass
    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')

    code_tokens = [index_to_code_token(x, code) for x in tokens_index]
    index_to_code = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_to_code[index] = (idx, code)

    index_table = {}
    for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
        index_table[idx] = index

    DFG, _ = DFG_java(root_node, index_to_code, {})

    DFG = sorted(DFG, key=lambda x: x[1])
    return DFG, index_table, code_tokens

def is_valid_variable_java(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in java_keywords:
        return False
    elif name in java_special_ids:
        return False
    return True

def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

def get_identifiers(code):
    dfg, index_table, code_tokens = extract_dataflow(code)
    ret = []
    for d in dfg:
        if is_valid_variable_java(d[0]):
            ret.append(d[0])
    ret = unique(ret)
    ret = [[i] for i in ret]
    return ret, code_tokens


java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while"]
java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Long", "Float", "Double", "Character",
                    "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object", "String", "StringBuffer",
                    "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet", "HashMap"]
