import re

p = re.compile('^[a-zA-Z_$][a-zA-Z_$0-9]*$')

java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while", "long", "this"]
java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Float", "Double",
                    "Character",
                    "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object", "String",
                    "StringBuffer",
                    "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set", "Queue", "ArrayList", "HashSet",
                    "HashMap", "Long"]


def is_valid(var: str) -> bool:
    if p.match(var) is not None and var not in java_keywords and var not in java_special_ids:
        return True
    else:
        return False
