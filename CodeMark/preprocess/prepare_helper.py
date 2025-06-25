class Spliter:
    def __init__(self, lang):
        from tree_sitter import Language, Parser

        self.lang = lang
        LANGUAGE = Language('/home/liwei/Code-Watermark/variable-watermark/resources/my-languages.so', lang)
        self.parser = Parser()
        self.parser.set_language(LANGUAGE)

        self.split_positions: dict = None

    def add_space_to_identifier(self, func):
        self.split_positions = {}
        if self.lang == 'java':
            func = 'public class A { \n' + func + ' } \n'
        tree = self.parser.parse(bytes(func, 'utf8'))
        self.get_identifier_begin_end(tree.root_node)

        code_tokens = []
        for line_id, code_line in enumerate(func.split('\n')):
            line_spilt_positions = self.split_positions.get(line_id, None)
            if line_spilt_positions is None:
                code_tokens.append(code_line)
            else:
                # print('before', line_spilt_positions)
                if line_spilt_positions[0] != 0:
                    line_spilt_positions.insert(0, 0)
                if line_spilt_positions[-1] != len(code_line):
                    line_spilt_positions.insert(len(line_spilt_positions), len(code_line))

                # print('modify', line_spilt_positions)
                for i in range(len(line_spilt_positions) - 1):
                    begin = line_spilt_positions[i]
                    end = line_spilt_positions[i + 1]
                    token = code_line[begin:end]
                    if token.endswith(' '):
                        code_tokens.append(token)
                    else:
                        code_tokens.append(token + ' ')

            code_tokens.append('\n')

        func = ''.join(code_tokens)
        if self.lang == 'java':
            func = func[len('public class A  {'):]
            func = func[:func.rindex('}')]
        return func

    def get_identifier_begin_end(self, node):
        if node.type == 'identifier':
            assert node.start_point[0] == node.end_point[0]
            line_id = node.start_point[0]
            line_spilt_positions = self.split_positions.get(line_id, None)
            if line_spilt_positions is None:
                line_spilt_positions = []
            line_spilt_positions.append(node.start_point[1])
            line_spilt_positions.append(node.end_point[1])
            self.split_positions[line_id] = line_spilt_positions

        if len(node.children) > 0:
            for child in node.children:
                self.get_identifier_begin_end(child)


if __name__ == '__main__':
    my_sitter = My_sitter(lang='java')
    code = '''public static long sddCsp(long a, long b) {
                long u=a+b;
                if (u < 0L) {
                    return Long.MAX_VALUE;
                }
                return u;
            }
            '''
    print(my_sitter.add_space_to_identifier(code))
