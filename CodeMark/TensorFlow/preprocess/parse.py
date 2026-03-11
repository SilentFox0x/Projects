from typing import Dict
from tree_sitter import Language, Parser
from preprocess.data_flow.utils import tree_to_token_index, index_to_code_token
from preprocess.graph import ast_to_graph, merge_var_node, get_variable_subgraphs
from .var_analyzer import VarAnalyzer


class MyASTNode:
    def __init__(self, node_type, text, start_point, end_point, children):
        assert text is not None, 'ast node text is empty'
        assert text != '', 'ast node text is empty'
        self.type = node_type
        self.text = text
        self.start_point = start_point
        self.end_point = end_point
        self.children = children


class MyParser:
    def __init__(self, lang: str):
        self.lang = lang
        language = Language('/home/liwei/Code-Watermark/variable-watermark/resources/my-languages.so', lang)
        self.parser = Parser()
        self.parser.set_language(language)

    def _create_my_ast_node(self, ast_node):
        children = []
        if len(ast_node.children) > 0:
            text = ''
            for child in ast_node.children:
                child_node = self._create_my_ast_node(child)
                text += ' ' + child_node.text
                children.append(child_node)
            my_ast_node = MyASTNode(node_type=ast_node.type, text=text,
                                    start_point=ast_node.start_point, end_point=ast_node.end_point, children=children)
        else:
            text = ast_node.text.decode("utf-8")
            my_ast_node = MyASTNode(node_type=ast_node.type, text=text,
                                    start_point=ast_node.start_point, end_point=ast_node.end_point, children=children)
        return my_ast_node

    def _get_ast_info(self, code: str, tree=None):
        if tree is None:
            tree = self.parser.parse(bytes(code, 'utf8'))
        # new_code = []
        # refine_code_by_lexical(tree.root_node, new_code)

        # code = ' '.join(new_code)
        # tree = parser.parse(bytes(code, 'utf8'))
        if self.lang == 'java':
            root_node = tree.root_node.children[0].children[3].children[1]
            assert root_node.type == 'method_declaration' or root_node.type == 'constructor_declaration', 'cannot find method'
        elif self.lang in ['javascript', 'python', 'c++']:
            root_node = tree.root_node
        else:
            raise KeyError(f'do not support {self.lang}')
        tokens_index = tree_to_token_index(root_node)
        code = code.split('\n')
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        return root_node, index_to_code

    def parse_code(self, code: str, tree=None) -> Dict:
        statements = code.split('\n')
        # ast
        ast_root_node, index_to_code = self._get_ast_info(code, tree)

        # 根据变量出现顺序, 对变量做拓扑排序
        var_analyzer = VarAnalyzer(ast_root_node, self.lang)
        variables = var_analyzer.analyze()

        # print(variables)

        my_ast_root_node = self._create_my_ast_node(ast_root_node)
        # graph
        nodes, edges = ast_to_graph(my_ast_root_node)

        nodes, edges, variables_nodes = merge_var_node(nodes, edges, variables)

        subgraphs = get_variable_subgraphs(nodes, edges, variables_nodes, statements)
        # edges = add_data_flow(nodes, edges, variables, data_flows)
        return subgraphs

    def ast_has_error(self, code):
        tree = self.parser.parse(bytes(code, 'utf8'))
        has_error = self._ast_has_error(tree.root_node)
        return has_error, tree

    def _ast_has_error(self, node):
        if node.has_error is True:
            return True
        for child in node.children:
            if self._ast_has_error(child) is True:
                return True
        return False


if __name__ == '__main__':
    my_parser = MyParser('javascript')
    func = '''
        function ItemImage() {
      const { size } = 1
    }
        '''

    func = '''
        function HooksRunner (projectRoot) {
        this.projectRoot = 1;
    }
        '''
    subgraphs = my_parser.parse_code(func)
