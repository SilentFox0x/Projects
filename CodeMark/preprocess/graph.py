from typing import List, Dict
from pygments import lex
from pygments.lexers import get_lexer_by_name


class JavaTokenizer:
    def __init__(self):
        self.lexer = get_lexer_by_name('Java', stripall=True)

    def tokenize(self, code: str) -> str:
        token_list = lex(code, self.lexer)
        token_strings = ' '.join([x[1] for x in token_list if x[1] != ' '])
        token_strings = token_strings.strip()
        return token_strings


java_tokenizer = JavaTokenizer()


class Node:
    def __init__(self, node_id, ast_node):
        self.index = node_id
        self.type = ast_node.type
        self.text = ast_node.text
        self.lines = set()
        if ast_node.start_point[0] == ast_node.end_point[0]:
            self.lines.add(ast_node.start_point[0])


class Edge:
    def __init__(self, from_id, to_id):
        self.from_id = from_id
        self.to_id = to_id


def ast_to_graph(ast_root_node):
    def link_edge(node, my_node_index):
        nonlocal nodes
        nonlocal edges
        nonlocal leaf_node_index
        if len(node.children) > 0:
            for child_node in node.children:
                child_index = get_node_id()
                nodes.append(Node(child_index, child_node))
                edges.append(Edge(my_node_index, child_index))
                link_edge(child_node, child_index)
        else:
            leaf_node_index.append(my_node_index)

    def get_node_id():
        nonlocal node_index
        node_index = node_index + 1
        return node_index

    nodes: List[Node] = []
    edges: List[Edge] = []
    leaf_node_index: List[Node] = []
    node_index = -1

    ast_root_node_index = get_node_id()
    nodes.append(Node(ast_root_node_index, ast_root_node))
    link_edge(ast_root_node, ast_root_node_index)

    for i in range(len(leaf_node_index) - 1):
        edges.append(Edge(leaf_node_index[i], leaf_node_index[i + 1]))
    return nodes, edges


def merge_var_node(nodes: List[Node], edges: List[Edge], variables: List[str]):
    variable_node = {}
    for v in variables:
        variable_node[v] = []
        for remove_node in nodes:
            if remove_node.text == v and (remove_node.type == 'identifier' or
                                          remove_node.type == 'shorthand_property_identifier_pattern'):
                variable_node[v].append(remove_node)

    for v, v_nodes in variable_node.items():
        keep_node = v_nodes[0]
        for remove_node in v_nodes[1:]:
            nodes = [x for x in nodes if x.index != remove_node.index]
            for edge in edges:
                if edge.from_id == remove_node.index:
                    edge.from_id = keep_node.index
                if edge.to_id == remove_node.index:
                    edge.to_id = keep_node.index
            keep_node.lines = keep_node.lines.union(remove_node.lines)
        variable_node[v] = keep_node

    return nodes, edges, variable_node


def get_variable_subgraphs(nodes: List[Node], edges: List[Edge], variable_node: Dict[str, Node],
                           statements: List[str]) -> Dict:
    var_edge_statements = {}
    for v, var_node in variable_node.items():
        statement_node_ids = set()
        for node in nodes:
            if len(node.lines.intersection(var_node.lines)):
                statement_node_ids.add(node.index)

        var_edges = []
        for edge in edges:
            if edge.to_id in statement_node_ids and edge.from_id in statement_node_ids:
                var_edges.append([edge.from_id, edge.to_id])

        var_statements = [java_tokenizer.tokenize(statements[line]) for line in var_node.lines]
        var_edge_statements[v] = {'edges': var_edges, 'statements': var_statements}
    return refine_subgraph(nodes, var_edge_statements)


def refine_subgraph(nodes: List[Node], var_edge_statements: Dict) -> Dict:
    subgraphs = {}
    for var_str in var_edge_statements.keys():
        edges = var_edge_statements[var_str]['edges']
        old_ids = set()
        for edge in edges:
            old_ids.add(edge[0])
            old_ids.add(edge[1])
        old_ids = list(old_ids)
        old_ids.sort()

        for node in nodes:
            if node.text == var_str and node.type == 'identifier':
                v_id = node.index
                old_ids.remove(v_id)
                old_ids.insert(0, v_id)

        old_to_new = {}
        for new_id, old_id in enumerate(old_ids):
            old_to_new[old_id] = new_id

        new_edges = []
        for edge in edges:
            new_edges.append([old_to_new[edge[0]], old_to_new[edge[1]]])
        new_nodes = []
        for old_id, new_id in old_to_new.items():
            for node in nodes:
                if node.index == old_id:
                    new_nodes.append([new_id, node.type, node.text])
        subgraphs[var_str] = {
            'nodes': new_nodes,
            'edges': new_edges,
            'statements': var_edge_statements[var_str]['statements']
        }
    return subgraphs
