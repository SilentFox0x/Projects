from typing import List
import functools


class VarAnalyzer:
    def __init__(self, ast_root_node, lang: str):
        self.ast_root_node = ast_root_node
        self.var_info = {}
        self.lang = lang

    def record_variable(self, ast_node):
        var = ast_node.text
        start_point = ast_node.start_point
        if var in self.var_info:
            old_start_point = self.var_info[var]
            # identifier in different line
            if start_point[0] < old_start_point[0]:
                self.var_info[var] = start_point
            # identifier in the same line
            if start_point[0] == old_start_point[0] and start_point[1] < old_start_point[1]:
                self.var_info[var] = start_point
        else:
            self.var_info[var] = start_point

    def _find_variable_for_java(self, ast_node):
        if ast_node.type == 'variable_declarator':
            name_node = ast_node.child_by_field_name('name')
            if name_node.type == 'identifier':
                # String s
                self.record_variable(name_node)
        elif ast_node.type == 'enhanced_for_statement':
            # for (String s : list) {}
            name_node = ast_node.child_by_field_name('name')
            self.record_variable(name_node)
        elif ast_node.type == 'formal_parameter':
            # public void hi( My a ) { }
            for child in ast_node.children:
                if child.type == 'identifier':
                    self.record_variable(child)
        elif len(ast_node.children) > 0:
            for child in ast_node.children:
                self.find_variable(child)

    def _find_variable_for_javascript(self, ast_node):
        if ast_node.type == 'variable_declarator':
            name_node = ast_node.child_by_field_name('name')
            if name_node.type == 'identifier':
                # var s
                self.record_variable(name_node)
            elif self.lang == 'javascript' and name_node.type == 'object_pattern':
                for child in name_node.children:
                    if child.type == 'shorthand_property_identifier_pattern':
                        self.record_variable(child)
        elif ast_node.type == 'formal_parameters':
            # function hi(a){}
            for child in ast_node.children:
                if child.type == 'identifier':
                    self.record_variable(child)
        elif len(ast_node.children) > 0:
            for child in ast_node.children:
                self.find_variable(child)

    def _find_variable_for_python(self, ast_node):
        if ast_node.type == 'assignment':
            left_node = ast_node.child_by_field_name('left')
            if left_node.type == 'identifier':
                # s = 1
                self.record_variable(left_node)
        elif ast_node.type == 'parameters':
            # def hi(a):
            for child in ast_node.children:
                if child.type == 'identifier':
                    self.record_variable(child)
        elif len(ast_node.children) > 0:
            for child in ast_node.children:
                self.find_variable(child)

    def _find_variable_for_cpp(self, ast_node):
        if ast_node.type == 'declaration':
            declarator = ast_node.child_by_field_name('declarator')
            if declarator.type == 'identifier':
                # int x;
                self.record_variable(declarator)
            elif declarator.type == 'init_declarator':
                declarator = ast_node.child_by_field_name('declarator')
                if declarator.type == 'identifier':
                    # int x = y;
                    self.record_variable(declarator)
        elif ast_node.type == 'parameter_declaration':
            # void hi( My a ) { }
            for child in ast_node.children:
                if child.type == 'identifier':
                    self.record_variable(child)
        elif len(ast_node.children) > 0:
            for child in ast_node.children:
                self.find_variable(child)

    def find_variable(self, ast_node):
        if self.lang == 'java':
            self._find_variable_for_java(ast_node)
        elif self.lang == 'javascript':
            self._find_variable_for_javascript(ast_node)
        elif self.lang == 'python':
            self._find_variable_for_python(ast_node)
        elif self.lang == 'c++':
            self._find_variable_for_cpp(ast_node)

    def sort_variable(self):
        def compare_fun(var_1, var_2):
            point_1, point_2 = var_1[1], var_2[1]
            if point_1[0] < point_2[0]:
                return -1
            elif point_1[0] == point_2[0] and point_1[1] < point_2[1]:
                return -1
            else:
                return 1

        sorted(self.var_info.items(), key=functools.cmp_to_key(compare_fun))

    def analyze(self) -> List[str]:
        self.find_variable(self.ast_root_node)
        self.sort_variable()
        return [k.decode('utf-8') for k in self.var_info.keys()]
