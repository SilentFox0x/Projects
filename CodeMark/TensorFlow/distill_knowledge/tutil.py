class TokenBeginEndInSubToken:
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end


def _tokenize(func, tokenizer):
    func = func.replace('\n', '')
    tokens = func.split(' ')

    sub_tokens = []
    occurrences = []
    index = 0
    for word in tokens:
        # 并非直接tokenize这句话，而是tokenize了每个splited tokens.
        sub = tokenizer.tokenize(word)
        sub_tokens += sub
        occurrences.append(TokenBeginEndInSubToken(index, index + len(sub)))
        # 将subwords对齐
        index += len(sub)

    return tokens, sub_tokens, occurrences


def get_variable_posistions_from_code(tokens: list, variable_names: list) -> dict:
    '''
    给定一串代码，以及variable的变量名，如: a
    返回这串代码中这些变量名对应的位置.
    '''
    positions = {}
    for name in variable_names:
        for index, token in enumerate(tokens):
            if name == token:
                try:
                    positions[name].append(index)
                except:
                    positions[name] = [index]

    return positions