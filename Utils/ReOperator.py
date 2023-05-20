import re


class ReOperator:
    def __init__(self, string: str):
        self.string = string

    def search(self, pattern):
        return re.search(pattern=pattern, string=self.string)

    def sub(self, pattern, repl):
        return re.sub(pattern=pattern, repl=repl, string=self.string, flags=re.M)

    def chinese_num_to_digital(self):
        temp = '零一二三四五六七八九'
        string = self.string
        for ind in range(temp.__len__()):
            string = string.replace(temp[ind], str(ind))
        if string[0] == '十':
            string = '1' + string[1:]
        return int(string.replace('十', str()))

    def add1(self):
        return int(self.string) + 1


class MatchObject:
    def __init__(self, match_object):
        self.match_object = match_object

    def string(self):
        return self.match_object.string[self.match_object.regs[0][0]:self.match_object.regs[0][1]]

    def group(self, ind):
        return self.match_object.group(ind)


class ReplInterface:
    def __init__(self, func, output_format: str = '{:02d}'):
        self.output_format = output_format
        self.func = func

    def repl(self, match_object):
        return ReOperator(string=MatchObject(match_object=match_object).string()).sub(
            pattern=MatchObject(match_object=match_object).group(1),
            repl=self.output_format.format(self.func(ReOperator(
                string=MatchObject(match_object=match_object).group(1))))
        )
