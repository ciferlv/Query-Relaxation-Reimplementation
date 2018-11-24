from RuleBased.VirtuosoSearch.Util import Util


class Rule:
    def __init__(self, rule_chain, accuracy, recall, f1):
        self.util = Util()
        self.rule_chain = rule_chain
        self.accuracy = accuracy
        self.recall = recall
        self.f1 = f1
        self.parsed_rule = self.util.rule_parser(self.rule_chain)

    def calculate_f1(self):
        self.f1 = self.accuracy * self.recall * 2 / (self.accuracy + self.recall)

    def __lt__(self, other):
        # return self.f1 > other.f1
        return self.accuracy > other.accuracy

    def __gt__(self, other):
        # return self.f1 < other.f1
        return self.accuracy < other.accuracy

    def __eq__(self, other):
        # return self.f1 == other.f1
        return self.accuracy == other.accuracy

    def __str__(self):
        msg = "{}\t{}\t{}\t{}".format(self.rule_chain, self.accuracy, self.recall, self.f1)
        return msg