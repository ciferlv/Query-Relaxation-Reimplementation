from dit.divergences import kullback_leibler_divergence as kl_divergence
from dit.divergences import jensen_shannon_divergence as js_divergence
import dit


class EntityDoc:
    def __init__(self, entityURI):
        self.entityURI = entityURI
        self.entityAsSubjDoc = []
        self.entityAsObjDoc = []


if __name__ == "__main__":
    p = dit.ScalarDistribution(['0', '1', '2'], [1 / 9, 1 / 9, 7 / 9])
    q = dit.ScalarDistribution(['0', '1','2'], [1 / 9,1/9, 7 / 9])
    # q = dit.Distribution(['1', '2'], [1 / 2, 1 / 4])
    print(js_divergence([p, q]))
    entityURI = "http://dbpedia.org/resource/United_Kingdom"
