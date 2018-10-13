from dit.divergences import kullback_leibler_divergence as kl_divergence
import dit


class EntityDoc:
    def __init__(self, entityURI):
        self.entityURI = entityURI
        self.entityAsSubjDoc = []
        self.entityAsObjDoc = []

    def getEneityDoc(self):


if __name__ == "__main__":
    p = dit.Distribution(['0', '1', '2'], [0, 1 / 2, 1 / 2])
    # p = dit.Distribution(['0', '1', '2'], [1 / 2, 1 / 2, 0])
    q = dit.Distribution(['0', '1', '2'], [1 / 4, 1 / 2, 1 / 4])
    print(kl_divergence(p, q))
    entityURI = "http://dbpedia.org/resource/United_Kingdom"
