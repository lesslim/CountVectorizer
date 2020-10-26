from typing import List, Set
from collections import Counter


class CountVectorizer():
    """
    Класс для создания массива токенов и матрицы частоты
    вхождения слов из корпуса в этот массив.
    """

    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase
        self.features_set: Set[str] = set()
        self.feature_names: List[str] = []
        self.matrix: List[List[int]] = []

    def _fit(self, corpus: List[str], anew: bool):
        """
        Создание массива токенов, подготовка корпуса.
        """
        if anew:
            self.features_set = set()
            self.feature_names = []
        list_of_counters = []
        self.matrix = []

        for text in corpus:
            if self.lowercase:
                text = text.lower()
            words = text.split()
            list_of_counters.append(Counter(words))

            for word in words:
                if word in self.features_set:
                    continue
                self.features_set.add(word)
                self.feature_names.append(word)

        return list_of_counters

    def fit_transform(self, corpus: List[str], anew: bool = True):
        """
        Возвращает матрицу частоты вхождений слов в тексты корпуса.
        """
        list_of_counters = self._fit(corpus, anew)
        features_len = len(self.feature_names)
        for cntr in list_of_counters:
            ans = [0] * features_len

            for i in cntr:
                ans[self.feature_names.index(i)] = cntr[i]
            self.matrix.append(ans)
        return self.matrix

    def get_feature_names(self):
        """
        Возвращает массив токенов.
        """
        return self.feature_names


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    assert vectorizer.get_feature_names() == ['crock', 'pot', 'pasta', 'never',
                                              'boil', 'again', 'pomodoro',
                                              'fresh', 'ingredients',
                                              'parmesan', 'to', 'taste']
    assert count_matrix == [[1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    corpus2 = [
        'Crock not to taste'
    ]
    assert vectorizer.fit_transform(corpus2, False) == [[1, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0, 1, 1, 1]]
    assert vectorizer.get_feature_names() == ['crock', 'pot', 'pasta', 'never',
                                              'boil', 'again', 'pomodoro',
                                              'fresh', 'ingredients',
                                              'parmesan', 'to', 'taste', 'not']
