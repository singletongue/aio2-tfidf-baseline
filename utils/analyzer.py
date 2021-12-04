import unicodedata

import ipadic
from fugashi import GenericTagger
from typing import Any, Dict, List, Optional


class JapaneseTextAnalyzer:
    def __init__(
        self,
        do_unicode_normalize: bool = True,
        pos_list: Optional[List[str]] = None,
        stop_words: Optional[List[str]] = None,
    ) -> None:
        if do_unicode_normalize and stop_words is not None:
            stop_words = [unicodedata.normalize("NFKC", word) for word in stop_words]

        self._do_unicode_normalize = do_unicode_normalize
        self._pos_list = pos_list
        self._stop_words = stop_words

        self.tagger = GenericTagger(ipadic.MECAB_ARGS)

    def __call__(self, text: str) -> str:
        if self._do_unicode_normalize:
            text = unicodedata.normalize("NFKC", text)

        tokens = []

        # tokenize the text
        for token in self.tagger(text):
            if self._pos_list is not None and token.feature[0] not in self._pos_list:
                continue
            if self._stop_words is not None and token.surface in self._stop_words:
                continue

            tokens.append(token.surface)

        return tokens

    def __getstate__(self)-> Dict[str, Any]:
        state = self.__dict__.copy()
        del state["tagger"] # unpicklable object
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.tagger = GenericTagger(ipadic.MECAB_ARGS)
