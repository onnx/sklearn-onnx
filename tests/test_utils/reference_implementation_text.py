# SPDX-License-Identifier: Apache-2.0
"""
Helpers to test runtimes.
"""

from typing import List
import re
import numpy as np
from onnx.defs import onnx_opset_version


if onnx_opset_version() >= 18:
    from onnx.reference.op_run import OpRun
    from onnx.reference.ops.op_tfidf_vectorizer import (
        WeightingCriteria,
        NgramPart,
        populate_grams,
    )

    class Tokenizer(OpRun):
        op_domain = "com.microsoft"

        def _run(
            self,
            text,
            mark=None,
            mincharnum=None,
            pad_value=None,
            separators=None,
            tokenexp=None,
            tokenexpsplit=None,
            stopwords=None,
        ):
            char_tokenization_ = tokenexp == "." or list(separators or []) == [""]
            stops_ = set(stopwords or [])
            try:
                str_separators_ = set(_ for _ in (separators or ""))
            except AttributeError as e:  # pragma: no cover
                raise TypeError(
                    f"Unable to interpret separators {separators!r}."
                ) from e
            if tokenexp not in (None, ""):
                tokenexp_ = re.compile(tokenexp)

            if char_tokenization_:
                return self._run_char_tokenization(text, stops_, mark, pad_value)
            if str_separators_ is not None and len(str_separators_) > 0:
                str_separators = [re.compile(s) for s in str_separators_]
                return self._run_sep_tokenization(
                    text, stops_, str_separators, mark, pad_value
                )
            if tokenexp not in (None, ""):
                return self._run_regex_tokenization(
                    text, stops_, tokenexp_, tokenexpsplit, mark, pad_value
                )
            raise RuntimeError(  # pragma: no cover
                "Unable to guess which tokenization to use, sep={}, "
                "tokenexp='{}'.".format(separators, tokenexp)
            )

        @staticmethod
        def _run_tokenization(text, stops, split, mark, pad_value):
            """
            Tokenizes a char level.
            """
            begin = 1 if mark else 0
            res = []
            if len(text.shape) == 1:
                for i in range(text.shape[0]):
                    row = [pad_value for _ in range(begin)]
                    for c in split(text[i]):
                        if c not in stops:
                            row.append(c)
                    if mark:
                        row.append(pad_value)
                    res.append(row)
                max_pos = max(map(len, res))
                for row in res:
                    while len(row) < max_pos:
                        row.append(pad_value)
                res = np.array(res)
            elif len(text.shape) == 2:
                max_pos = 0
                for i in range(text.shape[0]):
                    row2 = []
                    for ii in range(text.shape[1]):
                        row = [pad_value for _ in range(begin)]
                        for c in split(text[i, ii]):
                            if c not in stops:
                                row.append(c)
                        if mark:
                            row.append(pad_value)
                        max_pos = max(max_pos, len(row))
                        row2.append(row)
                    res.append(row2)
                for row2 in res:
                    for row in row2:
                        while len(row) < max_pos:
                            row.append(pad_value)
                res = np.array(res)
            else:
                raise RuntimeError(  # pragma: no cover
                    f"Only vector or matrices are supported not shape {text.shape}."
                )
            return (res,)

        @staticmethod
        def _run_char_tokenization(text, stops, mark, pad_value):
            """
            Tokenizes by charaters.
            """

            def split(t):
                for c in t:
                    yield c

            return Tokenizer._run_tokenization(text, stops, split, mark, pad_value)

        @staticmethod
        def _run_sep_tokenization(text, stops, separators, mark, pad_value):
            """
            Tokenizes using separators (as regular expressions).
            The function should use a trie to find text.
            """

            def split(t):
                begin = 0
                pos = 0
                while pos < len(t):
                    for sep in separators:
                        if isinstance(sep, str):
                            if (
                                pos + len(sep) <= len(t)
                                and sep == t[pos : pos + len(sep)]
                            ):
                                word = t[begin:pos]
                                yield word
                                begin = pos + len(sep)
                                break
                        else:
                            se = sep.match(t[pos:])
                            if se:
                                sep = se.group(0)
                                word = t[begin:pos]
                                yield word
                                begin = pos + len(sep)
                                break
                    pos += 1
                if begin < pos:
                    word = t[begin:pos]
                    yield word

            return Tokenizer._run_tokenization(text, stops, split, mark, pad_value)

        @staticmethod
        def _run_regex_tokenization(text, stops, exp, tokenexpsplit, mark, pad_value):
            """
            Tokenizes using a regular expression.
            """
            if tokenexpsplit:

                def split(t):
                    return filter(lambda x: x, exp.split(t))

            else:

                def split(t):
                    return filter(lambda x: x, exp.findall(t))

            return Tokenizer._run_tokenization(text, stops, split, mark, pad_value)

    class TfIdfVectorizer(OpRun):
        def __init__(self, onnx_node, run_params):  # type: ignore
            OpRun.__init__(self, onnx_node, run_params)
            mode = self.mode  # type: ignore

            value = getattr(WeightingCriteria, mode, None)
            if value is None:
                value = getattr(WeightingCriteria, "k" + mode, None)
            if value is None:
                raise ValueError(
                    f"Unexpected mode={mode!r}, "
                    f"not found in {dir(WeightingCriteria)}."
                )
            self.weighting_criteria_ = value  # type: ignore

            self.min_gram_length_ = self.min_gram_length  # type: ignore
            self.max_gram_length_ = self.max_gram_length  # type: ignore
            self.max_skip_count_ = self.max_skip_count  # type: ignore
            self.ngram_counts_ = self.ngram_counts  # type: ignore
            self.max_gram_length_ = self.max_gram_length  # type: ignore
            self.ngram_indexes_ = self.ngram_indexes  # type: ignore
            self.output_size_ = max(self.ngram_indexes_) + 1
            self.weights_ = self.weights  # type: ignore

            if len(self.pool_strings) != 0:
                pool_strings_ = np.array(self.pool_strings)
                mapping = {}
                pool_int64s = []
                for i, w in enumerate(pool_strings_):
                    if w not in mapping:
                        # 1-gram are processed first.
                        mapping[w] = i
                    pool_int64s.append(mapping[w])
            else:
                mapping = None
                pool_int64s = self.pool_int64s
                pool_strings_ = None

            self.mapping_ = mapping
            self.pool_int64s_ = pool_int64s
            self.pool_strings_ = pool_strings_
            self.int64_map_ = NgramPart(-10)
            self.int64_map_.init()

            total_items = len(self.pool_int64s_)
            ngram_id = 1  # start with 1, 0 - means no n-gram
            # Load into dictionary only required gram sizes
            ngram_size = 1
            for i in range(len(self.ngram_counts_)):
                start_idx = self.ngram_counts_[i]
                end_idx = (
                    self.ngram_counts_[i + 1]
                    if (i + 1) < len(self.ngram_counts_)
                    else total_items
                )
                items = end_idx - start_idx
                if items > 0:
                    ngrams = items // ngram_size
                    if (
                        ngram_size >= self.min_gram_length_
                        and ngram_size <= self.max_gram_length_
                    ):
                        ngram_id = populate_grams(
                            self.pool_int64s_,
                            start_idx,
                            ngrams,
                            ngram_size,
                            ngram_id,
                            self.int64_map_,
                        )
                    else:
                        ngram_id += ngrams
                ngram_size += 1

        def increment_count(
            self, ngram_id: int, row_num: int, frequencies: List[int]
        ) -> None:
            ngram_id -= 1
            # assert(ngram_id < ngram_indexes_.size());
            output_idx = row_num * self.output_size_ + self.ngram_indexes_[ngram_id]
            # assert(static_cast<size_t>(output_idx) < frequencies.size());
            frequencies[output_idx] += 1

        def output_result(self, B: int, frequencies: List[int]) -> np.ndarray:
            def _getattr(cls, name):
                try:
                    return getattr(cls, name)
                except AttributeError:
                    return getattr(cls, "k" + name)

            l_output_dims: List[int] = []
            if B == 0:
                l_output_dims.append(self.output_size_)
                B = 1
            else:
                l_output_dims.append(B)
                l_output_dims.append(self.output_size_)
            output_dims = tuple(l_output_dims)

            row_size = self.output_size_

            total_dims = np.prod(output_dims)
            Y = np.empty((total_dims,), dtype=np.float32)

            w = self.weights_
            if self.weighting_criteria_ == _getattr(WeightingCriteria, "TF"):
                i = 0
                for f in frequencies:
                    Y[i] = f
                    i += 1
            elif self.weighting_criteria_ == _getattr(WeightingCriteria, "IDF"):
                if len(w) > 0:
                    p = 0
                    for _batch in range(B):
                        for i in range(row_size):
                            Y[p] = w[i] if frequencies[p] > 0 else 0
                            p += 1
                else:
                    p = 0
                    for f in frequencies:
                        Y[p] = 1 if f > 0 else 0
                        p += 1
            elif self.weighting_criteria_ == _getattr(WeightingCriteria, "TFIDF"):
                if len(w) > 0:
                    p = 0
                    for _batch in range(B):
                        for i in range(row_size):
                            Y[p] = w[i] * frequencies[p]
                            p += 1
                else:
                    p = 0
                    for f in frequencies:
                        Y[p] = f
                        p += 1
            else:
                raise RuntimeError("Unexpected weighting_criteria.")
            return Y.reshape(output_dims)

        def compute_impl(  # type: ignore
            self,
            X: np.ndarray,
            row_num: int,
            row_size: int,
            frequencies: List[int],
            max_gram_length=None,
            max_skip_count=None,
            min_gram_length=None,
            mode=None,
            ngram_counts=None,
            ngram_indexes=None,
            pool_int64s=None,
            pool_strings=None,
            weights=None,
        ) -> None:
            X_flat = X[row_num] if len(X.shape) > 1 else X
            row_begin = 0
            row_end = row_begin + row_size

            max_skip_distance = max_skip_count + 1
            start_ngram_size = min_gram_length

            for skip_distance in range(1, max_skip_distance + 1):
                ngram_start = row_begin
                ngram_row_end = row_end

                while ngram_start < ngram_row_end:
                    # We went far enough so no n-grams of any size can be
                    # gathered
                    at_least_this = ngram_start + skip_distance * (start_ngram_size - 1)
                    if at_least_this >= ngram_row_end:
                        break

                    ngram_item = ngram_start
                    int_map = self.int64_map_
                    ngram_size = 1
                    while (
                        int_map.has_leaves()
                        and ngram_size <= max_gram_length
                        and ngram_item < ngram_row_end
                    ):
                        val = X_flat[ngram_item]
                        hit = int_map.find(val)
                        if hit is None:
                            break
                        hit = int_map[val].id_
                        if ngram_size >= start_ngram_size and hit != 0:
                            self.increment_count(hit, row_num, frequencies)
                        int_map = int_map[val]
                        ngram_size += 1
                        ngram_item += skip_distance

                    ngram_start += 1

                # We count UniGrams only once since they are not affected by
                # skip_distance
                if start_ngram_size == 1:
                    start_ngram_size += 1
                    if start_ngram_size > max_gram_length:
                        break

        def _run(  # type: ignore
            self,
            X,
            max_gram_length=None,
            max_skip_count=None,
            min_gram_length=None,
            mode=None,
            ngram_counts=None,
            ngram_indexes=None,
            pool_int64s=None,
            pool_strings=None,
            weights=None,
        ):
            if self.mapping_ is not None:
                xi = np.empty(X.shape, dtype=np.int64)
                for i in range(0, X.shape[0]):
                    for j in range(0, X.shape[1]):
                        try:
                            xi[i, j] = self.mapping_[X[i, j]]
                        except KeyError:
                            xi[i, j] = -1
            else:
                xi = X

            # weights should be identical to self.weights as well as
            # pool_strings, pool_int64s, ngram_indexes, ngram_counts, mode.
            # This means none of those attributes can be used in one function.

            total_items = np.prod(xi.shape)

            num_rows = 0
            B = 0
            C = 0
            input_dims = xi.shape
            if len(input_dims) == 0:
                num_rows = 1
                C = 1
                if total_items != 1:
                    raise ValueError(f"Unexpected total of items {total_items}.")
            elif len(input_dims) == 1:
                num_rows = 1
                C = input_dims[0]
            elif len(input_dims) == 2:
                B = input_dims[0]
                C = input_dims[1]
                num_rows = B
                if B < 1:
                    raise ValueError(
                        f"Input shape must have either [C] or [B,C] "
                        f"dimensions with B > 0, B={B}, C={C}."
                    )
            else:
                raise ValueError(
                    f"Input shape must have either [C] or [B,C] "
                    f"dimensions with B > 0, B={B}, C={C}."
                )

            if num_rows * C != total_items:
                raise ValueError(
                    f"Unexpected total of items, num_rows * C = "
                    f"{num_rows * C} != total_items = {total_items}."
                )
            # Frequency holder allocate [B..output_size_] and init all to zero
            frequencies = np.zeros((num_rows * self.output_size_,), dtype=np.int64)

            if total_items == 0 or self.int64_map_.empty():
                return (self.output_result(B, frequencies),)

            def fn(row_num):
                self.compute_impl(
                    xi,
                    row_num,
                    C,
                    frequencies,
                    max_gram_length=max_gram_length,
                    max_skip_count=max_skip_count,
                    min_gram_length=min_gram_length,
                    mode=mode,
                    ngram_counts=ngram_counts,
                    ngram_indexes=ngram_indexes,
                    pool_int64s=pool_int64s,
                    pool_strings=pool_strings,
                    weights=weights,
                )

            # can be parallelized.
            for i in range(num_rows):
                fn(i)

            return (self.output_result(B, frequencies),)
