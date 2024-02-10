from collections import defaultdict, Counter
import re


class TransformerTokenizer:
    """Byte Pair Encoding"""
    def __init__(self, vocab):
        self.vocab = Counter()
        self.bpe_rules = []
        self.num_merges = num_merges

    def build_vocab_from_corpus(self, corpus):
        self.vocab = Counter(re.findall(r'\w+|[^\w\s]', corpus, re.UNICODE))

        for word in list(self.vocab):
            del self.vocab[word]
            word += '<\w>'
            self.vocab[word] = 1

        tokens = set()
        for word in self.vocab:
            tokens.update(list(word))

        for _ in range(self.num_merges):
            pairs = defaultdict(int)
            for word, freq in self.vocab.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pairs[symbols[i], symbols[i+1]] += freq

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            self.bpe_rules.append(best_pair)

            new_tokens = ' '.join((best_pair))
            updates = {}
            for word in self.vocab:
                new_word = re.sub(' '.join(best_pair), new_tokens, word)
                updates[new_word] = self.vocab[word]
                del self.vocab[word]
            self.vocab.update(updates)

    def tokenize(self, text):
        words = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        tokens = []
        for word in words:
            word += '</w>'
            word_tokens = list(word)
            for i in range(len(self.bpe_rules)):
                if len(word_tokens) == 1:
                    break
                new_word_tokens = []
                skip = False
                for j in range(len(word_tokens)-1):
                    pair = (word_tokens[j], word_tokens[j+1])
                    if skip:
                        skip = False
                        continue
                    elif pair == self.bpe_rules[i]:
                        new_word_tokens.append(''.join(pair))
                        skip = True
                    else:
                        new_word_tokens.append(word_tokens[j])
                if not skip:
                    new_word_tokens.append(word_tokens[-1])
                word_tokens = new_word_tokens
            tokens.extend(word_tokens)
        return tokens
    


if __name__ == '__main__':
    corpus = "This is a simple example to illustrate how BPE works. BPE works by iteratively merging frequent pairs."
    tokenizer = TransformerTokenizer(num_merges=50)
    tokenizer.build_vocab_from_corpus(corpus)
    print(tokenizer.tokenize("This is a test sentence for our BPE tokenizer."))

