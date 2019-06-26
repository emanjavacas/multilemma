
import copy
import collections

import torch

from pie.data import LabelEncoder
from pie import torch_utils

import random
random.seed(1001)


def readlines_conll(path, skip_contractions=False):
    """
    path: str, path to file with data
    skip_contractions: bool, whether to skip contracted forms.
        Some languages have word forms like split in their abstract constituents
        (e.g. gl "no" => "en" "o"). The original form usually has no lemma. For
        experimentation, it probably doesn't matter, but it might have big effect
        on out-of-domain data, which won't be presented in the split way.
    """
    with open(path) as f:
        sent, tasks = [], collections.defaultdict(list)
        skips = 0

        for line in f:
            line = line.strip()
            # skip metadata
            if line.startswith('#'):
                continue
            # new sentence marker
            elif not line:
                yield sent, dict(tasks)
                sent, tasks = [], collections.defaultdict(list)
                skips = 0
            # actual line
            else:
                num, token, lemma, pos, ppos, morph, *_ = line.split()
                # skip line based on decontractions (6-7: al => 6: a, 7: il)
                if skip_contractions:
                    if skips > 0:
                        skips -= 1
                        continue
                    if '-' in num:
                        assert skips == 0, "Overlaping decontractions"
                        a, b = num.split('-')
                        skips = 1 + (int(b) - int(a))

                # accumulate
                sent.append(token)
                for task, data in {'lemma': lemma, 'pos': pos, 'ppos': ppos}.items():
                    tasks[task].append(data)

        if sent:
            yield sent, dict(tasks)


def readlines(path, maxlen=35):
    """
    path: str, path to file with data
    maxlen: int, maximum sentence length
    """
    with open(path) as f:
        # skip header
        next(f)
        sent, tasks = [], collections.defaultdict(list)

        for line in f:
            line = line.strip()

            # new sentence marker
            if not line:
                yield sent, dict(tasks)
                sent, tasks = [], collections.defaultdict(list)

            # actual line
            else:
                token, lemma, pos, *_ = line.split()

                # accumulate
                sent.append(token)
                for task, data in {'lemma': lemma, 'pos': pos}.items():
                    tasks[task].append(data)

                if len(sent) >= maxlen:
                    yield sent, dict(tasks)
                    sent, tasks = [], collections.defaultdict(list)

        if sent:
            yield sent, dict(tasks)


class BatchIterator:
    def __init__(self, reader, batch_size):
        self.reader = reader
        self.batch_size = batch_size
        self.reset()

    def get_batch(self):
        sent, tasks = zip(*[next(self.sents) for _ in range(self.batch_size)])
        # transform [{"t1": [], ...n}, ...m] => {"t1": [[], m], ...n}
        task_batch = collections.defaultdict(list)
        for item in tasks:
            for task, data in item.items():
                task_batch[task].append(data)
        return sent, dict(task_batch)

    def __iter__(self):
        # get first batch
        try:
            old_batch = self.get_batch()
        except StopIteration:
            raise ValueError("Not enough data for batch size: {}".format(
                self.batch_size))

        while True:
            try:
                batch = self.get_batch()
                yield old_batch
                old_batch = batch
            except StopIteration:
                if self.done:
                    yield
                else:
                    self.done = True
                    yield old_batch

    def reset(self):
        self.sents = self.reader.readsents()
        self.done = False


class LanguageReader:
    def __init__(self, path,
                 linereader=readlines_conll, buffersize=25000, maxlen=35, minlen=0):
        self.path = path
        self.linereader = linereader
        self.buffersize = buffersize
        self.maxlen = maxlen
        self.minlen = minlen

    def readsents(self):
        buf = []
        for sent, tasks in self.linereader(self.path):

            if len(sent) < self.minlen:
                continue

            while len(sent) >= self.maxlen:
                buf.append(
                    (sent[:self.maxlen],
                     {t: tdata[:self.maxlen] for t, tdata in tasks.items()}))
                sent = sent[self.maxlen:]
                tasks = {t: tdata[self.maxlen:] for t, tdata in tasks.items()}

            if sent:
                buf.append((sent, tasks))

            if len(buf) >= self.buffersize:
                random.shuffle(buf)
                yield from buf
                buf = buf[self.buffersize:]

        yield from buf


class MultiLanguageEncoder:

    LanguageEncoder = collections.namedtuple('LanguageEncoder', ('char', 'lemma'))

    def __init__(self, languages, share_encoder=True, share_decoder=True,
                 prepend=('char', 'lemma'), **kwargs):
        self.char, self.lemma = {}, {}
        self.languages = languages
        self.share_encoder = share_encoder
        self.share_decoder = share_decoder
        self.prepend = prepend

        char = LabelEncoder(
            level='char', name='char', eos=True, bos=True,
            reserved=tuple(self.languages) if 'char' in prepend else (), **kwargs)
        lemma = LabelEncoder(
            level='char', name='lemma', eos=True, bos=True,
            reserved=tuple(self.languages) if 'lemma' in prepend else (), **kwargs)

        for lang in languages:
            self.char[lang] = char if share_encoder else copy.deepcopy(char)
            self.lemma[lang] = lemma if share_decoder else copy.deepcopy(lemma)

        self.stats = collections.defaultdict(collections.Counter)
        self.known = collections.defaultdict(collections.Counter)
        self.ambig = collections.defaultdict(lambda: collections.defaultdict(set))

    def to_dict(self):
        return {'char': {lang: le.jsonify() for lang, le in self.char.items()},
                'lemma': {lang: le.jsonify() for lang, le in self.lemma.items()},
                'languages': self.languages,
                'share_encoder': self.share_encoder,
                'share_decoder': self.share_decoder,
                'prepend': self.prepend,
                'ambig': {lang: list(s) for lang, s in self.ambig.items()},
                'known': {lang: dict(d) for lang, d in self.known.items()}}

    @classmethod
    def from_dict(cls, d):
        inst = cls(d['languages'],
                   share_encoder=d['share_encoder'], share_decoder=d['share_decoder'],
                   prepend=tuple(d['prepend']))
        inst.char = {lang: LabelEncoder.from_json(d) for lang, d in d['char'].items()}
        inst.lemma = {lang: LabelEncoder.from_json(d) for lang, d in d['lemma'].items()}
        inst.ambig = {lang: set(s) for lang, s in d['ambig'].items()}
        inst.known = d['known']

        return inst

    def fit_readers(self, lreaders, verbose=True):
        for lang, reader in lreaders.items():
            for sent, tasks in reader.readsents():
                # add sent data
                self.char[lang].add(sent)
                self.lemma[lang].add(tasks['lemma'])
                # add stats
                self.stats[lang]['sents'] += 1
                self.stats[lang]['words'] += len(sent)
                self.stats[lang]['chars'] += sum(len(w) for w in sent)
                # add known and ambiguous tokens
                for w, l in zip(sent, tasks['lemma']):
                    self.known[lang][w] += 1
                    self.ambig[lang][w].add(l)

        for lang in self.languages:
            # trim ambiguous
            self.ambig[lang] = set(w for w, l in self.ambig[lang].items() if len(l) > 1)
            if not self.char[lang].fitted:
                self.char[lang].compute_vocab()
            if not self.lemma[lang].fitted:
                self.lemma[lang].compute_vocab()

        if verbose:
            self.print_stats()

        return self

    def print_stats(self):
        for lang in self.languages:
            if not self.char[lang].fitted:
                continue

            print("Language:", lang)
            print('---' * 10)
            if not self.share_encoder:
                print("Character-level encoder")
                print(self.char[lang])
                print()
            if not self.share_decoder:
                print("Lemma-level encoder")
                print(self.lemma[lang])
                print()
            print("* # sents", self.stats[lang]['sents'])
            print("* # words/mean sent length (words)", self.stats[lang]['words'],
                  round(self.stats[lang]['words'] / self.stats[lang]['sents'], 2))
            print("* # chars/mean sent length (chars)", self.stats[lang]['chars'],
                  round(self.stats[lang]['chars'] / self.stats[lang]['sents'], 2))
            print("* % ambiguous tokens in training set",
                  round(sum(self.known[lang][w] for w in self.ambig[lang])
                        / sum(self.known[lang].values()), 2))
            print()
            print()

    def get_random_lang(self):
        return next(iter(self.languages))

    def get_lang_encoder(self, lang):
        return MultiLanguageEncoder.LanguageEncoder(self.char[lang], self.lemma[lang])


def prepare_batch(encoder, batch, device='cpu', lang=None, prepend=()):
    # unpack
    sents, tasks = batch
    # transform to numeric
    nwords = torch.tensor([len(s) for s in sents]).to(device)
    char = [encoder.char.transform(w) for s in sents for w in s]
    lemma = [encoder.lemma.transform(w) for s in tasks['lemma'] for w in s]
    # replace <eos> with lang id if required
    if 'char' in prepend:
        char = [[encoder.char.table[lang]] + w[1:] for w in char]
    if 'lemma' in prepend:
        lemma = [[encoder.lemma.table[lang]] + w[1:] for w in lemma]
    # create tensors
    char = torch_utils.pad_batch(char, encoder.char.get_pad(), device=device)
    lemma = torch_utils.pad_batch(lemma, encoder.lemma.get_pad(), device=device)

    inp = (None, nwords), char
    target = lemma

    return inp, target


class MultiLanguageDataset:
    def __init__(self, encoder, readers, batch_size, device):
        assert isinstance(encoder, MultiLanguageEncoder)
        self.device = device
        self.encoder = encoder
        self.readers = readers
        self.iters = {lang: BatchIterator(r, batch_size) for lang, r in readers.items()}

    def get_batches(self):
        raise NotImplementedError


class MultiLanguageOversampling(MultiLanguageDataset):
    def get_batches(self):
        for lang in self.iters:
            self.iters[lang].reset()

        iters = {lang: iter(it) for lang, it in self.iters.items()}
        active = set(self.readers)
        while len(active) > 0:
            lang = random.choice(list(iters))
            if self.iters[lang].done:
                if lang in active:
                    active.remove(lang)
                self.iters[lang].reset()

            encoder = self.encoder.get_lang_encoder(lang)
            batch = prepare_batch(
                encoder, next(iters[lang]), device=self.device,
                lang=lang, prepend=self.encoder.prepend)
            yield batch, lang


class MultiLanguageBalanced(MultiLanguageDataset):
    def get_batches(self):
        for lang in self.iters:
            self.iters[lang].reset()

        iters = {lang: iter(it) for lang, it in self.iters.items()}
        langs, weights = zip(*[(l, self.encoder.stats[l]['sents']) for l in iters])
        langs, weights = list(langs), list(weights)

        while len(langs) > 0:
            lang = random.choices(langs, weights)[0]
            if self.iters[lang].done:
                idx = langs.index(lang)
                langs.pop(idx)
                weights.pop(idx)
            else:
                encoder = self.encoder.get_lang_encoder(lang)
                batch = prepare_batch(
                    encoder, next(iters[lang]), device=self.device,
                    lang=lang, prepend=self.encoder.prepend)
                yield batch, lang


class MultiLanguageScheduled(MultiLanguageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # store schedule params

    def get_batches(self):
        raise NotImplementedError

    def update(self, iteration):
        pass


# import numpy as np
# # def mean_average_precision(m, labels):
# m = np.random.normal(0, 1, (100, 2000))
# qy = np.random.randint(0, 20, 100)
# ty = np.random.randint(0, 20, 2000)
# trues = qy[:, None] == ty[None, :]
# m_ranks = np.argsort(m, axis=1)
# x, _ = np.indices(m.shape)
# trues_ranks = trues[x, m_ranks].astype(float)
