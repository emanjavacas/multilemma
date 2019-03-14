
import glob
import os
from dataset import MultiLanguageDataset, MultiLanguageEncoder, LanguageReader

def get_paths(langpath, splits=('train', 'test', 'dev')):
    paths = glob.glob(os.path.join(langpath, '*conllu'))
    output = []
    for split in splits:
        for path in paths:
            if split in path:
                output.append(path)

    return output

root = '../datasets/ud-treebanks-v2.2/'
langs = {
        'ca': 'UD_Catalan-AnCora',
        'fr': 'UD_French-GSD',
        'gl': 'UD_Galician-CTG',
        'pt': 'UD_Portuguese-GSD',
        'es': 'UD_Spanish-GSD',
        'it': 'UD_Italian-ISDT',
        'ro': 'UD_Romanian-RRT'}

print("Creating readers")
readers = {lang: LanguageReader(get_paths(os.path.join(root, path))[0])
           for lang, path in langs.items()}
encoder = MultiLanguageEncoder(list(langs)).fit_readers(readers)
dataset = MultiLanguageDataset(encoder, readers, 10, 'cpu')
batches = dataset.get_batches_balanced()
import tqdm
import collections
counts = collections.Counter()
sents = collections.Counter()
for (((_, nwords), _), _), lang in tqdm.tqdm(batches):
    counts[lang] += sum(nwords).item()
    sents[lang] += len(nwords)
print(counts)
print(sents)

reader = readers['gl']
from dataset import BatchIterator
it = BatchIterator(reader, 10)
# for b in it:
#     print()
