
import os
import glob
import collections
import logging
import time

import tqdm
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from pie import torch_utils
from pie import RNNEmbedding, AttentionalDecoder, RNNEncoder
from dataset import prepare_batch, BatchIterator
from dataset import MultiLanguageDataset, MultiLanguageEncoder, LanguageReader


class Model(nn.Module):
    def __init__(self, encoder, cemb_dim, hidden_size, num_layers, dropout=0.0,
                 cemb_layers=1, cell='GRU', init_rnn='xavier_uniform', scorer='general'):
        self.encoder = encoder
        self.dropout = dropout
        super().__init__()

        encoder = encoder.get_lang_encoder()

        # Embeddings
        self.cemb = RNNEmbedding(len(encoder.char), cemb_dim,
                                 padding_idx=encoder.char.get_pad(), dropout=dropout,
                                 num_layers=cemb_layers, cell=cell, init_rnn=init_rnn)

        # TODO: Encoder

        # Decoder
        self.decoder = AttentionalDecoder(
            encoder.lemma, cemb_dim, self.cemb.embedding_dim,
            # fix for now
            context_dim=0,
            scorer=scorer, num_layers=cemb_layers, cell=cell,
            dropout=dropout, init_rnn=init_rnn)

    def device(self):
        return next(self.parameters()).device

    def loss(self, batch_data, lang):
        ((_, wlen), (char, clen)), (lemma, llen) = batch_data

        # Embedding
        cemb, cemb_outs = self.cemb(char, clen, wlen)

        # Decoder
        cemb_outs = F.dropout(cemb_outs, p=self.dropout, training=self.training)
        logits = self.decoder(lemma, llen, cemb_outs, clen)
        loss = self.decoder.loss(logits, lemma)

        return loss, sum(wlen).item()

    def predict(self, inp, lang, use_beam=False, width=10):
        encoder = self.encoder.get_lang_encoder()
        # prepend with lang id
        bos = encoder.lemma.table[lang] if 'lemma' in self.encoder.prepend else None

        (_, wlen), (char, clen) = inp
        cemb, cemb_outs = self.cemb(char, clen, wlen)
        if use_beam:
            hyps, _ = self.decoder.predict_beam(cemb_outs, clen, width=width, bos=bos)
        else:
            hyps, _ = self.decoder.predict_max(cemb_outs, clen, bos=bos)

        if encoder.lemma.preprocessor_fn is None:
            hyps = [''.join(hyp) for hyp in hyps]

        return hyps

    def evaluate(self, reader, lang, batch_size=50, **kwargs):
        assert not self.training
        encoder = self.encoder.get_lang_encoder(lang)
        trues, preds = [], []

        with torch.no_grad(), tqdm.tqdm() as pbar:
            it = BatchIterator(reader, batch_size)
            batches = iter(it)
            while not it.done:
                pbar.update(batch_size)
                sents, tasks = next(batches)
                inp, target = prepare_batch(
                    encoder, (sents, tasks), device=self.device(),
                    lang=lang, prepend=self.encoder.prepend)
                hyps = self.predict(inp, lang, **kwargs)
                true = [lemma for sent in tasks['lemma'] for lemma in sent]
                for true, hyp in zip(true, hyps):
                    trues.append(true)
                    preds.append(hyp)

        acc = float(accuracy_score(trues, preds))
        true_amb, pred_amb, true_unk, pred_unk = [], [], [], []
        for true, pred in zip(trues, preds):
            if true not in self.encoder.known[lang]:
                true_unk.append(true)
                pred_unk.append(pred)
            if true in self.encoder.ambig[lang]:
                true_amb.append(true)
                pred_amb.append(pred)
        acc_amb = float(accuracy_score(true_amb, pred_amb))
        acc_unk = float(accuracy_score(true_unk, pred_unk))
        return acc, acc_amb, acc_unk

    def train_epoch(self, batches, optimizer, clip_norm, report_freq):
        stats = {}
        stats['loss'] = collections.defaultdict(float)
        stats['batches'] = collections.defaultdict(int)
        stats['start'] = time.time()
        stats['nwords'] = 0

        for b, (batch_data, lang) in enumerate(batches):
            # loss
            loss, nwords = self.loss(batch_data, lang)

            # optimize
            optimizer.zero_grad()
            loss.backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
            optimizer.step()

            # accumulate
            stats['loss'][lang] += loss.item()
            stats['batches'][lang] += 1
            stats['nwords'] += nwords

            # report
            if b > 0 and b % report_freq == 0:
                losses = ""
                for lang, loss in sorted(stats['loss'].items()):
                    losses += "{}:{:.3f}  ".format(lang, loss / stats['batches'][lang])
                logging.info("Batch [{}] || {} || {:.0f} words/sec".format(
                    b, losses, stats['nwords'] / (time.time() - stats['start'])))
                # reset
                stats = {}
                stats['loss'] = collections.defaultdict(float)
                stats['batches'] = collections.defaultdict(int)
                stats['start'] = time.time()
                stats['nwords'] = 0

    def train_epochs(self, dataset, devreaders, epochs, sampling, clip_norm, report_freq,
                     patience, lr_factor, optimizer, **kwargs):
        optim = getattr(torch.optim, optimizer)(self.parameters(), **kwargs)
        best = tries = 0
        best_params = self.state_dict()
        for k, v in best_params.items():
            best_params[k] = v.to('cpu')

        for epoch in range(epochs):
            epoch += 1
            print("Starting epoch", epoch)

            # train epoch
            self.train()
            if sampling == 'oversampling':
                batches = dataset.get_batches_oversampling()
            else:
                batches = dataset.get_batches_balanced()
            self.train_epoch(batches, optim, clip_norm, report_freq)
            self.eval()

            # evaluate
            total = 0
            for lang in sorted(devreaders):
                acc, acc_amb, acc_unk = self.evaluate(readers[lang], lang)
                total += acc
                print("Lang:", lang)
                print("* Overall accuracy: {:.4f}".format(acc))
                print("* Ambiguous accuracy: {:.4f}".format(acc_amb))
                print("* Unknown accuracy: {:.4f}".format(acc_unk))
                print()
            acc = total / len(self.encoder.languages)
            print("* Epoch {} => Mean accuracy {:.4f}".format(epoch, acc))

            # monitor
            if acc - best <= 0.0001:
                tries += 1
                optim.param_groups[0]['lr'] *= lr_factor
                print("* New learning rate: {:.4f}".format(optim.param_groups[0]['lr']))
            else:
                tries = 0
                best = acc
                best_params = self.state_dict()
                for k, v in best_params.items():
                    best_params[k] = v.to('cpu')

            # early stopping
            if tries == patience:
                logging.info("Finished training after {} tries!".format(patience))
                logging.info("Best dev accuracy {:.4f}".format(best))
                break

        return best_params


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--langs", nargs='+',
                        default=("ca", "fr", "gl", "pt", "es", "it", "ro"))
    parser.add_argument("--prepend", nargs="*", default=("char", "lemma"))
    # training
    parser.add_argument("--sampling", default="oversampling")
    parser.add_argument("--batch_size", default=25, type=int)
    parser.add_argument("--dropout", default=0.25, type=float)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--lr_factor", default=0.75, type=float)
    parser.add_argument("--patience", default=3, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default='cpu')
    parser.add_argument("--clip_norm", default=5, type=float)
    parser.add_argument("--optimizer", default='Adam')
    # model
    parser.add_argument("--cell", default="GRU")
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--hidden_size", default=150, type=int)
    parser.add_argument("--wemb_dim", default=0, type=int)
    parser.add_argument("--cemb_dim", default=300, type=int)
    parser.add_argument("--cemb_type", default="rnn")
    parser.add_argument("--cemb_layers", default=2, type=int)
    # report
    parser.add_argument("--report_freq", default=200, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--results_path", default="results.csv")

    args = parser.parse_args()
    print(args.langs, args.prepend)

    if args.sampling not in ("oversampling", "balanced"):
        raise ValueError("Unknown sampling method", args.sampling)

    def get_paths(langpath, splits=('train', 'test', 'dev')):
        paths = glob.glob(os.path.join(langpath, '*conllu'))
        output = []
        for split in splits:
            for path in paths:
                if split in path:
                    output.append(path)

        return output

    root = 'datasets/ud-treebanks-v2.2/'
    langs = {
        'ca': 'UD_Catalan-AnCora',
        'fr': 'UD_French-GSD',
        'gl': 'UD_Galician-CTG',
        'pt': 'UD_Portuguese-GSD',
        'es': 'UD_Spanish-GSD',
        'it': 'UD_Italian-ISDT',
        'ro': 'UD_Romanian-RRT'}
    langs = {lang: langs[lang] for lang in args.langs}

    print("Creating readers for langs,", sorted(langs))
    readers = {lang: LanguageReader(get_paths(os.path.join(root, path))[0])
               for lang, path in langs.items()}
    encoder = MultiLanguageEncoder(list(langs), prepend=args.prepend)
    encoder.fit_readers(readers)
    dataset = MultiLanguageDataset(encoder, readers, args.batch_size, args.device)
    devreaders = {lang: LanguageReader(get_paths(os.path.join(root, path))[2])
                  for lang, path in langs.items()}

    print("Creating model")
    model = Model(encoder, args.cemb_dim, args.hidden_size, args.num_layers,
                  dropout=args.dropout, cemb_layers=args.cemb_layers, cell=args.cell)
    print(model)
    print("* Parameters", sum(p.nelement() for p in model.parameters()))
    model.to(args.device)

    print("Starting training")
    params = model.train_epochs(dataset, devreaders, args.epochs, args.sampling,
                                args.clip_norm, args.report_freq,
                                args.patience, args.lr_factor,
                                # optimizer params
                                args.optimizer, lr=args.lr)

    if not args.test:
        print("Saving results to '{}'".format(args.results_path))
        model.eval()
        model.load_state_dict(params)
        model.to(args.device)

        args = vars(args)
        metrics = ['all-accuracy', 'amb-accuracy', 'unk-accuracy']
        header = sorted(list(args) + ['lang', 'time']) + metrics
        run = time.time()
        exists = os.path.isfile(args['results_path'])
        with open(args['results_path'], 'a+') as f:
            if not exists:
                f.write('\t'.join(header) + '\n')
            else:
                f.seek(0)       # go to start
                header = next(f).strip().split('\t')
                f.readlines()   # go to end

            for lang, reader in devreaders.items():
                acc = dict(zip(metrics, model.evaluate(reader, lang)))
                row = []
                for key in header:
                    if key in metrics:
                        row.append(acc[key])
                    elif key in args:
                        val = args[key]
                        if isinstance(val, (tuple, list)):
                            val = '-'.join(sorted(val))
                        row.append(val)
                    elif key == 'lang':
                        row.append(lang)
                    elif key == 'time':
                        row.append(run)
                    else:
                        print("Missing arg", key)
                        row.append("NA")
                f.write('\t'.join(map(str, row)) + '\n')
