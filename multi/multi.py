

import tarfile
import json
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
import dataset


class Model(nn.Module):
    def __init__(self, encoder, cemb_dim, hidden_size, num_layers=0, dropout=0.0,
                 cemb_layers=1, cell='GRU', init_rnn='xavier_uniform', scorer='general',
                 prepend_dim=0):
        self.encoder = encoder
        self.dropout = dropout
        self.cemb_dim = cemb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cemb_layers = cemb_layers
        self.cell = cell
        self.scorer = scorer
        self.prepend_dim = prepend_dim
        super().__init__()

        # Embeddings
        embedding_dim = None
        if self.encoder.share_encoder:
            encoder = self.encoder.get_lang_encoder(self.encoder.get_random_lang()).char
            self.cemb = RNNEmbedding(
                len(encoder), cemb_dim,
                padding_idx=encoder.get_pad(), dropout=dropout,
                num_layers=cemb_layers, cell=cell, init_rnn=init_rnn)
            embedding_dim = self.cemb.embedding_dim
        else:
            cemb = {}
            for lang, encoder in self.encoder.char.items():
                lang_cemb = RNNEmbedding(
                    len(encoder), cemb_dim,
                    padding_idx=encoder.get_pad(), dropout=dropout,
                    num_layers=cemb_layers, cell=cell, init_rnn=init_rnn)
                self.add_module('{}-cemb'.format(lang), lang_cemb)
                cemb[lang] = lang_cemb
            self.cemb = cemb
            embedding_dim = lang_cemb.embedding_dim

        # Encoder
        self.rnn_encoder = self.lang_encoder = self.lang_w2i = None
        context_dim = 0
        if num_layers > 0:
            in_dim = embedding_dim
            if prepend_dim:
                self.lang_encoder = nn.Embedding(
                    len(self.encoder.languages), prepend_dim)
                in_dim += prepend_dim
                self.lang_w2i = {
                    lang: idx for idx, lang in enumerate(self.encoder.languages)}
            self.rnn_encoder = RNNEncoder(in_dim, hidden_size, num_layers=num_layers,
                                          cell=cell, dropout=dropout, init_rnn=init_rnn)
            context_dim = hidden_size * 2

        # Decoder
        if self.encoder.share_decoder:
            encoder = self.encoder.get_lang_encoder(self.encoder.get_random_lang())
            self.decoder = AttentionalDecoder(
                encoder.lemma, cemb_dim, embedding_dim,
                context_dim=context_dim,
                scorer=scorer, num_layers=cemb_layers, cell=cell,
                dropout=dropout, init_rnn=init_rnn)
        else:
            decoder = {}
            for lang, encoder in self.encoder.lemma.items():
                lang_decoder = AttentionalDecoder(
                    encoder, cemb_dim, embedding_dim,
                    # fix for now
                    context_dim=context_dim,
                    scorer=scorer, num_layers=cemb_layers, cell=cell,
                    dropout=dropout, init_rnn=init_rnn)
                self.add_module('{}-decoder'.format(lang), lang_decoder)
                decoder[lang] = lang_decoder
            self.decoder = decoder

    def get_args_and_kwargs(self):
        return {"args": [self.cemb_dim, self.hidden_size],
                "kwargs": {
                    "dropout": self.dropout,
                    "num_layers": self.num_layers,
                    "cemb_layers": self.cemb_layers,
                    "cell": self.cell,
                    "scorer": self.scorer,
                    "prepend_dim": self.prepend_dim}}

    def save(self, fpath, settings):
        """
        Serialize model to path
        """
        import pie
        fpath = pie.utils.ensure_ext(fpath, 'tar')

        # create dir if necessary
        dirname = os.path.dirname(fpath)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        with tarfile.open(fpath, 'w') as tar:
            # serialize label_encoder
            string = json.dumps(self.encoder.to_dict())
            path = 'label_encoder.zip'
            pie.utils.add_gzip_to_tar(string, path, tar)

            # serialize parameters
            string, path = json.dumps(self.get_args_and_kwargs()), 'parameters.zip'
            pie.utils.add_gzip_to_tar(string, path, tar)

            # serialize weights
            pie.utils.add_weights_to_tar(self.state_dict(), 'state_dict.pt', tar)

            # if passed, serialize settings
            string, path = json.dumps(settings), 'settings.zip'
            pie.utils.add_gzip_to_tar(string, path, tar)

        return fpath

    @classmethod
    def load(cls, fpath):
        import pie

        with tarfile.open(pie.utils.ensure_ext(fpath, 'tar'), 'r') as tar:
            # load label encoder
            le = dataset.MultiLanguageEncoder.from_dict(
                json.loads(pie.utils.get_gzip_from_tar(tar, 'label_encoder.zip')))
            # load model parameters
            params = json.loads(pie.utils.get_gzip_from_tar(tar, 'parameters.zip'))
            print(params)
            inst = cls(le, *params['args'], **params['kwargs'])
            inst.load_state_dict(
                torch.load(tar.extractfile('state_dict.pt'), map_location='cpu'))
            inst.eval()

        return inst, le

    def device(self):
        return next(self.parameters()).device

    def loss(self, batch_data, lang):
        ((_, wlen), (char, clen)), (lemma, llen) = batch_data

        # Embedding
        cemb = self.cemb if self.encoder.share_encoder else self.cemb[lang]
        cemb, cemb_outs = cemb(char, clen, wlen)

        # Encoder
        context = None
        if self.rnn_encoder is not None:
            rnn_inp = cemb
            if self.lang_encoder is not None:
                lang_emb = torch.zeros_like(cemb[:, :, 0]).long() + self.lang_w2i[lang]
                lang_emb = self.lang_encoder(lang_emb)
                rnn_inp = torch.cat([lang_emb, rnn_inp], dim=2)
            context = self.rnn_encoder(rnn_inp, wlen)[-1]
            context = torch_utils.flatten_padded_batch(context, wlen)

        # Decoder
        decoder = self.decoder if self.encoder.share_decoder else self.decoder[lang]
        cemb_outs = F.dropout(cemb_outs, p=self.dropout, training=self.training)
        logits = decoder(lemma, llen, cemb_outs, clen, context=context)
        loss = decoder.loss(logits, lemma)

        return loss, sum(wlen).item()

    def predict(self, inp, lang, use_beam=False, width=10):
        encoder = self.encoder.get_lang_encoder(lang)
        # get modules
        cemb = self.cemb if self.encoder.share_encoder else self.cemb[lang]
        decoder = self.decoder if self.encoder.share_decoder else self.decoder[lang]
        # prepend with lang id
        bos = encoder.lemma.table[lang] if 'lemma' in self.encoder.prepend else None

        (_, wlen), (char, clen) = inp
        # Embedding
        cemb, cemb_outs = cemb(char, clen, wlen)

        # Encoder
        context = None
        if self.rnn_encoder is not None:
            rnn_inp = cemb
            if self.lang_encoder is not None:
                lang_emb = torch.zeros_like(cemb[:, :, 0]).long() + self.lang_w2i[lang]
                lang_emb = self.lang_encoder(lang_emb)
                rnn_inp = torch.cat([lang_emb, rnn_inp], dim=2)
            context = self.rnn_encoder(rnn_inp, wlen)[-1]
            context = torch_utils.flatten_padded_batch(context, wlen)

        # Decoder
        if use_beam:
            hyps, _ = decoder.predict_beam(
                cemb_outs, clen, context=context, width=width, bos=bos)
        else:
            hyps, _ = decoder.predict_max(cemb_outs, clen, context=context, bos=bos)

        if encoder.lemma.preprocessor_fn is None:
            hyps = [''.join(hyp) for hyp in hyps]

        return hyps

    def evaluate(self, reader, lang, batch_size=50, **kwargs):
        assert not self.training
        encoder = self.encoder.get_lang_encoder(lang)
        trues, preds = [], []

        with torch.no_grad(), tqdm.tqdm() as pbar:
            it = dataset.BatchIterator(reader, batch_size)
            batches = iter(it)
            while not it.done:
                pbar.update(batch_size)
                sents, tasks = next(batches)
                inp, target = dataset.prepare_batch(
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

    def train_batches(self, dataset, devreaders, optimizer, clip_norm,
                      report_freq, check_freq, patience, lr_factor,
                      scheduler=None, target=None, **kwargs):

        optim = getattr(torch.optim, optimizer)(self.parameters(), **kwargs)
        best = tries = 0
        best_params = self.state_dict()
        for k, v in best_params.items():
            best_params[k] = v.to('cpu')
        # report
        stats = {}
        stats['loss'] = collections.defaultdict(float)
        stats['batches'] = collections.defaultdict(int)
        stats['start'] = time.time()
        stats['nwords'] = 0

        for b, (batch_data, lang) in enumerate(dataset.get_unlimited_batches()):

            self.train()
            # loss
            loss, nwords = self.loss(batch_data, lang)

            # optimize
            optim.zero_grad()
            (scheduler.apply_weight(lang, loss) if scheduler else loss).backward()
            if clip_norm > 0:
                nn.utils.clip_grad_norm_(self.parameters(), clip_norm)
            optim.step()

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

            # evaluate
            if b > 0 and b % check_freq == 0:
                self.eval()
                total = 0
                for lang in sorted(devreaders):
                    acc, acc_amb, acc_unk = self.evaluate(devreaders[lang], lang)
                    if scheduler is not None:
                        scheduler.step(lang, acc)
                    if target is not None:
                        if lang == target:
                            total = acc
                    else:
                        total += acc
                    print("Lang:", lang)
                    print("* Overall accuracy: {:.4f}".format(acc))
                    print("* Ambiguous accuracy: {:.4f}".format(acc_amb))
                    print("* Unknown accuracy: {:.4f}".format(acc_unk))
                    print()

                acc = total / (len(self.encoder.languages) if target is None else 1)
                print("* => Mean accuracy {:.4f}".format(acc))

                if scheduler is not None:
                    print(scheduler)

                # monitor
                if acc - best <= 0.0001:
                    tries += 1
                    if tries > 5:
                        optim.param_groups[0]['lr'] *= lr_factor
                        print("* New learning rate: {:.8f}".format(
                            optim.param_groups[0]['lr']))
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

    def train_epoch(self, batches, optimizer, clip_norm, report_freq, scheduler=None):
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
            (scheduler.apply_weight(lang, loss) if scheduler else loss).backward()
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

    def train_epochs(self, dataset, devreaders, epochs, clip_norm, report_freq, patience,
                     lr_factor, lr_patience, optimizer, scheduler=None, target=None,
                     **kwargs):
        optim = getattr(torch.optim, optimizer)(self.parameters(), **kwargs)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='max', factor=lr_factor, patience=lr_patience, min_lr=1e-6)
        best = tries = 0
        best_params = self.state_dict()
        for k, v in best_params.items():
            best_params[k] = v.to('cpu')

        for epoch in range(epochs):
            epoch += 1
            print("Starting epoch", epoch)

            # train epoch
            self.train()
            self.train_epoch(
                dataset.get_batches(), optim, clip_norm, report_freq,
                scheduler=scheduler)
            self.eval()

            # evaluate
            total = 0
            for lang in sorted(devreaders):
                acc, acc_amb, acc_unk = self.evaluate(devreaders[lang], lang)
                if scheduler is not None:
                    scheduler.step(lang, acc)
                if target is not None:
                    if lang == target:
                        total = acc
                else:
                    total += acc
                print("Lang:", lang)
                print("* Overall accuracy: {:.4f}".format(acc))
                print("* Ambiguous accuracy: {:.4f}".format(acc_amb))
                print("* Unknown accuracy: {:.4f}".format(acc_unk))
                print()
            acc = total / (len(self.encoder.languages) if target is None else 1)
            print("* Epoch {} => Mean accuracy {:.4f}".format(epoch, acc))

            lr_scheduler.step(acc)
            if scheduler is not None:
                print(scheduler)
                print("* Learning rate: {:.8f}".format(optim.param_groups[0]['lr']))

            # monitor
            if acc - best <= 0.0001:
                tries += 1
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


class Spec:
    def __init__(self, mode='max', min_weight=0, factor=1, patience=2):
        self.mode = mode
        self.min_weight = min_weight
        self.factor = factor
        self.patience = patience
        # state
        self.weight = 1
        self.fails = 0
        self.best = float('inf')
        if mode == 'max':
            self.best = -self.best

    def step(self, score):
        if (score > self.best and self.mode == 'min') \
           or (score < self.best and self.mode == 'max'):
            self.fails += 1
        else:
            self.fails = 0
            self.best = score

        # update weight
        if self.fails >= self.patience:
            self.weight = max(self.min_weight, self.weight * self.factor)

    def apply_weight(self, loss):    # must be called after step
        return self.weight * loss


class Scheduler:
    def __init__(self, *specs, **kwargs):
        self.specs = {lang: Spec(**dict(kwargs, **spec)) for lang, spec in specs}

    def step(self, lang, score):
        self.specs[lang].step(score)

    def apply_weight(self, lang, loss):
        return self.specs[lang].apply_weight(loss)

    def __repr__(self):
        string = '<Scheduler>\n'
        keys = ['weight', 'fails', 'patience', 'best']
        for lang, spec in self.specs.items():
            string += '\t<{} {}/>\n'.format(
                lang, ' '.join('{}="{}"'.format(k, getattr(spec, k)) for k in keys))
        string += '</Scheduler>'
        return string


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--langs", nargs='+',
                        default=("ca", "fr", "gl", "pt", "es", "it", "ro"))
    parser.add_argument("--prepend", nargs="*", default=("char", "lemma"))
    parser.add_argument("--target_lang")
    # training
    parser.add_argument("--sampling", default="oversampling")
    parser.add_argument("--batch_size", default=25, type=int)
    parser.add_argument("--dropout", default=0.25, type=float)
    parser.add_argument("--lr", default=0.0008, type=float)
    parser.add_argument("--lr_factor", default=0.5, type=float)
    parser.add_argument("--lr_patience", default=2, type=int)
    parser.add_argument("--aux_patience", default=3, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--device", default='cpu')
    parser.add_argument("--clip_norm", default=5, type=float)
    parser.add_argument("--optimizer", default='Adam')
    # model
    parser.add_argument("--share", nargs='*', default=('encoder', 'decoder'))
    # add extra lang_id embedding as input to sentence encoder
    parser.add_argument("--prepend_dim", type=int, default=0)
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
    parser.add_argument("--disable_cudnn", action='store_true')

    args = parser.parse_args()
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    print("langs", args.langs)
    print("prepend", args.prepend)
    print("share", args.share)

    def get_paths(langpath, ext='conllu', splits=('train', 'test', 'dev')):
        paths = glob.glob(os.path.join(langpath, '*{}'.format(ext)))
        output = []
        for split in splits:
            for path in paths:
                if split in path:
                    output.append(path)

        return output

    # root = 'datasets/ud-treebanks-v2.2/'
    # langs = {
    #     'ca': 'UD_Catalan-AnCora',
    #     'fr': 'UD_French-GSD',
    #     'gl': 'UD_Galician-CTG',
    #     'pt': 'UD_Portuguese-GSD',
    #     'es': 'UD_Spanish-GSD',
    #     'it': 'UD_Italian-ISDT',
    #     'ro': 'UD_Romanian-RRT'}

    # historical
    root = 'datasets/'
    langs = {
        'cga': 'gysseling/cg-admin',
        'cgl': 'gysseling/cg-lit',
        'crm': 'gysseling/crm-adelheid',
        'rel': 'gysseling/relig',
        # 'ren': 'ren/'
    }

    langs = {lang: langs[lang] for lang in args.langs}

    print("Creating readers for langs,", sorted(langs))
    readers = {}
    for lang, path in langs.items():
        train, *_ = get_paths(os.path.join(root, path), ext='tab')
        print(train)
        readers[lang] = dataset.LanguageReader(train, linereader=dataset.readlines)

    encoder = dataset.MultiLanguageEncoder(
        list(readers), prepend=args.prepend,
        share_encoder='encoder' in args.share,
        share_decoder='decoder' in args.share)
    encoder.fit_readers(readers)
    if args.sampling == 'oversampling':
        trainset = dataset.MultiLanguageOversampling(
            encoder, readers, args.batch_size, args.device)
    elif args.sampling == 'balanced':
        trainset = dataset.MultiLanguageBalanced(
            encoder, readers, args.batch_size, args.device)
    else:
        raise ValueError("Unknown sampling type", args.sampling)
    devreaders = {
        lang: dataset.LanguageReader(
            get_paths(os.path.join(root, path), ext='tab')[2],
            linereader=dataset.readlines)  # only for historical languages
        for lang, path in langs.items()}

    print("Creating model")
    model = Model(encoder, args.cemb_dim, args.hidden_size, args.num_layers,
                  dropout=args.dropout, cemb_layers=args.cemb_layers, cell=args.cell,
                  prepend_dim=args.prepend_dim)
    print(model)
    print("* Parameters", sum(p.nelement() for p in model.parameters()))
    model.to(args.device)

    # scheduler
    target = scheduler = None
    if args.target_lang:
        target = args.target_lang
        assert target in langs, 'Unknown lang: {}'.format(target)
        scheduler = Scheduler(
            # don't down-weigh target
            *[(lang, {"min_weight": 1 if lang == target else 0}) for lang in langs],
            # params for all languages
            factor=0.5,
            patience=args.aux_patience)
        print(scheduler)

    print("Starting training")
    params = model.train_epochs(trainset, devreaders, args.epochs,
                                args.clip_norm, args.report_freq,
                                args.patience, args.lr_factor,
                                args.lr_patience, args.optimizer,
                                scheduler=scheduler, target=target,
                                # optimizer params
                                lr=args.lr)

    # params = model.train_batches(trainset, devreaders, args.optimizer,
    #                              args.clip_norm, args.report_freq, 500,
    #                              args.patience, args.lr_factor,
    #                              scheduler=scheduler, target= target,
    #                              # optimizer params
    #                              lr=args.lr)

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

        # serialize model
        model.save('models/' + str(run) + '.tar', args)
