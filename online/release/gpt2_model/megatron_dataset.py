import os
import torch
import logging
import numpy as np
from torch.utils.data import Dataset


def logger(msg):
    logging.info(msg)


class GPTDataset(Dataset):
    def __init__(self, args, data_prefix, documents, indexed_dataset, num_epochs=None):
        self.indexed_dataset = indexed_dataset
        self.seq_length = args.max_len
        self.seed = args.seed
        if num_epochs is None:
            num_epochs = args.epochs
        # Build index mappings.
        self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
            data_prefix, documents, self.indexed_dataset.sizes, num_epochs,
            self.seq_length, self.seed)

    def __len__(self):
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        return self.sample_idx.shape[0] - 1
    
    def __getitem__(self, idx):
        # Get the shuffled index.
        idx = self.shuffle_idx[idx]
        # Start and end documents and offsets.
        doc_index_f = self.sample_idx[idx][0]
        doc_index_l = self.sample_idx[idx + 1][0]
        offset_f = self.sample_idx[idx][1]
        offset_l = self.sample_idx[idx + 1][1]
        # If we are within the same document, just extract the chunk.
        if doc_index_f == doc_index_l:
            sample = self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                              offset=offset_f,
                                              length=offset_l - offset_f + 1)
        else:
            # Otherwise, get the rest of the initial document.
            sample_list = [self.indexed_dataset.get(self.doc_idx[doc_index_f],
                                                    offset=offset_f)]
            # Loop over all in between documents and add the entire document.
            for i in range(doc_index_f + 1, doc_index_l):
                sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
            # And finally add the relevant portion of last document.
            sample_list.append(self.indexed_dataset.get(
                self.doc_idx[doc_index_l],
                length=offset_l + 1))
            sample = np.concatenate(sample_list)
        return np.array(sample, dtype=np.int64)


def _build_index_mappings(data_prefix, documents, sizes, num_epochs,
                          seq_length, seed):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = np.sum(sizes[documents])
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = "{}_indexmap_{}ns_{}sl_{}s".format(data_prefix, num_epochs, seq_length, seed)
    doc_idx_filename = _filename + '_doc_idx.npy'
    sample_idx_filename = _filename + '_sample_idx.npy'
    shuffle_idx_filename = _filename + '_shuffle_idx.npy'

    # Build the indexed mapping if not exist.
    # Maybe popdist.rank is needed
    if (not os.path.isfile(doc_idx_filename)) or \
        (not os.path.isfile(sample_idx_filename)) or \
        (not os.path.isfile(shuffle_idx_filename)):

        # print_rank_0(' > WARNING: could not find index map files, building '
        #              'the indices on rank 0 ...')

        # doc-idx.
        doc_idx = _build_doc_idx(documents, num_epochs, np_rng)
        np.save(doc_idx_filename, doc_idx, allow_pickle=True)
        # sample-idx.
        # Megatron Used C++ implementation for speed here.
        assert doc_idx.dtype == np.int32
        assert sizes.dtype == np.int32
        sample_idx = _build_sample_idx(sizes, doc_idx, seq_length,
                                        num_epochs, tokens_per_epoch)
        np.save(sample_idx_filename, sample_idx, allow_pickle=True)
        # shuffle-idx.
        # -1 is due to data structure used to retieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        num_samples_ = sample_idx.shape[0] - 1
        shuffle_idx = _build_shuffle_idx(num_samples_,
                                            sample_idx.shape[0] - 1, np_rng)
        np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)

    # Load mappings.
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode='r')
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode='r')
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r')
    logger('    total number of samples: {}'.format(sample_idx.shape[0]))
    logger('    total number of epochs: {}'.format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx


def _build_doc_idx(documents, num_epochs, np_rng, separate_last_epoch=False):
    """Build an array with length = number-of-epochs * number-of-dcuments.
    Each index is mapped to a corresponding document."""
    if not separate_last_epoch or num_epochs == 1:
        doc_idx = np.mgrid[0:num_epochs, 0:len(documents)][1]
        doc_idx[:] = documents
        doc_idx = doc_idx.reshape(-1)
        doc_idx = doc_idx.astype(np.int32)
        np_rng.shuffle(doc_idx)
        return doc_idx

    doc_idx_first = _build_doc_idx(documents, num_epochs-1, np_rng, False)
    doc_idx_last = _build_doc_idx(documents, 1, np_rng, False)
    return np.concatenate((doc_idx_first, doc_idx_last))


def _build_sample_idx(sizes, doc_idx, seq_length,
                      num_epochs, tokens_per_epoch):
    """Sample index mapping is a 2D array with sizes
    [number-of-samples + 1, 2] where [..., 0] contains
    the index into `doc_idx` and [..., 1] is the
    starting offset in that document."""

    # Total number of samples. For -1 see comments in `_num_epochs`.
    num_samples = (num_epochs * tokens_per_epoch - 1) // seq_length
    sample_idx = np.zeros([num_samples + 1, 2], dtype=np.int32)

    # Index into sample_idx.
    sample_index = 0
    # Index into doc_idx.
    doc_idx_index = 0
    # Begining offset for each document.
    doc_offset = 0
    # Start with first document and no offset.
    sample_idx[sample_index][0] = doc_idx_index
    sample_idx[sample_index][1] = doc_offset
    sample_index += 1
    while sample_index <= num_samples:
        # Start with a fresh sequence.
        remaining_seq_length = seq_length + 1
        while remaining_seq_length != 0:
            # Get the document length.
            doc_id = doc_idx[doc_idx_index]
            doc_length = sizes[doc_id] - doc_offset
            # And add it to the current sequence.
            remaining_seq_length -= doc_length
            # If we have more than a full sequence, adjust offset and set
            # remaining length to zero so we return from the while loop.
            # Note that -1 here is for the same reason we have -1 in
            # `_num_epochs` calculations.
            if remaining_seq_length <= 0:
                doc_offset += (remaining_seq_length + doc_length - 1)
                remaining_seq_length = 0
            else:
                # Otherwise, start from the begining of the next document.
                doc_idx_index += 1
                doc_offset = 0
        # Record the sequence.
        sample_idx[sample_index][0] = doc_idx_index
        sample_idx[sample_index][1] = doc_offset
        sample_index += 1

    return sample_idx


def _build_shuffle_idx(num_samples, total_size, np_rng):
    """Build the range [0, size) and shuffle."""
    print(' > building shuffle index with split [0, {}) and [{}, {}) '
          '...'.format(num_samples, num_samples, total_size), flush=True)
    
    dtype_ = np.uint32
    if total_size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64

    shuffle_idx_first = np.arange(start=0, stop=num_samples,
                                  step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_first)
    if num_samples == total_size:
        return shuffle_idx_first

    shuffle_idx_last = np.arange(start=num_samples, stop=total_size,
                                 step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx_last)

    return np.concatenate((shuffle_idx_first, shuffle_idx_last))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int, required=False, help='epochs for training')
    parser.add_argument('--max-len', default=1024, type=int, required=False, help='max length of input sequence')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    data_prefix="./data/my-gpt2_text_document"

    from indexed_dataset import make_indexed_dataset
    indexed_dataset = make_indexed_dataset(data_prefix)
    total_num_of_documents = indexed_dataset.sizes.shape[0]
    documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32)
    gpt2_dataset = GPTDataset(args, data_prefix, documents, indexed_dataset)

    gpt2_dataloader = torch.utils.data.DataLoader(gpt2_dataset, batch_size=8, collate_fn=None)

    import pdb
    for batch in gpt2_dataloader:
        pdb.set_trace()
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        # logits = model(input_ids, labels)
