"""Pure source freezer and explicit inference adapter for Control/Shuffle/LBSP.

Importing this module uses the Python standard library only. It does not read
files, import Torch/NumPy, inspect/load weights, instantiate a model or infer.
The caller must hash-check original source files before passing their text,
write/hash the returned definition module, and load it only on the server.
"""
import ast
from numbers import Integral

VERSION = 'x3_inference_adapter_v1'
MAX_LEN = 50
METHODS = ('control', 'shuffle', 'lbsp')


def method_name(method):
    aliases = {'control': 'control', 'shuffle': 'shuffle', 'trainingshuffle': 'shuffle',
               'training_shuffle': 'shuffle', 'lbsp': 'lbsp'}
    if not isinstance(method, str) or method.lower() not in aliases:
        raise ValueError('Method must be control, shuffle, TrainingShuffle or lbsp')
    return aliases[method.lower()]


def model_class_name(method):
    return 'RecentBucketSetR1SASRec' if method_name(method) == 'lbsp' else 'SASRecControl'


def _definition(source, name):
    if not isinstance(source, str):
        raise TypeError('Pass verified source text, never a source path')
    found = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
    if len(found) != 1:
        raise ValueError('Expected one source definition: ' + name)
    return found[0]


def build_frozen_module(control_text, lbsp_text=None, lbsp_forward_text=None):
    """Freeze whitelisted definitions; never execute original top-level code.

    Control-only operation needs the original 38_1 text or the audited X3
    model_definition text. LBSP additionally needs audit33_2 and the safe
    masked-forward 38_20 text. Source-byte SHA verification is the caller's
    responsibility. All selected function/class ASTs remain exact.
    """
    if (lbsp_text is None) != (lbsp_forward_text is None):
        raise ValueError('LBSP class and safe-forward sources must be supplied together')
    selected = [(control_text, _definition(control_text, 'SASRecControl'))]
    if lbsp_text is not None:
        for name in ('build_r1_group_positions_from_size', 'canonicalize_last_bucket_items',
                     'RecentBucketSetR1SASRec'):
            selected.append((lbsp_text, _definition(lbsp_text, name)))
        for name in ('require', 'predict_batch_masked'):
            selected.append((lbsp_forward_text, _definition(lbsp_forward_text, name)))
    result = '''# Frozen definitions only. No archived training/evaluation top-level execution.
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
MAX_LEN = 50

'''
    result += '\n\n'.join(ast.get_source_segment(text, node) for text, node in selected) + '\n'
    if lbsp_text is not None:
        result += '\nRecentBucketSetR1SASRec.predict_batch_masked = predict_batch_masked\n'
    rebuilt = [node for node in ast.parse(result).body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    if [ast.dump(node) for _, node in selected] != [ast.dump(node) for node in rebuilt]:
        raise RuntimeError('Frozen scientific definition AST changed')
    ast.parse(result, feature_version=(3, 10))
    return result


def prepare_history(method, history, last_k=None, max_len=MAX_LEN):
    """Prepare one already-mapped visible history as Python lists.

    IMPORTANT: both old model families receive right-aligned events with LEFT
    padding, matching their original CSV loaders. Control's masked forward
    rotates these events internally. This is different from X4 TiSASRec.
    No truncation, item deduplication, target removal, or mask is applied here.
    The caller must preserve the sealed visible cutoff and supply last_k after
    any protocol-required truncation/label validation.
    """
    method = method_name(method)
    if isinstance(max_len, bool) or not isinstance(max_len, Integral) or not 1 <= max_len <= MAX_LEN:
        raise ValueError('max_len must be an integer in 1..50')
    max_len = int(max_len)
    values = list(history)
    if not 1 <= len(values) <= max_len:
        raise ValueError('Visible history must already contain 1..max_len events')
    if any(isinstance(item, bool) or not isinstance(item, Integral) or item < 1 for item in values):
        raise ValueError('History needs mapped positive integer item indices; raw IDs/padding are invalid')
    values = [int(item) for item in values]
    length = len(values)
    if method == 'lbsp':
        if isinstance(last_k, bool) or not isinstance(last_k, Integral) or not 1 <= last_k <= length:
            raise ValueError('LBSP requires a validated visible last_k in 1..history_length')
        last_k = int(last_k)
        if last_k >= 2:
            prefix = length - last_k
            # Stable numeric item-index canonicalization, preserving repeats.
            values[prefix:] = sorted(values[prefix:])
            event_groups = list(range(prefix)) + [prefix] * last_k
            tokens = prefix + 1
        else:
            event_groups, tokens = list(range(length)), length
    else:
        event_groups, tokens = list(range(length)), length
    padding = max_len - length
    return {'seq': [0] * padding + values, 'groups': [-1] * padding + event_groups,
            'num_tokens': tokens}


def score_batch(model, method, seq, lengths, groups=None, num_tokens=None):
    """Return RAW unmasked [B,N+1] candidate logits from an eval-mode model.

    The caller owns inference_mode/no_grad, batching (retain original BS256
    partition for reproduction), device transfer, finite-score checks, class0
    and seen-item masking, target ranks, Top10, averaging, and result writes.
    In particular mean-score ensembles must average raw scores and then apply
    their shared canonical candidate mask. No score normalization is added.
    No Torch import or checkpoint operation occurs in this function.
    """
    method = method_name(method)
    if getattr(model, 'training', True):
        raise ValueError('Call model.eval() before inference scoring')
    if len(seq.shape) != 2 or seq.shape[1] != getattr(model, 'max_len', None):
        raise ValueError('Batch sequence width must equal the frozen model max_len')
    if len(lengths.shape) != 1 or lengths.shape[0] != seq.shape[0]:
        raise ValueError('Lengths must match the batch')
    if method == 'lbsp':
        if groups is None or num_tokens is None or groups.shape != seq.shape:
            raise ValueError('LBSP requires aligned groups and num_tokens')
        if len(num_tokens.shape) != 1 or num_tokens.shape[0] != seq.shape[0]:
            raise ValueError('LBSP token counts must match the batch')
        scores = model.predict_batch_masked(seq, groups, num_tokens)
    else:
        scores = model.predict_batch_masked(seq, lengths)
    if len(scores.shape) != 2 or scores.shape != (seq.shape[0], model.num_items + 1):
        raise RuntimeError('Model returned an unexpected candidate-score shape')
    return scores
