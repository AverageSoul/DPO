"""Audit HH split sizes and reserve every normalized test prompt for evaluation."""
import argparse
import hashlib
import json
import os
import unicodedata


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split())


def split_hh_row(row):
    """Use the last shared Assistant boundary; responses can contain later turns."""
    marker = '\n\nAssistant:'
    chosen, rejected = row['chosen'], row['rejected']
    common = os.path.commonprefix([chosen, rejected])
    index = common.rfind(marker)
    if index < 0:
        raise ValueError('Invalid HH row: no shared Assistant prompt boundary')
    end = index + len(marker)
    return chosen[:end], chosen[end:], rejected[end:]


def row_identity(row):
    prompt, chosen, rejected = split_hh_row(row)
    prompt = normalized(prompt)
    responses = sorted([normalized(chosen), normalized(rejected)])
    # Ignore preference direction when detecting duplicate test pairs.
    identity = hashlib.sha256(json.dumps([prompt, responses], ensure_ascii=False).encode()).hexdigest()
    return prompt, identity


def audit_hh_splits(train, test):
    test_prompts, seen_pairs, test_indices = set(), set(), []
    for index, row in enumerate(test):
        prompt, pair = row_identity(row)
        test_prompts.add(prompt)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            test_indices.append(index)
    train_indices, overlap_prompts = [], set()
    for index, row in enumerate(train):
        prompt, _ = row_identity(row)
        if prompt in test_prompts:
            overlap_prompts.add(prompt)
        else:
            train_indices.append(index)
    report = dict(train_raw_pairs=len(train), test_raw_pairs=len(test),
                  test_unique_pairs=len(test_indices), test_unique_prompts=len(test_prompts),
                  test_duplicate_pairs_removed=len(test) - len(test_indices),
                  overlapping_prompts_removed_from_train=len(overlap_prompts),
                  train_pairs_removed=len(train) - len(train_indices),
                  train_clean_pairs=len(train_indices), remaining_prompt_overlap=0,
                  normalization='Unicode NFKC + collapsed whitespace', policy_version=1)
    def select(rows, indices):
        return rows.select(indices) if hasattr(rows, 'select') else [rows[i] for i in indices]
    return {'train': select(train, train_indices), 'test': select(test, test_indices)}, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', required=True)
    args = parser.parse_args()
    from preference_datasets import get_audited_hh_splits
    _, report = get_audited_hh_splits(dataset_path=args.dataset_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['test_unique_pairs'] < 1000:
        raise SystemExit('Insufficient unique test pairs for a 1000-pair evaluation')
