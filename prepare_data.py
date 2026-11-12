#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

QUESTION_MARKER = '\n\nAnswer the question based on the given documents.'


def load_rows(path):
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix.lower() == '.json':
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, list) else value.get('data', [])
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def doc_prompt(documents):
    return '\n'.join(f'Document {i}:\n{text}' for i, text in enumerate(documents, 1))


def make_input(question, documents, instruction):
    return f"{instruction}\n{doc_prompt(documents)}{QUESTION_MARKER}\nQuestion: {question}\nAnswer:"


def musique_docs(row):
    paragraphs = row.get('paragraphs', [])
    gold, all_docs = [], []
    for paragraph in paragraphs:
        title = str(paragraph.get('title', '')).strip()
        body = str(paragraph.get('paragraph_text', '')).strip()
        if not body:
            continue
        text = f'{title}\n{body}' if title else body
        all_docs.append((title, text))
        if paragraph.get('is_supporting') is True:
            gold.append((title, text))
    return gold, all_docs


def twowiki_docs(row):
    context = row.get('context', [])
    if isinstance(context, str):
        context = json.loads(context)
    support = {str(item[0]) for item in row.get('supporting_facts', []) if isinstance(item, (list, tuple)) and item}
    gold, all_docs = [], []
    for item in context:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        title, sentences = str(item[0]), item[1]
        body = ' '.join(sentences) if isinstance(sentences, list) else str(sentences)
        text = f'{title}\n{body}'.strip()
        if not body.strip():
            continue
        all_docs.append((title, text))
        if title in support:
            gold.append((title, text))
    return gold, all_docs


def normalize_outputs(row):
    value = row.get('outputs', row.get('answer', row.get('answers')))
    if isinstance(value, dict):
        value = value.get('text', [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def ruler_qa1(rows):
    articles = rows
    corpus = []
    questions = []
    for article in articles:
        article_docs = []
        for paragraph in article.get('paragraphs', []):
            text = paragraph.get('context', '')
            if text and text not in corpus:
                corpus.append(text)
            if text and text not in article_docs:
                article_docs.append(text)
            for qa in paragraph.get('qas', []):
                if qa.get('is_impossible', False):
                    continue
                answers = [str(a.get('text', '')) for a in qa.get('answers', []) if a.get('text')]
                if answers:
                    questions.append((qa['question'], answers, text, article_docs))
    return questions, corpus


def ruler_qa2(rows):
    questions = []
    corpus = {}
    for row in rows:
        context = row.get('context', [])
        if isinstance(context, str):
            context = json.loads(context)
        support = {str(x[0]) for x in row.get('supporting_facts', []) if isinstance(x, (list, tuple)) and x}
        docs = []
        for title, sentences in context:
            title = str(title)
            body = ' '.join(sentences) if isinstance(sentences, list) else str(sentences)
            text = f'{title}\n{body}'.strip()
            if not body.strip():
                continue
            corpus.setdefault((title, text), None)
            docs.append((title, text))
        gold = [text for title, text in docs if title in support]
        if row.get('question') and gold:
            questions.append((row['question'], [str(row.get('answer', ''))], gold, docs))
    return questions, [text for _, text in corpus]


def build(args):
    rows = load_rows(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)
    if args.kind == 'ruler':
        output = []
        for pos, row in enumerate(rows[:args.count or None]):
            text = row.get('input')
            if not isinstance(text, str) or 'Document 1:\n' not in text or QUESTION_MARKER not in text:
                raise ValueError(f'RULER row {pos} is not a prepared CacheMend prompt')
            output.append({'index': row.get('index', pos), 'input': text,
                           'outputs': normalize_outputs(row)})
        return output

    if args.kind in ('ruler_qa1', 'ruler_qa2'):
        if args.kind == 'ruler_qa1':
            questions, corpus = ruler_qa1(rows)
        else:
            questions, corpus = ruler_qa2(rows)
        result = []
        for index, item in enumerate(questions[:args.count or None]):
            question, answers, gold, local_docs = item
            if isinstance(gold, str):
                gold = [gold]
            if args.gold_document_count and len(gold) != args.gold_document_count:
                continue
            candidates = [doc for doc in corpus if doc not in gold]
            if args.kind == 'ruler_qa1':
                candidates = list(dict.fromkeys(local_docs + candidates))
                candidates = [doc for doc in candidates if doc not in gold]
            rng.shuffle(candidates)
            documents = list(gold)
            for candidate in candidates:
                trial = [*documents, candidate]
                if args.target_tokens and len(tokenizer(make_input(question, trial, args.instruction), add_special_tokens=False)['input_ids']) > args.target_tokens:
                    continue
                documents.append(candidate)
            text = make_input(question, documents, args.instruction)
            if args.target_tokens and len(tokenizer(text, add_special_tokens=False)['input_ids']) > args.target_tokens:
                raise ValueError(f'Gold evidence for row {index} exceeds target token count')
            result.append({'index': index, 'input': text, 'outputs': answers})
        return result

    extractor = musique_docs if args.kind == 'musique' else twowiki_docs
    parsed = []
    pool = {}
    for row in rows:
        question = row.get('question', row.get('query'))
        gold, documents = extractor(row)
        if (not isinstance(question, str) or not question.strip() or not gold
                or args.gold_document_count and len(gold) != args.gold_document_count):
            continue
        parsed.append((row, question, gold, documents))
        for title, text in documents:
            pool.setdefault((title, text), None)
    pool = list(pool)
    result = []
    for source_index, (row, question, gold, local_docs) in enumerate(parsed[:args.count or None]):
        gold_titles = {title for title, _ in gold}
        candidates = [doc for doc in pool if doc[0] not in gold_titles and doc not in gold]
        rng.shuffle(candidates)
        documents = list(gold)
        if args.target_tokens:
            for candidate in candidates:
                trial = [text for _, text in documents + [candidate]]
                if len(tokenizer(make_input(question, trial, args.instruction), add_special_tokens=False)['input_ids']) > args.target_tokens:
                    continue
                documents.append(candidate)
        elif args.kind == 'musique':
            documents.extend(doc for doc in local_docs if doc not in gold)
        text = make_input(question, [body for _, body in documents], args.instruction)
        if args.target_tokens and len(tokenizer(text, add_special_tokens=False)['input_ids']) > args.target_tokens:
            raise ValueError(f'Gold evidence for row {source_index} exceeds target token count')
        result.append({'index': row.get('index', source_index), 'input': text,
                       'outputs': normalize_outputs(row)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('ruler', 'ruler_qa1', 'ruler_qa2', 'musique', '2wiki'), required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--target-tokens', type=int, required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--instruction', required=True)
    parser.add_argument('--gold-document-count', type=int, required=True)
    args = parser.parse_args()
    if args.target_tokens < 0 or args.count < 0 or args.gold_document_count < 0:
        parser.error('target-tokens, count, and gold-document-count must be nonnegative')
    output = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Wrote {len(output)} rows to {args.output}')


if __name__ == '__main__':
    main()
