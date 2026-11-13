import json
import re
from numbers import Integral
from pathlib import Path

DATASET_NAMES = ('custom', 'ruler', 'musique', '2wiki', 'longbench')
QUESTION_MARKER = '\n\nAnswer the question based on the given documents.'


def row_index(row, position):
    value = row.get('index')
    if value is None:
        return position
    if isinstance(value, bool):
        raise ValueError(f'Invalid sample index at file position {position}: {value!r}')
    if isinstance(value, Integral):
        index = int(value)
    elif isinstance(value, str) and re.fullmatch(r'[0-9]+', value):
        index = int(value)
    else:
        raise ValueError(f'Sample index must be a nonnegative integer at file position {position}')
    if index < 0:
        raise ValueError(f'Negative sample index at file position {position}')
    return index


def select_rows(rows, start=0, count=0):
    stop = len(rows) if count == 0 else min(len(rows), start + count)
    if start < 0 or count < 0 or start >= stop:
        raise ValueError('Requested row range is empty or invalid')
    selected, seen = [], set()
    for position in range(start, stop):
        row = rows[position]
        if not isinstance(row, dict):
            raise ValueError(f'Record at file position {position} must be an object')
        index = row_index(row, position)
        if index in seen:
            raise ValueError(f'Duplicate sample index {index}; cannot share a cache filename')
        seen.add(index)
        selected.append((position, index, row))
    return selected


def is_prepared(text):
    if not isinstance(text, str):
        return False
    header = re.search(r'(?m)^Document 1:\n', text)
    return header is not None and header.start() < text.rfind(QUESTION_MARKER)


def two_wiki_context(value):
    return isinstance(value, (list, tuple)) and bool(value) and all(
        isinstance(item, (list, tuple)) and len(item) == 2 and
        isinstance(item[0], str) and
        (isinstance(item[1], str) or
         isinstance(item[1], (list, tuple)) and all(isinstance(s, str) for s in item[1]))
        for item in value)


def load_rows(path):
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix.lower() == '.json':
        rows = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(rows, list):
            raise ValueError('JSON input must be an array of records; use JSONL for line records')
        return rows
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def split_document(text, tokenizer, chunk_tokens):
    if chunk_tokens == 0:
        return [text]
    offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
    # Slice the original text rather than decoding token slices, retaining spacing.
    boundaries = [0] + [int(offsets[i][0]) for i in range(chunk_tokens, len(offsets), chunk_tokens)] + [len(text)]
    return [text[a:b] for a, b in zip(boundaries, boundaries[1:]) if text[a:b].strip()]


def prepare_prompt(row, tokenizer, *, dataset_name='custom', data_format='auto',
                   chunk_tokens=0, instruction_prefix='Answer the question using the documents.'):
    if not isinstance(row, dict):
        raise ValueError('Each dataset record must be an object')
    if data_format == 'prepared' or (data_format == 'auto' and is_prepared(row.get('input'))):
        text = row.get('input')
        if not is_prepared(text):
            raise ValueError('Prepared input needs numbered Document headers and the question marker; use a raw-record adapter otherwise')
        return text
    kind = data_format
    if kind == 'auto':
        if 'documents' in row:
            kind = 'documents'
        elif 'paragraphs' in row:
            kind = 'musique'
        elif two_wiki_context(row.get('context')):
            kind = '2wiki'
        elif isinstance(row.get('context'), str):
            try:
                decoded = json.loads(row['context'])
            except (json.JSONDecodeError, TypeError):
                decoded = None
            kind = '2wiki' if two_wiki_context(decoded) else 'longbench'
        else:
            raise ValueError('Cannot infer record format; set --data-format explicitly')
    if kind == 'documents':
        documents = row['documents']
    elif kind == 'musique':
        documents = ['\n'.join(filter(None, (p.get('title', ''), p['paragraph_text']))) for p in row['paragraphs']]
    elif kind == '2wiki':
        context = row['context']
        if isinstance(context, str):
            context = json.loads(context)
        if not two_wiki_context(context):
            raise ValueError('2Wiki context must contain [title, sentences] pairs')
        documents = [str(title) + '\n' + (' '.join(sentences) if isinstance(sentences, list) else str(sentences)) for title, sentences in context]
    elif kind == 'longbench':
        documents = [row['context']]
    else:
        raise ValueError('Unsupported raw format; RULER should use a prepared numbered-document prompt')
    if not isinstance(documents, list) or not documents or not all(isinstance(d, str) and d.strip() for d in documents):
        raise ValueError('documents must contain nonempty strings')
    question = row.get('question', row.get('query', row.get('input')))
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Missing question/query field')
    documents = [chunk for document in documents for chunk in split_document(document, tokenizer, chunk_tokens)]
    return (instruction_prefix + '\n' + '\n'.join(f'Document {i+1}:\n{doc}' for i, doc in enumerate(documents)) +
            QUESTION_MARKER + '\nQuestion: ' + question + '\nAnswer:')
