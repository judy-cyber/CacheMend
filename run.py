#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
import time


def parser(fixed_stage=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    if fixed_stage is None:
        p.add_argument('stage',choices=['cache','evaluate','all'])
    else:
        p.set_defaults(stage=fixed_stage)
    for name in ['model-path','data-path','kv-cache-dir']:
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--output-dir',type=Path,help='Required for evaluate/all; optional for cache')
    p.add_argument('--dataset-name',default='custom',help='Dataset label, e.g. custom, ruler, musique, 2wiki, longbench; arbitrary labels accepted')
    p.add_argument('--data-format',choices=['auto','prepared','documents','musique','2wiki','longbench'],default='auto')
    p.add_argument('--chunk-tokens',type=int,default=0,help='Split raw documents into approximately this many tokenizer tokens; 0 preserves boundaries. Prepared inputs are unchanged')
    p.add_argument('--instruction-prefix',default='Answer the question using the documents.')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--torch-dtype',choices=['float32','float16','bfloat16'],default='bfloat16')
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--count',type=int,default=0,help='0 = all rows after start')
    p.add_argument('--budget',type=float,default=.1)
    p.add_argument('--check-layer',type=int,default=1,help='Zero-based layer index')
    p.add_argument('--window-size',type=int,default=64)
    p.add_argument('--density-threshold',type=int,default=8)
    p.add_argument('--head-aggregation',choices=['max','sum','zsum','ranksum'],default='max')
    p.add_argument('--value-weight',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--output-weight',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--mask-prefix',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--restore-rope',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--cache-weight-check',choices=['stat','sha256'],default='stat',
                   help='Check weight size/mtime, or read all weights for strict SHA256 checking')
    p.add_argument('--allow-legacy-kv',action='store_true',
                   help='Allow old caches without model fingerprints after structural checks')
    p.add_argument('--local-batch-size',type=int,default=4)
    p.add_argument('--query-chunk-size',type=int,default=128)
    p.add_argument('--max-new-tokens',type=int,default=32)
    p.add_argument('--num-samples','--sample-count',dest='num_samples',type=int,default=1)
    p.add_argument('--sample-batch-size',type=int,default=1)
    p.add_argument('--suffix-chunk-size',type=int,default=64)
    p.add_argument('--do-sample',action='store_true')
    p.add_argument('--temperature',type=float,default=.7)
    p.add_argument('--top-p',type=float,default=.9)
    p.add_argument('--seed','--sampling-seed',dest='seed',type=int,default=0)
    p.add_argument('--dry-run',action='store_true',help='Validate and print arguments without importing torch')
    return p


def sync(device):
    import torch
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)


def run(args):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
    import torch
    from cachemend.cache import load_model
    from cachemend.data import _split_prompt_tokens
    from cachemend.datasets import load_rows, prepare_prompt, select_rows
    from cachemend.decoding import decode
    from cachemend.kv_builder import cache_signature, load_or_build_kv, model_fingerprint
    from cachemend.method import repair
    rows=load_rows(args.data_path)
    selected_rows=select_rows(rows,args.start,args.count)
    model,tokenizer=load_model(args.model_path,device_map=args.device,
                             torch_dtype=getattr(torch,args.torch_dtype),attn_implementation='sdpa')
    if not hasattr(model,'model') or not hasattr(model.model,'layers'):
        raise ValueError('Checkpoint does not expose the supported decoder layout')
    fingerprint=model_fingerprint(args.model_path,weight_check=args.cache_weight_check,
                                  model=model,tokenizer=tokenizer)
    cache_dir=args.kv_cache_dir if args.dataset_name == 'custom' else args.kv_cache_dir/args.dataset_name
    cache_dir.mkdir(parents=True,exist_ok=True)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    config={k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (args.output_dir/'arguments.json').write_text(json.dumps(config,indent=2))
    results=args.output_dir/'predictions.jsonl'
    if args.stage in ('evaluate','all') and results.exists():
        raise FileExistsError(f'{results} already exists; choose a new output directory')
    for position,index,row in selected_rows:
        text=prepare_prompt(row,tokenizer,dataset_name=args.dataset_name,
                            data_format=args.data_format,chunk_tokens=args.chunk_tokens,
                            instruction_prefix=args.instruction_prefix)
        split=_split_prompt_tokens(tokenizer,text)
        signature=cache_signature(split['document_ids'].tolist(),args.model_path,args.torch_dtype,
                args.restore_rope,fingerprint=fingerprint,
                prefix_length=int(split['prefix_ids'].numel()),document_ranges=split['document_ranges'])
        stored=load_or_build_kv(cache_dir/f'{index:06d}.pt',signature=signature,
                model=model,split=split,build=args.stage in ('cache','all'),
                local_batch_size=args.local_batch_size,restore_rope=args.restore_rope,
                dataset_name=args.dataset_name,row_index=index,fingerprint=fingerprint,
                allow_legacy=args.allow_legacy_kv,legacy_cache_dir=args.kv_cache_dir)
        if args.stage == 'cache': continue
        spliced=[(k.to(args.device),v.to(args.device)) for k,v in stored['spliced']]
        # Exclude cache loading/device transfer; include all online scoring,
        # selection, repair, suffix prefill and the first greedy token decision.
        sync(args.device);started=time.perf_counter()
        with torch.inference_mode():
            repaired,selected=repair(model,split['document_ids'],split['suffix_ids'][None],spliced,
                budget=args.budget,check_layer=args.check_layer,window_size=args.window_size,
                density_threshold=args.density_threshold,
                prefix_length=int(split['prefix_ids'].numel()) if args.mask_prefix else 0,
                head_aggregation=args.head_aggregation,value_weight=args.value_weight,
                output_weight=args.output_weight,query_chunk_size=args.query_chunk_size)
        sync(args.device);repair_seconds=time.perf_counter()-started
        sampling_seed=args.seed+index*100
        decoded=decode(model,tokenizer,past=repaired,
            suffix_ids=split['suffix_ids'][None].to(args.device),
            max_new_tokens=args.max_new_tokens,do_sample=args.do_sample,
            sample_count=args.num_samples,sample_batch_size=args.sample_batch_size,
            temperature=args.temperature,top_p=args.top_p,seed=sampling_seed,
            suffix_chunk_size=args.suffix_chunk_size,timing_origin=started)
        answers=decoded['answers']
        record={'dataset_name':args.dataset_name,'index':index,'row_position':position,
                'sampling_seed':sampling_seed,**decoded,'selected_positions':selected,
                'cache_verification':stored['cache_verification'],
                'document_tokens':int(split['document_ids'].numel()),'repair_seconds':repair_seconds}
        with results.open('a') as handle: handle.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(json.dumps({'index':index,'selected':len(selected),'answers':answers},ensure_ascii=False),flush=True)


def main(argv=None, fixed_stage=None):
    p=parser(fixed_stage);args=p.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",args.dataset_name): p.error("dataset-name must be a simple label, without slashes")
    if args.chunk_tokens < 0: p.error("chunk-tokens must be nonnegative")
    if args.output_dir is None:
        if args.stage != "cache": p.error("--output-dir is required for evaluate/all")
        args.output_dir=args.kv_cache_dir/args.dataset_name/"build_metadata"
    if not 0 < args.budget <= 1: p.error('budget must be in (0,1]')
    if args.start < 0 or args.count < 0 or args.check_layer < 0: p.error('start/count/check-layer must be nonnegative')
    if min(args.window_size,args.density_threshold,args.local_batch_size,args.query_chunk_size,args.suffix_chunk_size,args.sample_batch_size,args.max_new_tokens,args.num_samples) < 1: p.error('Sizes must be positive')
    if args.seed < 0: p.error('seed must be nonnegative')
    if args.density_threshold > args.window_size: p.error('density threshold exceeds window size')
    if args.temperature <= 0 or not 0 < args.top_p <= 1: p.error('Invalid temperature/top-p')
    if args.dry_run:
        print(json.dumps(vars(args),default=str,indent=2));return
    for path in [args.model_path,args.data_path]:
        if not path.exists(): p.error(f'Missing input: {path}')
    run(args)

if __name__=='__main__': main()
