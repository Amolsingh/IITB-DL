import json,time,re,unicodedata,hashlib,traceback
from pathlib import Path
import nbformat
import argparse
parser=argparse.ArgumentParser(description='Evaluate retrieval and grounded generation on verified Wikipedia questions.')
parser.add_argument('--notebook',type=Path,required=True)
parser.add_argument('--questions',type=Path,default=Path(__file__).with_name('rag_test_questions_500.json'))
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--limit',type=int,default=500)
args=parser.parse_args()
if not 1<=args.limit<=500:parser.error('--limit must be 1–500')
OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
NOTEBOOK=args.notebook.resolve()
manifest=dict(notebook=str(NOTEBOOK),notebook_sha256=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),questions_sha256=hashlib.sha256(args.questions.read_bytes()).hexdigest(),limit=args.limit,started=time.strftime('%Y-%m-%dT%H:%M:%S'),application_cache=False)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
n=nbformat.read(NOTEBOOK,as_version=4);nbformat.validate(n);ns={}
# Explicit setup cells avoid executing optional evaluation/ingestion cells.
for i in [1,5,6,7,9,10,11,12,15,17]:
 c=n.cells[i];exec(compile(c.source,f'notebook cell {i}','exec'),ns)
def norm(s):return ' '.join(re.findall(r'\w+',''.join(c for c in unicodedata.normalize('NFKD',s.casefold()) if not unicodedata.combining(c))))
def contains(s,aliases):return any(' '+norm(a)+' ' in ' '+norm(s)+' ' for a in aliases)
def hits(rows,q):
 return {'article':any(h['article_id']==q['article_id'] for h in rows),'passage':any(h['passage_id']==q['passage_id'] for h in rows)}
bank=json.load(open(args.questions))[:args.limit];rows=[]
tracepath=OUT/'rag_500_generation_traces.jsonl';tracefile=tracepath.open('w')
(OUT/'candidates').mkdir(exist_ok=True)
started=time.perf_counter()
try:
 for q in bank:
  t=time.perf_counter();row=dict(id=q['id'],question=q['question'],group=q['test_group'],shard=q['shard'],gold_kind=q['gold_kind'])
  try:
   pool=ns['retriever'].search(q['question'],30,use_cache=False,mode='hybrid',candidate_pool=True)
   timings=dict(ns['retriever'].last_timings);warnings=list(ns['retriever'].last_warnings)
   for name,candidates in ns['retriever'].last_branch_candidates.items():
    for k in [5,30,50]:row[f'{name}_hit_at_{k}']=hits(candidates[:k],q)
   expanded=ns['expand_article_leads'](ns['retriever'],pool)
   selected=ns['reranker'].rank(q['question'],expanded,3,use_cache=False)
   row['selected_hit_at_3']=hits(selected,q);row['candidate_pool_hit']=hits(expanded,q)
   r=ns['answer_question'](ns['generator'],q['question'],selected,'concatenate')
   for h in selected:
    offset=h['char_start']-h['original_char_start']
    assert h['original_text'][offset:offset+len(h['text'])]==h['text'],'Non-source excerpt'
   assert all(s['input_tokens']<=512 for s in r['trace'])
   visible=' '.join(p['text'] for p in r['visible_evidence'])
   row.update(answer=r['answer'],expected=q['answers'],alias_match=contains(r['answer'],q['answers']) if q['answers'] else None,alias_in_context=contains(visible,q['answers']) if q['answers'] else None,final_source_hit={'article':any(p['passage_id'] in {h['passage_id'] for h in selected if h['article_id']==q['article_id']} for p in r['visible_evidence']),'passage':any(p['passage_id']==q['passage_id'] for p in r['visible_evidence'])},abstained=r['abstained'],word_count=len(r['answer'].split()),packing=r['context_packing'],timings=dict(timings,rerank_ms=ns['reranker'].last_ms,generation_ms=r['latency_seconds']*1000),warnings=warnings,error=None)
   tracefile.write(json.dumps(dict(id=q['id'],question=q['question'],selected=selected,trace=r['trace']),ensure_ascii=False)+'\n');tracefile.flush()
   (OUT/'candidates'/f"{q['id']}.json").write_text(json.dumps(dict(pool=expanded,selected=selected)))
  except Exception as e:
   row.update(error=repr(e));traceback.print_exc()
  row['end_to_end_seconds']=time.perf_counter()-t;rows.append(row)
  print(q['id'],row.get('answer',row.get('error')),row.get('selected_hit_at_3'),round(row['end_to_end_seconds'],2),flush=True)
  if len(rows)%10==0:
   (OUT/'rag_500_results.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
   (OUT/'status.json').write_text(json.dumps(dict(completed=len(rows),total=len(bank),elapsed_seconds=time.perf_counter()-started,last_id=q['id'])))
 (OUT/'rag_500_results.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
 print('COMPLETE',len(rows),flush=True)
finally:tracefile.close();ns['retriever'].close()
