from pathlib import Path
import json,random,re,hashlib
import pyarrow.parquet as pq
import psycopg
from psycopg.rows import dict_row
import argparse
parser=argparse.ArgumentParser(description='Build 500 source-verified Wikipedia RAG diagnostics.')
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--base-questions',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--dsn',default='host=localhost dbname=rag_vectors')
args=parser.parse_args();root=args.root;args.output.mkdir(parents=True,exist_ok=True)
bank=json.load(open(args.base_questions))
for q in bank:q['test_group']='curated_factual';q['split']='development';q['gold_kind']='answer_alias'
seen_questions={q['question'] for q in bank};seen={q['article_id'] for q in bank};rng=random.Random(20261027);samples=[]
with psycopg.connect(args.dsn,autocommit=True,row_factory=dict_row) as db:
 for si,p in enumerate(sorted((root/'data/wikipedia_snapshot/20231101.en').glob('*.parquet'))[:8]):
  pf=pq.ParquetFile(p);target=44 if si<6 else 43;count=0
  # Reproducible uniform row candidates, then eligibility filtering. Not a uniform topic sample.
  indices=sorted(rng.sample(range(pf.metadata.num_rows),min(800,pf.metadata.num_rows)))
  offset=0
  for ri in range(pf.num_row_groups):
   size=pf.metadata.row_group(ri).num_rows;take=[x-offset for x in indices if offset<=x<offset+size];offset+=size
   if not take:continue
   table=pf.read_row_group(ri,columns=['id','title','url','text'])
   for j in take:
    a=table.slice(j,1).to_pylist()[0]
    if a['id'] in seen or a['title'].startswith(('List of','Timeline of')):continue
    row=db.execute('SELECT id,chunk_id,article_id,title,url,body,char_start,char_end FROM rag_wiki_bge_full_v1.chunks WHERE article_id=%s ORDER BY char_start LIMIT 1',(a['id'],)).fetchone()
    if not row or row['char_start']!=0 or not 150<=len(row['body'])<=2200:continue
    if row['body']!=a['text'][row['char_start']:row['char_end']]:raise AssertionError('Source mismatch')
    # Alternate title lookup and masked descriptive retrieval tests.
    kind='named_article' if len(samples)%2==0 else 'description_lookup'
    if kind=='named_article':
     question=f'What is {a["title"]}? Give a short description.';aliases=[];gold='source_only'
    else:
     # Use a verbatim definition tail after is/was/are, never include the title.
     m=re.search(r'\b(?:is|was|are|were)\s+(.+?[.!?])(?:\s|$)',row['body'])
     if not m:continue
     desc=m.group(1).strip()
     if not 8<=len(desc.split())<=65 or a['title'].casefold() in desc.casefold():continue
     question=f'Which Wikipedia subject fits this description: "{desc}"?';aliases=[a['title']];gold='title_alias'
    if question in seen_questions:continue
    seen_questions.add(question)
    samples.append(dict(id=f'Q{151+len(samples):03d}',test_group=kind,split='held_out_sample',gold_kind=gold,question=question,answers=aliases,article_id=a['id'],title=a['title'],url=a['url'],shard=p.name,passage_id=row['chunk_id'],pg_id=row['id'],evidence_quote=row['body'],evidence_sha256=hashlib.sha256(row['body'].encode()).hexdigest(),answerable=True,domain='unclassified sampled article'))
    seen.add(a['id']);count+=1
    if count==target:break
   if count==target:break
  assert count==target,(p.name,count,target)
  print(p.name,count,flush=True)
bank+=samples
assert len(bank)==500 and len({q['question'] for q in bank})==500
(args.output/'rag_test_questions_500.json').write_text(json.dumps(bank,indent=2,ensure_ascii=False))
(args.output/'rag_test_questions_500.md').write_text('# 500 RAG test questions\n\n150 curated factual questions plus 350 templated retrieval diagnostics; the latter are not independently authored factual QA. All expected source passages verified against PostgreSQL and the local shard text.\n\n'+'\n'.join(f"{i+1}. **{q['test_group']} · {q['title']} · {q['shard']}** — {q['question']}" for i,q in enumerate(bank)))
print('VERIFIED 500',flush=True)
