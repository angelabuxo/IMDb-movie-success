"""Fetch current English Wikipedia wikitext for movies with 100k+ IMDb votes.
Resumable: titles already in the output are skipped. Run from build/imdb."""
import csv,json,os,sys,time,urllib.error,urllib.parse as U,urllib.request as R
UA='DTU-02805-course-project (https://github.com/horrrt/02805_social_graphs)'
OUT='wiki/articles.jsonl'
mov={r['tconst']:r for r in csv.DictReader(open('wiki/movies_100k.csv'))}
link={}
for row in csv.reader(open('wiki/wd_tt_enwiki.tsv'),delimiter='\t',quoting=csv.QUOTE_NONE):
    if len(row)<3: continue
    t=row[0].strip('"')
    if t in mov:
        link.setdefault(t,[]).append(U.unquote(row[2].strip('<>').split('/wiki/',1)[1]).replace('_',' '))
json.dump({k:v for k,v in link.items()},open('wiki/links.json','w'),ensure_ascii=False)
print('movies',len(mov),'linked',len(link),'multi-link',sum(len(v)>1 for v in link.values()),'unlinked',sorted(set(mov)-set(link)),flush=True)
done=set()
if os.path.exists(OUT):
    done={json.loads(l)['title_requested'] for l in open(OUT)}
todo=sorted({a for v in link.values() for a in v}-done)
print('to fetch',len(todo),flush=True)
def get(params):
    u='https://en.wikipedia.org/w/api.php?'+U.urlencode(params)
    for a in range(8):
        try:
            d=json.load(R.urlopen(R.Request(u,headers={'User-Agent':UA}),timeout=90))
            if d.get('error',{}).get('code')=='maxlag': time.sleep(10); continue
            return d
        except urllib.error.HTTPError as e:
            if e.code not in (429,500,502,503,504): raise
            w=int(e.headers.get('Retry-After') or 0) or 30*(a+1); print('HTTP',e.code,'wait',w,flush=True); time.sleep(w)
    raise SystemExit('gave up')
back={a:[t for t,v in link.items() if a in v] for a in todo}
with open(OUT,'a') as f:
    for i in range(0,len(todo),25):
        b=todo[i:i+25]; base={'action':'query','format':'json','formatversion':2,'redirects':1,'maxlag':5,'prop':'revisions','rvprop':'ids|timestamp|content','rvslots':'main','titles':'|'.join(b)}
        got={}; extra={}; m={}
        while True:
            d=get({**base,**extra})
            for k in ('normalized','redirects'):
                for x in d['query'].get(k,[]): m[x['from']]=x['to']
            for p in d['query']['pages']:
                if p.get('revisions'): got[p['title']]=p
            if 'continue' not in d: break
            extra=d['continue']; time.sleep(2)
        for a in b:
            r=a;seen=set()
            while r in m and r not in seen: seen.add(r); r=m[r]
            p=got.get(r); rev=(p or {}).get('revisions',[{}])[0]
            f.write(json.dumps({'title_requested':a,'title':r,'tconst':back[a],'pageid':(p or {}).get('pageid'),'revid':rev.get('revid'),'timestamp':rev.get('timestamp'),'wikitext':rev.get('slots',{}).get('main',{}).get('content')},ensure_ascii=False)+'\n')
        f.flush(); print('done',min(i+25,len(todo)),'/',len(todo),flush=True); time.sleep(3)
print('finished',flush=True)
