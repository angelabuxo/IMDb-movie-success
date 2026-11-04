"""Count MovieSum rows, keys and script structure without printing script text. Run from build/imdb."""
import csv,json,re,statistics as s,collections as C
rows=[]
for f in ('train','val','test'):
    for l in open(f'scripts/moviesum/{f}.jsonl'):
        r=json.loads(l); r['_split']=f; rows.append(r)
print('rows',len(rows),'keys',sorted(rows[0]),{f:sum(r['_split']==f for r in rows) for f in ('train','val','test')})
K=next(k for k in rows[0] if k.lower().replace('_','') in('imdbid','imdb'))
S=next(k for k in rows[0] if k.lower() in('script',))
ids=[r[K] for r in rows]; print('distinct ids',len(set(ids)),'bad ids',sum(not re.fullmatch(r'tt\d+',str(i)) for i in ids),'dups',[i for i,c in C.Counter(ids).items() if c>1][:15])
out=csv.writer(open('features/moviesum_stats.csv','w',newline=''));out.writerow('tconst split chars scenes dialogue_lines distinct_characters dialogue_chars'.split())
A=[];SC=[];D=[];CH=[];bad=0
seen=set()
for r in rows:
    t=r[S] or ''; sc=len(re.findall(r'<scene>',t)); d=re.findall(r'(?s)<dialogue>(.*?)</dialogue>',t); ch={x.strip().upper() for x in re.findall(r'(?s)<character>(.*?)</character>',t)}
    if not sc or not d: bad+=1
    out.writerow([r[K],r['_split'],len(t),sc,len(d),len(ch),sum(map(len,d))])
    A.append(len(t));SC.append(sc);D.append(len(d));CH.append(len(ch))
print('median chars',s.median(A),'scenes',s.median(SC),'dialogue lines',s.median(D),'distinct character labels',s.median(CH),'no scenes or dialogue',bad)
print('min/max chars',min(A),max(A))
