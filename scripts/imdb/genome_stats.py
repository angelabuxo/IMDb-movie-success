"""Count Tag Genome reviews per movie by IMDb ID without printing review text. Run from build/imdb."""
import collections as C,csv,io,json,re,statistics as S,zipfile
z=zipfile.ZipFile('reviews/genome/genome_2021.zip')
names=z.namelist(); pick=lambda k:next(n for n in names if n.endswith(k))
m={}
with z.open(pick('metadata.json')) as f:
    for i,l in enumerate(io.TextIOWrapper(f,encoding='utf8')):
        r=json.loads(l)
        if i==0: print('metadata keys',sorted(r))
        if r.get('imdbId') not in (None,''): m[r['item_id']]='tt%07d'%int(r['imdbId'])
print('metadata items with imdb',len(m))
cnt=C.Counter();chars=C.Counter();total=0;empty=0
with z.open(pick('reviews.json')) as f:
    for i,l in enumerate(io.TextIOWrapper(f,encoding='utf8')):
        r=json.loads(l)
        if i==0: print('review keys',sorted(r))
        total+=1; t=r.get('txt') or ''
        if not t.strip(): empty+=1
        if r['item_id'] in m: cnt[m[r['item_id']]]+=1; chars[m[r['item_id']]]+=len(t)
print('reviews',total,'empty',empty,'films with reviews',len(cnt),'median per film',S.median(cnt.values()),'max',max(cnt.values()))
w=csv.writer(open('features/genome_review_counts.csv','w',newline=''));w.writerow(['tconst','reviews','chars']);[w.writerow([k,v,chars[k]]) for k,v in cnt.items()]
rows=list(csv.DictReader(open('features/movies_features_10k.csv')))
ok=lambda r:r['budget_cur']=='USD' and r['gross_cur']=='USD' and float(r['b'] or 0)>=1000 and float(r['g'] or 0)>=1000 and 'rentals' not in r['gross_flag'] and 'regional' not in r['gross_flag'] and not re.search('netflix|amazon|apple|hulu',r['distributors'] or '',re.I) and int(r['y'] or 0)<2026
cl=[r for r in rows if ok(r)]
for name,sub in (('100k+',[r for r in cl if r['tier']=='100k+']),('10k-100k',[r for r in cl if r['tier']!='100k+']),('all clean',cl)):
    n=[cnt[r['tconst']] for r in sub if r['tconst'] in cnt]
    print(name,'clean',len(sub),'with reviews',len(n),'reviews',sum(n),'median',S.median(n) if n else 0,'>=10',sum(x>=10 for x in n),'>=50',sum(x>=50 for x in n))
D=C.defaultdict(lambda:[0,0,0])
for r in cl:
    d=int(r['y'])//10*10; D[d][0]+=1; D[d][1]+=r['tconst'] in cnt; D[d][2]+=cnt[r['tconst']]>=10
for d in sorted(D):
    if d>=1970: print(d,'clean',D[d][0],'any',D[d][1],'>=10',D[d][2],'pct',round(100*D[d][1]/D[d][0]))
Y=C.Counter(int(r['y']) for r in cl if r['tconst'] in cnt); print('by late year',{y:Y[y] for y in range(2016,2026)})
below=[r for r in cl if r['tconst'] in cnt and float(r['g'])<float(r['b'])]; print('with reviews and under budget',len(below))
