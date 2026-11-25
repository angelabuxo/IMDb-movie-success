import json,re,statistics as s
n=ok=plot=0;L=[];PL=[];LK=[];T=set();ex=None
rows=[json.loads(l) for l in open('wiki/articles.jsonl')]
titles={r['title'] for r in rows}
inner=0
for r in rows:
    n+=1;w=r['wikitext']
    if not w: continue
    ok+=1;L.append(len(w))
    links={x.split('|')[0].split('#')[0].strip() for x in re.findall(r'\[\[([^\]]+)\]\]',w)}
    links={x[0].upper()+x[1:] for x in links if x}
    LK.append(len(links));inner+=len(links&titles-{r['title']})
    m=re.search(r'(?im)^==\s*(plot|synopsis|plot summary|premise)[^=]*==\s*$',w)
    if m:
        plot+=1;a=w[m.end():];e=re.search(r'(?m)^==[^=]',a);PL.append(len(a[:e.start()] if e else a))
    if 'tt1375666' in r['tconst']: ex=(r['title'],r['pageid'],r['revid'],r['timestamp'],len(w),len(links))
print({'rows':n,'with_text':ok,'median_chars':s.median(L),'plot_section':plot,'median_plot_chars':s.median(PL),'median_links':s.median(LK),'links_to_other_movies_in_set':inner});print('Inception',ex)
