"""Extract budget, gross, studios, stars and critic scores from wikitext. Run from build/imdb."""
import csv,html,json,re,sys
SRC=sys.argv[1] if len(sys.argv)>1 else 'wiki/articles.jsonl'; DST=sys.argv[2] if len(sys.argv)>2 else 'features/wiki_infobox.csv'
MON={m:i+1 for i,m in enumerate('january february march april may june july august september october november december'.split())}
import mwparserfromhell as M
SYM={'US$':'USD','$':'USD','USD':'USD','£':'GBP','GBP':'GBP','€':'EUR','EUR':'EUR','₹':'INR','INR':'INR','¥':'JPY','JP¥':'JPY','CN¥':'CNY','RMB':'CNY','A$':'AUD','AU$':'AUD','C$':'CAD','CA$':'CAD','HK$':'HKD','NZ$':'NZD','₩':'KRW','R$':'BRL','USD$':'USD','NT$':'TWD','S$':'SGD','MX$':'MXN','CAD$':'CAD','AUD$':'AUD','GBP$':'GBP','HUF$':'HUF'}
MUL={'thousand':1e3,'million':1e6,'m':1e6,'billion':1e9,'bn':1e9,'crore':1e7,'lakh':1e5}
# A code written straight before '$' (HUF$, NT$, S$) is that currency, never US dollars.
CUR=r'(US\$|A\$|AU\$|C\$|CA\$|HK\$|NZ\$|JP¥|CN¥|R\$|[A-Z]{1,4}\$|RMB|USD|GBP|EUR|INR|\$|£|€|₹|¥|₩)'
NUM=r'(\d[\d,]*(?:\.\d+)?)'
PAT=re.compile(CUR+r'\s*'+NUM+r'(?:\s*(?:[-–—−‒]|to)\s*'+CUR+r'?\s*'+NUM+r')?\s*(thousand|million|billion|crore|lakh|bn|m)?s?\b',re.I)
def clean(raw):
    s=re.sub(r'(?is)<ref[^>]*/>|<ref.*?</ref>|<!--.*?-->','',raw)
    s=re.sub(r'(?i)\{\{\s*(?:nbsp?|spaces|snd|thinsp)[^}]*\}\}',' ',s)
    s=re.sub(r'(?i)\{\{\s*(?:ndash|endash|en dash|mdash)\s*\}\}','–',s)
    s=re.sub(r'\[\[(?:[^\]|]*\|)?([^\]]*)\]\]',r'\1',s)
    s=re.sub(r'(?i)\{\{\s*(?:estimation|est\.?|circa|c\.|efn|sfn|refn|cn|citation needed|better source)[^}]*\}\}','',s)
    for tpl,sym in (('US\\$|USD','$'),('GBP|£','£'),('EUR|€','€'),('INR|₹','₹'),('JPY|¥','¥'),('CAD','C$'),('AUD','A$')):
        s=re.sub(r'(?i)\{\{\s*(?:'+tpl+r')\s*\|\s*([^|}]+)[^}]*\}\}',lambda m:sym+m[1],s)
        s=re.sub(r'(?i)\{\{\s*(?:'+tpl+r')\s*\}\}\s*',sym,s)
    s=re.sub(r'\{\{\s*INRConvert\s*\|\s*([^|}]+)\|?\s*(c|l|b|m)?[^}]*\}\}',lambda m:'₹'+m[1]+{'c':' crore','l':' lakh','b':' billion','m':' million'}.get(m[2] or '',''),s,flags=re.I)
    return s.replace('&nbsp;',' ').replace(' ',' ').replace('&ndash;','–')
def money(raw):
    if not raw: return (None,None,'')
    s=clean(raw); ms=list(PAT.finditer(s))
    if not ms: return (None,None,s.strip()[:80])
    m=next((x for x in ms if SYM.get(x[1],SYM.get(x[1].upper()))=='USD'),ms[0])
    cur=SYM.get(m[1],SYM.get(m[1].upper())) or m[1].rstrip('$').upper(); lo=float(m[2].replace(',','')); hi=float(m[4].replace(',','')) if m[4] else None
    k=MUL.get((m[5] or '').lower(),1)
    if hi:
        if lo>=1000 and k>1: v=(lo+hi*k)/2
        else: v=(lo+hi)/2*k
    else: v=lo*k
    return (v,cur,s.strip()[:80])
def scope(s):
    t=s.lower(); f=[]
    if re.search(r'rental',t): f.append('rentals')
    if re.search(r'\bu\.?s\.?\b|domestic|north america|canada|\buk\b|italy|japan',t) and not re.search(r'worldwide|global',t): f.append('regional')
    if re.search(r'\(net\)|\(gross\)',t): f.append('net_gross')
    return ';'.join(f)
DTV=re.compile(r'(?i)\[\[\s*Category\s*:[^\]]*direct-to-video')
NOT_RENTALS=re.compile(r'(?i)worldwide rentals beat domestic take|\bname\s*=\s*"?[^">|}]*"?|https?://[^\s|\]}]+')  # a Variety 1995 headline on 1994 worldwide grosses; ref names; URLs
def gross_flags(raw,shown,wikitext='',title=''):
    """scope() of the ref-stripped gross, plus 'rentals' when the raw value, refs included, says rentals
    (The Haunting's ref reads 'figures are rentals as opposed to total gross'), plus 'direct_to_video' when
    the article is in a direct-to-video category. Whole words only, so 'parental' doesn't count, and the film's
    own title is ignored, so The Rental (2020) isn't flagged by its Box Office Mojo citation."""
    f=[x for x in scope(shown).split(';') if x]
    s=NOT_RENTALS.sub('',raw or '')
    name=re.sub(r'\s*\([^()]*\)\s*$','',title or '').strip()
    if name: s=re.sub(re.escape(name),'',s,flags=re.I)
    if 'rentals' not in f and re.search(r'(?i)\brentals?\b',s): f.insert(0,'rentals')
    if DTV.search(wikitext or ''): f.append('direct_to_video')
    return ';'.join(f)
LIST_TPL=re.compile(r'(?i)\s*(ubl\w*|ubi\w*|blist|idp|pl|[\w ]*list)\s*$')  # ubl, ubli, ubil, idp, pl, plainlist, indented plainlist, unbulleted list, hlist...
def _items(text):
    # One name per line: a line's link titles if it has links, else its plain text. Bare '(...)' notes are dropped.
    text=re.sub(r'\([^()]*\)',lambda m:m[0].replace(';',','),text)  # a ';' inside brackets is not a separator
    for line in re.split(r'\n|;|\x1f',text):
        c=M.parse(line)
        links=[re.sub(r'^:?[a-z]{2,3}(?:-[a-z]+)?:','',str(l.title).split('#')[0].strip()) for l in c.filter_wikilinks()]  # drop ':de:' prefixes
        if links: yield from links; continue
        x=c.strip_code().strip(' *,\t')
        if x and not (x[0]=='(' and x[-1]==')') and not re.search(r'\[\[|\]\]|\{\{|\}\}',x): yield x
def names(raw):
    if not raw: return []
    s=re.sub(r'(?is)<ref[^>]*/>|<ref.*?</ref>|<!--.*?-->','',raw)
    s=html.unescape(s)  # before any split, so the ';' of '&nbsp;' isn't a separator
    s=re.sub(r'(?i)\{\{\s*(?:nowrap|nobr|nobreak|small|ill|interlanguage link)\s*\|([^{}|]*)[^{}]*\}\}',r'\1',s)  # strip_code() would drop these names
    s=re.sub(r'(?i)\{\{\s*break\s*\}\}','<br>',s)
    s=re.sub(r'\[\[[^\]|]*\|([^\]]*<br[^\]]*)\]\]',r'\1',s,flags=re.I)  # [[Duo|A<br>B]]: keep both names as text
    s=re.sub(r'(?i)<br\s*/?\s*>','\x1f',s)  # split on <br> ourselves: strip_code() deletes the tag and merges names
    c=M.parse(s); out=[]
    for t in c.filter_templates(recursive=False):
        if LIST_TPL.match(str(t.name)):
            for x in t.params:
                if str(x.name).strip().isdigit(): out+=list(_items(str(x.value)))
            c.remove(t)
        else:
            c.remove(t)  # any other template (a nested infobox, a citation) holds no names
    out+=list(_items(str(c)))
    return [x for x in dict.fromkeys(out) if x]
IB=re.compile(r'(?i)^infobox[ _](?:film|movie|television|short film)')
w=csv.writer(open(DST,'w',newline=''))
w.writerow('tconst title has_infobox budget budget_cur budget_raw budget_flag gross gross_cur gross_raw gross_flag studios distributors producers starring rt_pct metacritic critic_chars release_month'.split())
n=0
for l in open(SRC):
    r=json.loads(l); t=r['wikitext'] or ''; p=M.parse(t[:60000],skip_style_tags=True); box=None
    for tp in p.filter_templates(recursive=False):
        if IB.match(str(tp.name).strip()): box=tp;break
    g=lambda *k:next((str(box.get(x).value) for x in k if box is not None and box.has(x) and str(box.get(x).value).strip()),'')
    graw=g('gross','box office','box_office'); bv,bc,br=money(g('budget')); gv,gc,gr=money(graw)
    rt=None;mc=None
    m=re.search(r'\{\{\s*Rotten Tomatoes prose\s*\|\s*(\d{1,3})',t,re.I); rt=m and m[1]
    if not rt:
        m=re.search(r'Rotten Tomatoes[^\n]{0,250}?(\d{1,3})\s*(?:%|percent|\{\{)',t); rt=m and m[1]
    m=re.search(r'\{\{\s*Metacritic(?: film)? prose\s*\|\s*(\d{1,3})',t,re.I); mc=m and m[1]
    if not mc:
        m=re.search(r'Metacritic[^\n]{0,250}?(\d{1,3})\s*(?:out of|/)\s*100',t); mc=m and m[1]
    sec=re.search(r'(?im)^===?\s*(critical (?:response|reception)|reviews?|critics?)[^=\n]*===?\s*$',t); cc=0
    if sec:
        a=t[sec.end():];e=re.search(r'(?m)^==[^=]',a);cc=len(a[:e.start()] if e else a)
    rel=g('released','release_date','release date'); rm=None
    d=re.search(r'(?i)\{\{\s*(?:film date|start date|release date|filmdate)[^|}]*\|\s*(?:df=\w+\s*\|\s*)?\d{4}\s*\|\s*(\d{1,2})',rel)
    if d: rm=int(d[1])
    else:
        d=re.search(r'(?i)\b('+'|'.join(MON)+r')\b',rel); rm=d and MON[d[1].lower()]
    for tc in r['tconst']:
        w.writerow([tc,r['title'],int(box is not None),bv,bc,br,scope(br),gv,gc,gr,gross_flags(graw,gr,t,r['title']),'|'.join(names(g('studio','studios','production_companies','production companies'))),'|'.join(names(g('distributor','distributors'))),'|'.join(names(g('producer','producers'))),'|'.join(names(g('starring'))),rt,mc,cc,rm])
    n+=1
print('parsed',n)
