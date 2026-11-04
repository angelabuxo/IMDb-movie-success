.mode markdown
SET threads=8;
CREATE TEMP TABLE m AS SELECT tconst,primaryTitle,try_cast(startYear AS INT) y,genres,try_cast(votes AS INT) votes,try_cast(averageRating AS DOUBLE) rating FROM read_csv('wiki/movies_100k.csv',header=true,all_varchar=true);
CREATE TEMP TABLE i AS SELECT * FROM read_csv('features/wiki_infobox.csv',header=true,all_varchar=true) WHERE title<>'Apocalypse Now Redux';
CREATE TEMP TABLE f AS SELECT m.*,i.title,try_cast(budget AS DOUBLE) b,budget_cur,budget_raw,budget_flag,try_cast(gross AS DOUBLE) g,gross_cur,gross_raw,gross_flag,nullif(studios,'') studios,nullif(distributors,'') distributors,nullif(producers,'') producers,nullif(starring,'') starring,try_cast(rt_pct AS INT) rt,try_cast(metacritic AS INT) mc,try_cast(critic_chars AS INT) cc FROM m LEFT JOIN i USING(tconst);
CREATE TEMP TABLE wb AS SELECT replace(c0,'"','') t,try_cast(c1 AS DOUBLE) a FROM read_csv('features/wd_budget.tsv',delim='\t',quote='',header=false,skip=1,columns={'c0':'VARCHAR','c1':'VARCHAR','c2':'VARCHAR'}) WHERE c2 LIKE '%Q4917>';
CREATE TEMP TABLE wg AS SELECT replace(c0,'"','') t,try_cast(c1 AS DOUBLE) a FROM read_csv('features/wd_gross.tsv',delim='\t',quote='',header=false,skip=1,null_padding=true,columns={'c0':'VARCHAR','c1':'VARCHAR','c2':'VARCHAR','c3':'VARCHAR'}) WHERE c2 LIKE '%Q4917>';
.print COVERAGE, 100k+ SET
SELECT count(*) movies,sum((b IS NOT NULL)::int) budget_any,sum((budget_cur='USD')::int) budget_usd,sum((g IS NOT NULL)::int) gross_any,sum((gross_cur='USD')::int) gross_usd,sum((budget_cur='USD' AND gross_cur='USD')::int) both_usd,sum((studios IS NOT NULL)::int) studios,sum((distributors IS NOT NULL)::int) distrib,sum((producers IS NOT NULL)::int) producers,sum((starring IS NOT NULL)::int) starring,sum((rt IS NOT NULL)::int) rt,sum((mc IS NOT NULL)::int) mc,sum((cc>0)::int) critic_section FROM f;
.print WIKIPEDIA VS WIKIDATA (USD)
SELECT count(*) both_sources,sum((abs(b/a-1)<=.05)::int) within_5pct,round(100.0*sum((abs(b/a-1)<=.05)::int)/count(*),1) pct,sum((abs(b/a-1)>.5)::int) off_50pct FROM f JOIN (SELECT t,max(a) a FROM wb GROUP BY 1) ON t=tconst WHERE budget_cur='USD';
SELECT count(*) both_sources,sum((abs(g/a-1)<=.05)::int) within_5pct,round(100.0*sum((abs(g/a-1)<=.05)::int)/count(*),1) pct,sum((abs(g/a-1)>.5)::int) off_50pct FROM f JOIN (SELECT t,max(a) a FROM wg GROUP BY 1) ON t=tconst WHERE gross_cur='USD';
.print WORST DISAGREEMENTS, BUDGET
SELECT primaryTitle,y,b wiki,a wikidata,budget_raw FROM f JOIN (SELECT t,max(a) a FROM wb GROUP BY 1) ON t=tconst WHERE budget_cur='USD' AND abs(b/a-1)>.5 ORDER BY hash(tconst) LIMIT 8;
.print WORST DISAGREEMENTS, GROSS
SELECT primaryTitle,y,g wiki,a wikidata,gross_raw FROM f JOIN (SELECT t,max(a) a FROM wg GROUP BY 1) ON t=tconst WHERE gross_cur='USD' AND abs(g/a-1)>.5 ORDER BY hash(tconst) LIMIT 8;
.print FLAGS
SELECT sum((gross_flag LIKE '%rentals%')::int) rentals,sum((gross_flag LIKE '%regional%')::int) regional,sum((budget_flag LIKE '%net_gross%')::int) budget_net_gross,sum((distributors ILIKE '%netflix%' OR distributors ILIKE '%amazon%' OR distributors ILIKE '%apple%' OR distributors ILIKE '%hulu%')::int) streaming_distrib,sum((budget_cur NOT IN ('USD') OR gross_cur NOT IN ('USD'))::int) non_usd FROM f;
.print ROTTEN TOMATOES, WIKIPEDIA TEXT VS WIKIDATA
CREATE TEMP TABLE wr AS SELECT replace(c0,'"','') t,max(try_cast(regexp_extract(c1,'(\d+)%',1) AS INT)) s FROM read_csv('features/wd_scores.tsv',delim='\t',quote='',header=false,skip=1,columns={'c0':'VARCHAR','c1':'VARCHAR','c2':'VARCHAR','c3':'VARCHAR'}) WHERE c3 LIKE '%Rotten Tomatoes%' AND c1 LIKE '%\%%' ESCAPE '\' GROUP BY 1;
SELECT count(*) both_sources,sum((abs(rt-s)<=5)::int) within_5pts,round(100.0*sum((abs(rt-s)<=5)::int)/count(*),1) pct,sum((abs(rt-s)>20)::int) off_20pts,round(median(rt),0) med_rt FROM f JOIN wr ON t=tconst WHERE rt IS NOT NULL;
SELECT sum((rt IS NOT NULL OR tconst IN (SELECT t FROM wr))::int) rt_either FROM f;
.print CLEAN THEATRICAL SAMPLE
CREATE TEMP TABLE cl AS SELECT * FROM f WHERE budget_cur='USD' AND gross_cur='USD' AND b>=1000 AND coalesce(gross_flag,'') NOT LIKE '%rentals%' AND coalesce(gross_flag,'') NOT LIKE '%regional%' AND NOT coalesce(distributors ILIKE '%netflix%' OR distributors ILIKE '%amazon%' OR distributors ILIKE '%apple%' OR distributors ILIKE '%hulu%',false) AND y<2026;
SELECT count(*) clean,sum((y>=2000)::int) since2000,sum((y>=2010)::int) since2010,sum((g<b)::int) below_budget,sum((g<2.5*b)::int) below_2_5x,median(g/b) med_multiple,median(g-b) med_margin,sum((studios IS NOT NULL)::int) with_studio,sum((rt IS NOT NULL)::int) with_rt,sum((mc IS NOT NULL)::int) with_mc,sum((cc>0)::int) with_critics_text,round(corr(ln(g/b),rating),3) corr_lnmult_imdb,round(corr(ln(g/b),mc),3) corr_lnmult_mc FROM cl;
CREATE TEMP TABLE st AS SELECT DISTINCT tconst,trim(x) s FROM f,unnest(string_split(studios,'|')) u(x) WHERE tconst IN (SELECT tconst FROM cl);
SELECT count(DISTINCT s) distinct_studios,count(*) rows_ FROM st;
SELECT s studio,count(*) movies FROM st GROUP BY 1 ORDER BY 2 DESC LIMIT 8;
SELECT 'clean' g_,gn genre,count(*) movies,sum((y>=2000)::int) since2000,sum((y>=2010)::int) since2010 FROM cl,unnest(string_split(genres,',')) u(gn) GROUP BY 2 ORDER BY 3 DESC LIMIT 10;
.print DECADES (USD budget and gross)
SELECT floor(y/10)::int*10 AS dec_start,count(*) movies,sum((budget_cur='USD' AND gross_cur='USD')::int) complete,round(100.0*sum((budget_cur='USD' AND gross_cur='USD')::int)/count(*),0) pct FROM f GROUP BY 1 ORDER BY 1;
.print GENRE BY WINDOW, COMPLETE MOVIES
SELECT gn genre,count(*) movies,sum((budget_cur='USD' AND gross_cur='USD')::int) complete,sum((budget_cur='USD' AND gross_cur='USD' AND y>=2000)::int) complete_2000,sum((budget_cur='USD' AND gross_cur='USD' AND y>=2010)::int) complete_2010 FROM f,unnest(string_split(genres,',')) u(gn) GROUP BY 1 ORDER BY 2 DESC LIMIT 14;
.print PROFIT PROXY (gross minus budget, complete movies)
SELECT count(*) n,sum((g<b)::int) gross_below_budget,sum((g<2.5*b)::int) gross_below_2_5x,median(g-b) med_margin,median(g/b) med_multiple FROM f WHERE budget_cur='USD' AND gross_cur='USD';
.print HIGHEST MARGINS
SELECT primaryTitle,y,round(b/1e6,1) budget_m,round(g/1e6,1) gross_m,round(g/b,1) mult FROM f WHERE budget_cur='USD' AND gross_cur='USD' ORDER BY g-b DESC LIMIT 6;
.print EXTREME MULTIPLES
SELECT primaryTitle,y,b,g,round(g/b,1) mult,budget_raw,gross_raw FROM f WHERE budget_cur='USD' AND gross_cur='USD' ORDER BY g/b DESC LIMIT 8;
SELECT primaryTitle,y,b,g,round(g/b,3) mult,budget_raw,gross_raw FROM f WHERE budget_cur='USD' AND gross_cur='USD' ORDER BY g/b LIMIT 8;
.print RANDOM RAW SAMPLE
SELECT primaryTitle,b,budget_cur,budget_raw,g,gross_cur,gross_raw FROM f WHERE b IS NOT NULL AND g IS NOT NULL ORDER BY hash(tconst) LIMIT 15;
.print UNPARSED SAMPLE
SELECT primaryTitle,budget_raw,gross_raw FROM f WHERE (b IS NULL AND budget_raw<>'') OR (g IS NULL AND gross_raw<>'') ORDER BY hash(tconst) LIMIT 10;
.print MAIN CAST
CREATE TEMP TABLE p AS SELECT pr.tconst,pr.nconst,pr.category,try_cast(pr.ordering AS INT) o FROM read_csv('raw/title.principals.tsv.gz',delim='\t',quote='',escape='',header=true,nullstr='\N',all_varchar=true) pr JOIN m USING(tconst);
CREATE TEMP TABLE ca AS SELECT *,row_number() OVER (PARTITION BY tconst ORDER BY o) r FROM p WHERE category IN ('actor','actress');
SELECT k top_billed,count(DISTINCT nconst) actors,count(*) credits FROM ca,(VALUES (3),(5),(10)) x(k) WHERE r<=k GROUP BY 1 ORDER BY 1;
.print WIKIDATA TIERS, USD BUDGET AND GROSS
CREATE TEMP TABLE allm AS SELECT b.tconst,try_cast(r.numVotes AS INT) v FROM read_csv('raw/title.basics.tsv.gz',delim='\t',quote='',escape='',header=true,nullstr='\N',all_varchar=true) b JOIN read_csv('raw/title.ratings.tsv.gz',delim='\t',quote='',escape='',header=true,nullstr='\N',all_varchar=true) r USING(tconst) WHERE b.titleType='movie' AND b.isAdult='0';
SELECT mn minvotes,count(*) movies,sum((tconst IN (SELECT t FROM wb) AND tconst IN (SELECT t FROM wg))::int) wd_both FROM allm,(VALUES (1000),(10000),(100000)) x(mn) WHERE v>=mn GROUP BY 1 ORDER BY 1;
.print WIKIDATA SCORES, BY SOURCE
SELECT replace(c3,'"','') src,count(DISTINCT c0) movies FROM read_csv('features/wd_scores.tsv',delim='\t',quote='',header=false,skip=1,columns={'c0':'VARCHAR','c1':'VARCHAR','c2':'VARCHAR','c3':'VARCHAR'}) GROUP BY 1 ORDER BY 2 DESC LIMIT 6;
COPY (SELECT * EXCLUDE(budget_raw,gross_raw) FROM f) TO 'features/movies_features.csv' (HEADER);
