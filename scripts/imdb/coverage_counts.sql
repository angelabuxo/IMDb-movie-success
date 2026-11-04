.mode markdown
SET threads=8;
CREATE OR REPLACE MACRO rd(f) AS TABLE SELECT * FROM read_csv('raw/'||f||'.tsv.gz', delim='\t', quote='', escape='', header=true, nullstr='\N', all_varchar=true, ignore_errors=true);
CREATE TEMP TABLE en AS SELECT DISTINCT replace(column0,'"','') id FROM read_csv('wd_enwiki.tsv',delim='\t',quote='',header=false,skip=1,all_varchar=true,columns={'column0':'VARCHAR'},ignore_errors=true);
CREATE TEMP TABLE anyw AS SELECT DISTINCT replace(column0,'"','') id FROM read_csv('wd_anywiki.tsv',delim='\t',quote='',header=false,skip=1,all_varchar=true,columns={'column0':'VARCHAR'},ignore_errors=true);
CREATE TEMP TABLE t AS SELECT b.tconst,b.titleType,b.isAdult,b.primaryTitle,b.startYear,try_cast(r.numVotes AS INT) v, b.tconst IN (SELECT id FROM en) e, b.tconst IN (SELECT id FROM anyw) a FROM rd('title.basics') b LEFT JOIN rd('title.ratings') r USING(tconst);
CREATE TEMP TABLE n AS SELECT nconst,primaryName,primaryProfession, nconst IN (SELECT id FROM en) e, nconst IN (SELECT id FROM anyw) a FROM rd('name.basics');
CREATE TEMP TABLE pr AS SELECT tconst,nconst,category FROM rd('title.principals');
SELECT 'title.basics' f,count(*) AS nrows FROM t UNION ALL SELECT 'name.basics',count(*) FROM n UNION ALL SELECT 'title.principals',count(*) FROM pr UNION ALL SELECT 'title.crew',count(*) FROM rd('title.crew') UNION ALL SELECT 'title.ratings',(SELECT count(*) FROM rd('title.ratings')) UNION ALL SELECT 'title.episode',count(*) FROM rd('title.episode') UNION ALL SELECT 'title.akas',count(*) FROM rd('title.akas');
.print TITLES BY TYPE
SELECT titleType,count(*) n,sum(e::int) enwiki,round(100.0*sum(e::int)/count(*),1) en_pct,sum(a::int) anywiki,round(100.0*sum(a::int)/count(*),1) any_pct FROM t GROUP BY 1 UNION ALL SELECT 'ALL',count(*),sum(e::int),round(100.0*sum(e::int)/count(*),1),sum(a::int),round(100.0*sum(a::int)/count(*),1) FROM t ORDER BY n DESC;
.print MOVIES BY VOTES (non-adult)
SELECT m minvotes,count(*) n,sum(e::int) enwiki,round(100.0*sum(e::int)/count(*),1) en_pct,sum(a::int) anywiki,round(100.0*sum(a::int)/count(*),1) any_pct FROM t, (VALUES (0),(100),(1000),(10000),(100000)) x(m) WHERE titleType='movie' AND isAdult='0' AND coalesce(v,0)>=m GROUP BY 1 ORDER BY 1;
.print PEOPLE
SELECT 'all names' grp,count(*) n,sum(e::int) enwiki,round(100.0*sum(e::int)/count(*),1) en_pct,sum(a::int) anywiki,round(100.0*sum(a::int)/count(*),1) any_pct FROM n;
CREATE TEMP TABLE c AS SELECT nconst,count(*) k FROM pr GROUP BY 1;
SELECT '>='||m||' credits' grp,count(*) n,sum(e::int) enwiki,round(100.0*sum(e::int)/count(*),1) en_pct,sum(a::int) anywiki,round(100.0*sum(a::int)/count(*),1) any_pct FROM n JOIN c USING(nconst),(VALUES (1),(5),(20),(100)) x(m) WHERE k>=m GROUP BY m ORDER BY m;
.print CAST OF NON-ADULT MOVIES BY MIN VOTES (actor/actress)
CREATE TEMP TABLE cast1 AS SELECT DISTINCT p.nconst,m FROM pr p JOIN t USING(tconst),(VALUES (1000),(10000),(100000)) x(m) WHERE t.titleType='movie' AND t.isAdult='0' AND coalesce(t.v,0)>=m AND p.category IN ('actor','actress');
SELECT m minvotes,count(*) people,sum(e::int) enwiki,round(100.0*sum(e::int)/count(*),1) en_pct FROM cast1 JOIN n USING(nconst) GROUP BY 1 ORDER BY 1;
.print IDS NOT JOINED
SELECT sum((id LIKE 'tt%' AND id NOT IN (SELECT tconst FROM t))::int) tt_unmatched, sum((id LIKE 'nm%' AND id NOT IN (SELECT nconst FROM n))::int) nm_unmatched, sum((id NOT LIKE 'tt%' AND id NOT LIKE 'nm%')::int) other FROM en;
COPY (SELECT * FROM (SELECT tconst,primaryTitle,startYear,v FROM t WHERE titleType='movie' AND isAdult='0' AND v>=1000 AND NOT e) ORDER BY hash(tconst) LIMIT 100) TO 'sample_movies_nolink.csv' (HEADER);
COPY (SELECT * FROM (SELECT nconst,primaryName,primaryProfession,k FROM n JOIN c USING(nconst) WHERE k>=20 AND NOT e) ORDER BY hash(nconst) LIMIT 100) TO 'sample_people_nolink.csv' (HEADER);
