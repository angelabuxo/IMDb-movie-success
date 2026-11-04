.mode markdown
SET threads=8;
CREATE TEMP TABLE m AS SELECT * FROM read_csv('wiki/movies_100k.csv',header=true,all_varchar=true);
CREATE TEMP TABLE p AS SELECT pr.* FROM read_csv('raw/title.principals.tsv.gz',delim='\t',quote='',escape='',header=true,nullstr='\N',all_varchar=true) pr JOIN m USING(tconst);
.print MOVIES
SELECT count(*) movies,min(try_cast(startYear AS INT)) y0,max(try_cast(startYear AS INT)) y1,median(try_cast(votes AS INT)) med_votes,max(try_cast(votes AS INT)) max_votes,round(avg(try_cast(averageRating AS DOUBLE)),2) avg_rating,sum((genres IS NULL)::int) no_genre FROM m;
.print CREDITS
SELECT category,count(*) credits,count(DISTINCT nconst) people,count(DISTINCT tconst) movies FROM p GROUP BY 1 ORDER BY 2 DESC;
SELECT count(*) credits,count(DISTINCT nconst) people,round(count(*)*1.0/count(DISTINCT tconst),1) per_movie,sum((characters IS NOT NULL)::int) with_character FROM p;
.print MOVIE-MOVIE EDGES VIA SHARED PERSON
CREATE TEMP TABLE c AS SELECT DISTINCT tconst,nconst,category IN ('actor','actress','self') is_cast FROM p;
SELECT 'all credits' g,count(*) edges,count(DISTINCT a) touched FROM (SELECT DISTINCT x.tconst a,y.tconst b FROM c x JOIN c y ON x.nconst=y.nconst AND x.tconst<y.tconst)
UNION ALL SELECT 'cast only',count(*),count(DISTINCT a) FROM (SELECT DISTINCT x.tconst a,y.tconst b FROM c x JOIN c y ON x.nconst=y.nconst AND x.tconst<y.tconst WHERE x.is_cast AND y.is_cast);
.print EXAMPLE
SELECT * FROM m WHERE tconst='tt1375666';
SELECT p.ordering,p.nconst,n.primaryName,p.category,p.characters FROM p JOIN read_csv('raw/name.basics.tsv.gz',delim='\t',quote='',escape='',header=true,nullstr='\N',all_varchar=true) n USING(nconst) WHERE tconst='tt1375666' ORDER BY try_cast(ordering AS INT) LIMIT 5;
.print REVIEWS HF
SELECT split_part(filename,'/',3) f,count(*) n,median(length(text)) med_chars FROM read_parquet('reviews/hf/*.parquet',filename=true) GROUP BY 1;
