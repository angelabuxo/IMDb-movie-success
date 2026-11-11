-- Credits and people for the movie graph. Run from the data folder:
--   cd data/raw && mkdir -p features/graph && duckdb < ../../scripts/network/graph_tables.sql
-- Movies are the 10,000+ vote set in features/movies_features_10k.csv.
-- Directors and writers come from title.crew, which lists them in full. Everyone else comes from
-- title.principals, which lists about ten people a title, so producers and crew there are partial.
SET threads=8;
CREATE TEMP TABLE mv AS SELECT tconst, try_cast(y AS INT) y FROM read_csv('features/movies_features_10k.csv', header=true, all_varchar=true);

CREATE TEMP TABLE pr AS
SELECT p.tconst, p.nconst,
       CASE WHEN p.category IN ('actor', 'actress') THEN 'actor' ELSE p.category END AS role,
       try_cast(p.ordering AS INT) AS ordering
FROM read_csv('raw/title.principals.tsv.gz', delim='\t', quote='', escape='', header=true, nullstr='\N', all_varchar=true) p
JOIN mv USING (tconst)
WHERE p.category IN ('actor', 'actress', 'producer', 'composer', 'cinematographer', 'editor');

CREATE TEMP TABLE cr AS
SELECT c.tconst, trim(x) AS nconst, role, NULL::INT AS ordering
FROM read_csv('raw/title.crew.tsv.gz', delim='\t', quote='', escape='', header=true, nullstr='\N', all_varchar=true) c
JOIN mv USING (tconst),
LATERAL (SELECT unnest(string_split(c.directors, ',')) AS x, 'director' AS role
         UNION ALL SELECT unnest(string_split(c.writers, ',')), 'writer')
WHERE x IS NOT NULL AND x <> '';

CREATE TEMP TABLE credits AS
SELECT tconst, nconst, role, min(ordering) AS ordering FROM (SELECT * FROM pr UNION ALL SELECT * FROM cr)
GROUP BY ALL;

COPY (SELECT * FROM credits ORDER BY tconst, role, ordering NULLS LAST, nconst)
TO 'features/graph/credits.csv' (HEADER);

COPY (
  SELECT n.nconst, n.primaryName AS name, try_cast(n.birthYear AS INT) AS birth_year,
         try_cast(n.deathYear AS INT) AS death_year
  FROM read_csv('raw/name.basics.tsv.gz', delim='\t', quote='', escape='', header=true, nullstr='\N', all_varchar=true) n
  WHERE n.nconst IN (SELECT nconst FROM credits)
  ORDER BY n.nconst
) TO 'features/graph/people.csv' (HEADER);

SELECT role, count(*) AS credits, count(DISTINCT nconst) AS people, count(DISTINCT tconst) AS movies
FROM credits GROUP BY 1 ORDER BY 2 DESC;
