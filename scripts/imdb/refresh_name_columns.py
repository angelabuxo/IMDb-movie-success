"""Copy the name columns (studios, distributors, producers, starring) from the infobox tables into
features/movies_features.csv and features/movies_features_10k.csv, matched by tconst. Run from the data
folder after parse_infobox.py:

    cd data/raw && python /path/to/repo/scripts/imdb/refresh_name_columns.py

Those columns were copied from wiki_infobox*.csv when the movie tables were built, and nothing in the
repo rebuilds the movie tables, so a parser fix only reaches them through this script. Every value is
read and written as text, so no other column changes; rows missing from the infobox keep their values.
"""

import pandas as pd

NAMES = ["studios", "distributors", "producers", "starring"]
TABLES = {"features/movies_features.csv": ["features/wiki_infobox.csv"],
          "features/movies_features_10k.csv": ["features/wiki_infobox.csv", "features/wiki_infobox_10k.csv"]}


def main():
    for table, sources in TABLES.items():
        m = pd.read_csv(table, dtype=str, keep_default_na=False)
        ib = (pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in sources])
              .drop_duplicates("tconst").set_index("tconst"))
        hit = m.tconst.isin(ib.index)
        changed = {}
        for c in [c for c in NAMES if c in m.columns]:
            new = m.tconst.map(ib[c]).where(hit, m[c])
            changed[c] = int((new != m[c]).sum())
            m[c] = new
        m.to_csv(table, index=False)
        print(f"{table}: {len(m):,} rows, {int(hit.sum()):,} in the infobox tables; changed {changed}")


if __name__ == "__main__":
    main()
