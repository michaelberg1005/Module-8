# Module-8
Movies ETL
## Assumptions
1. The 3 data sets we are importing will be in the same format (wiki = json, kaggle metadata = csv, and ratings data = csv). Should the files not be in these specific formats, they will fail to load at the very beginning of the ETL automated process. There also wont be any other data sets to be input into this process.
2. The wiki data and kaggle metadata will be small enough files that they can be loaded all at once, and won't require batches, like the ratings data. Should the file sizes be too large, they may fail to write to the database without error, in the load part of the ETL process at the very end.
3. There are no other languages then the ones listed in our code in the data set that also have an alternate title.
4. There won't be any other columns in the data set that are supposed to be monetary data outside of box office (revenue) and budget, that need to parsed and cleaned and made into a uniform manne
5. There won't be any other numerical data in the kaggle data that needs to be converted.
6. There won't be any other dates that need to be convereted outside of release date (ie. final date in theatre or range of dates in theatre).
7. The kaggle ids and the ratings movies ids are a match and for the same movie.
8. The database we are using already has a movies_data database created, spelled exactly that way. The database also uses a 5432 as the last 4 digits of the URL prior to locating the database.
