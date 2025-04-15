## Docker setup

This folder contains scripts to set up a local database with QAECY graph data. The dataset is built from the public Duplex model and contains the data used for the simple NL to SPARQL example in the [presentation](https://slides.qaecy.com/bilt-2025.html).

1. Build a Fuseki instance with the database *bilt2025* `./build_db.sh`
1. Open http://localhost:3030/#/dataset/bilt2025/query
1. Log in with `user/pw = admin/admin`
1. Try below query to confirm that [Lucene](https://lucene.apache.org/) based text queries are enabled:
    ```sparql
    PREFIX text:  <http://jena.apache.org/text#>

    SELECT ?match ?score ?literal
    WHERE {
        (?match ?score ?literal) text:query "A102"
    }
    ORDER BY DESC(?score)
    ```