#!/bin/bash

# Use: ./create-database.sh <database_name> <port> <user> <password>

if [ "$#" -lt 1 ]; then
	echo "Error: You must enter at least 1 of the arguments: <database_name> <port> <user> <password>"
	exit 1
fi

DATABASE_NAME=$1
PORT=${2:-3030}
USER=${3:-admin}
PASSWORD=${4:-admin}

echo "Creating database $DATABASE_NAME on port $PORT with user $USER"

TEMPLATE=$(cat <<EOF
    @prefix :       <http://base/#> .
    @prefix fuseki: <http://jena.apache.org/fuseki#> .
    @prefix rdfs:   <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix tdb2:   <http://jena.apache.org/2016/tdb#> .
    @prefix text:   <http://jena.apache.org/text#> .
    @prefix skos:   <http://www.w3.org/2004/02/skos/core#> .
    @prefix qcy:    <https://dev.qaecy.com/ont#> .

    :service_tdb_all  a  fuseki:Service;
        fuseki:name      "$DATABASE_NAME" ;
        rdfs:label       "TDB2 $DATABASE_NAME";
        fuseki:dataset   :text_ds;
            fuseki:endpoint  [ fuseki:name       "update";
                            fuseki:operation  fuseki:update
                            ];
            fuseki:endpoint  [ fuseki:name       "query";
                            fuseki:operation  fuseki:query
                            ];
            fuseki:endpoint  [ fuseki:name       "get";
                            fuseki:operation  fuseki:gsp-r
                            ];
            fuseki:endpoint  [ fuseki:name       "shacl";
                            fuseki:operation  fuseki:shacl
                            ];
            fuseki:endpoint  [ fuseki:name       "data";
                            fuseki:operation  fuseki:gsp-rw
                            ];
            fuseki:endpoint  [ fuseki:name       "sparql";
                            fuseki:operation  fuseki:query
                            ].

    :text_ds a text:TextDataset ;
        text:dataset :tdb_dataset_readwrite ;
        text:index :lucene .

    :tdb_dataset_readwrite a tdb2:DatasetTDB2;
        tdb2:location  "/fuseki-base/fuseki/databases/$DATABASE_NAME/tdb";
        tdb2:unionDefaultGraph true ;
        tdb2:transactionMode "transactional" ;
        tdb2:transactionLogType "journal" ;
        tdb2:blockFileSize "32M" ;
        tdb2:blockCacheSize "1G" ;
        tdb2:fileMode "direct" .

    :lucene a text:TextIndexLucene ;
        text:directory <file:/fuseki-base/fuseki/databases/$DATABASE_NAME/lucene> ;
        text:storeValues true ;
        text:entityMap :entity-map .

    :entity-map a text:EntityMap ;
        text:entityField "uri" ;
        text:graphField "graph" ; ## enable graph-specific indexing
        text:defaultField "text" ; ## Must be defined in the text:map
        text:uidField "uid" ;
        text:langField "lang" ;
        text:map (
            [ text:field "text" ; text:predicate skos:prefLabel ]
            [ text:field "text" ; text:predicate skos:altLabel ]
            [ text:field "text" ; text:predicate skos:hiddenLabel ]
            [ text:field "text" ; text:predicate skos:notation ]
            [ text:field "text" ; text:predicate rdfs:label ]
            [ text:field "text" ; text:predicate qcy:value ]
            [ text:field "text" ; text:predicate qcy:label ]
            ) .
EOF
)

curl -u $USER:$PASSWORD -X POST \
    -H "Content-Type: text/turtle" \
    --data "$TEMPLATE" \
    http://localhost:${PORT}/$/datasets

echo "Database $DATABASE_NAME created successfully"