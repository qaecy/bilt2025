#!/bin/bash

# Use: ./load-data.sh <database_name> <file_path> <graph_iri> <port> <user> <password>

if [ "$#" -lt 2 ]; then
	echo "Error: You must enter at least 2 of the arguments: <database_name> <file_path> <graph_iri> <port> <user> <password>"
	exit 1
fi

DATABASE_NAME=$1
FILE_PATH=$2
GRAPH_IRI=$3
PORT=${4:-3030}
USER=${5:-admin}
PASSWORD=${6:-admin}

echo "Loading TTL in database $DATABASE_NAME in named graph $GRAPH_IRI on port $PORT with user $USER"

# GET SESSION COOKIE
# curl -v -s -c cookies.txt "username=${USER}&password=${PASSWORD}" "http://localhost:${PORT}/$DATABASE_NAME/data"

if [ -z "$GRAPH_IRI" ]; then
    ENDPOINT=http://localhost:${PORT}/$DATABASE_NAME/data
else
    ENDPOINT=http://localhost:${PORT}/$DATABASE_NAME/data?graph=$GRAPH_IRI
fi

curl -u $USER:$PASSWORD -X POST \
    -H "Content-Type: text/turtle" \
    --data-binary "@$FILE_PATH" \
    $ENDPOINT

echo "TTL successfully loaded in $DATABASE_NAME"