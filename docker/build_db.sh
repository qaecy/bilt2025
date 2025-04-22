#!/bin/bash

CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
TTL_DIR=$(cd "$(dirname "$0")/./data" && pwd)
SCRIPTS_DIR=$(cd "$(dirname "$0")/./scripts" && pwd)
DB_NAME="bilt2025"

# Start the triplestore
bash "$SCRIPTS_DIR/start-triplestore.sh"

# Create temp db
bash "$SCRIPTS_DIR/create-database.sh" $DB_NAME

# Load all files
for file in "$TTL_DIR"/*.ttl; do
    if [[ -f "$file" ]]; then
        filename=$(basename -- "$file")
        graph_iri="https://${filename%.*}"
        bash "$SCRIPTS_DIR/load-data.sh" $DB_NAME "$file" $graph_iri
        echo "Loaded file: $filename in named graph $graph_iri"
        # Add your processing logic here
    fi
done