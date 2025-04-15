#!/bin/sh
ADMIN_PASSWORD=admin
CURRENT_DIR=$(cd "$(dirname "$0")" && pwd)
CONFIG_DIR=$(cd "$(dirname "$0")/../config" && pwd)
DATA_DIR=$(cd "$(dirname "$0")/../fuseki-data" && pwd)

# BUILD IMAGE
chmod +x $CURRENT_DIR/start-triplestore.sh
docker build -t fuseki-5.3.0 -f "$CONFIG_DIR/Dockerfile" .

docker run --rm -it -p 3030:3030  \
    --name triplestore \
    -e ADMIN_PASSWORD=$ADMIN_PASSWORD  \
    -e QUERY_TIMEOUT=60000  \
    --detach \
    fuseki-5.3.0