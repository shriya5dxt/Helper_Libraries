#!/bin/bash
# set -e

### Configuration ###

SERVER116=ubuntu@172.20.218.116
SERVER48=ubuntu@172.20.218.48
SERVER71=ubuntu@172.20.218.71
# APP_DIR=/var/www/myapp
CURRENT_DIR="$(pwd)"
# echo $CURRENT_DIR
PARENT_DIR=$(dirname "$CURRENT_DIR")
# echo $PARENT_DIR
HOME_DIR=$(dirname "$PARENT_DIR")
# echo $HOME_DIR
KEYFILE="$HOME_DIR/QCET_HP_AWS_SERVER_KEY.pem"
echo $KEYFILE

REMOTE_SCRIPT_PATH=/home/ubuntu/deploy/deploy_server_side.sh

### Library ###

function run()
{
  echo "Running: $@"
  "$@"
}

### Automation steps ###

if [[ "$KEYFILE" != "" ]]; then
  KEYARG="-i $KEYFILE"
else
  KEYARG=
fi

# run scp $KEYARG deploy_server_side.sh $SERVER:$REMOTE_SCRIPT_PATH
echo
echo "---- Running deployment script on 116 server ----"
run ssh $KEYARG $SERVER116 bash $REMOTE_SCRIPT_PATH
echo "---- Running deployment script on 48 server ----"
run ssh $KEYARG $SERVER48 bash $REMOTE_SCRIPT_PATH
echo "---- Running deployment script on 71 server ----"
run ssh $KEYARG $SERVER71 bash $REMOTE_SCRIPT_PATH

$SHELL