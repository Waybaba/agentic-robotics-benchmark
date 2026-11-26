#!/bin/sh
set -eu
test -d /home/user/submission
install -d -m 755 /home/user/practice
for file in ego_practice.py practice-home.mjb.gz practice-home.json ground-start.png scout-start.png; do
    install -m 644 "$(dirname "$0")/payload/$file" "/home/user/practice/$file"
done
