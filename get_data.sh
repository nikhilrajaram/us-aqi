#!/bin/bash

mkdir -p data/

curl 'https://storage.googleapis.com/kaggle-data-sets/1556851/2564999/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20220211%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20220211T161929Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=4a4992f07e030475b3935b9d8d767622035c6905f319c3c8d7b1d116ba32adea93ff53776a1bdb3fc8361f3b3dc962e300a68efe1ffc5bf766a0ed582758132660bd133233047ba97d0bf81a5e03ada1992023d6e3584acd312a57c35f90825d12a3a17ea38d8486e1f0a17c2b237c921c9ec4bc0b6454e1c520cf3b32a223648cd269954b1d4b405b090b5fadc6e7235643f318b07ee0bb62da1c1b99d6dc16d44375fe331e72c14d8090927e0dbe72c05f3cfe2aeabd2a023989797669d9e60b6dee563957a46fb353fc4172e03a5471cb5ec9645db533b6c624aa9e8b2007ba127553c669695c02629dda1059a2afc7968a50114caaa07b3ce1d082379556' \
    -H 'authority: storage.googleapis.com'   \
    -H 'upgrade-insecure-requests: 1'   \
    -H 'dnt: 1'   \
    -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.99 Safari/537.36'   \
    -H 'accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'   \
    -H 'sec-fetch-site: cross-site'   \
    -H 'sec-fetch-mode: navigate'   \
    -H 'sec-fetch-user: ?1'   \
    -H 'sec-fetch-dest: document'   \
    -H 'sec-ch-ua: " Not;A Brand";v="99", "Google Chrome";v="97", "Chromium";v="97"'   \
    -H 'sec-ch-ua-mobile: ?0'   \
    -H 'sec-ch-ua-platform: "macOS"'   \
    -H 'referer: https://www.kaggle.com/'   \
    -H 'accept-language: en-US,en;q=0.9'   \
    --compressed   \
    -o data/aqi_daily_1980_to_2021.csv.zip

unzip -d data/ data/aqi_daily_1980_to_2021.csv.zip
