#!/bin/sh
# Run a build step with an optional extra CA (BuildKit secret "extra_ca"), e.g. a corporate TLS
# proxy. No-op when the secret is absent (CI/release). The combined bundle lives in /tmp only for
# the duration of the step and never lands in an image layer.
set -eu
if [ -f /run/secrets/extra_ca ]; then
  base=/etc/ssl/certs/ca-certificates.crt
  [ -f "$base" ] || base=/dev/null
  cat "$base" /run/secrets/extra_ca > /tmp/extra-ca-bundle.pem
  export SSL_CERT_FILE=/tmp/extra-ca-bundle.pem CURL_CA_BUNDLE=/tmp/extra-ca-bundle.pem \
         REQUESTS_CA_BUNDLE=/tmp/extra-ca-bundle.pem NODE_EXTRA_CA_CERTS=/run/secrets/extra_ca \
         UV_NATIVE_TLS=1 GIT_SSL_CAINFO=/tmp/extra-ca-bundle.pem
  trap 'rm -f /tmp/extra-ca-bundle.pem' EXIT
fi
"$@"
