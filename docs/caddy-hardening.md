# Caddy settings the application cannot set itself

Two of the audit fixes end at the reverse proxy, which lives on the VM
rather than in this repo. Neither has been tested from here — apply them on
the host and check with the commands at the bottom.

Both are additions to the existing site block for
`vidhyasagar-rag.duckdns.org`.

```caddyfile
vidhyasagar-rag.duckdns.org {
    # Refuse an oversized upload before it reaches the application.
    #
    # Caddy sets no body limit by default. The app now reads uploads in 1 MB
    # chunks and refuses past 50 MB (app/api/routes/documents.py), so it no
    # longer buffers a huge body — but it still has to read far enough to
    # know. This stops that at the edge. A little above the app's own limit
    # so the app is the thing that reports the friendly error.
    request_body {
        max_size 52MB
    }

    reverse_proxy 127.0.0.1:8000 {
        # Overwrite X-Forwarded-For rather than appending to it.
        #
        # Belt and braces. uvicorn now trusts only the private ranges
        # (app/api/rate_limit.py TRUSTED_PROXY_IPS) and walks the header from
        # the right, so a value a caller prepends is already ignored. With
        # this line there is nothing to prepend to: the header carries one
        # entry, the address Caddy actually saw.
        #
        # {remote_host} is Caddy's own view of the peer and cannot be set by
        # the caller.
        header_up X-Forwarded-For {remote_host}
    }
}
```

## Why the trust list matters more than it looks

The app used to run with `--forwarded-allow-ips "*"`, which makes uvicorn
take the **leftmost** `X-Forwarded-For` entry:

```python
if self.always_trust:
    return _parse_host_port(x_forwarded_for_hosts[0])
```

That was justified by the claim that Caddy overwrites the header. If Caddy
appends instead — which is its documented default — a caller who sends their
own `X-Forwarded-For` chooses the value the rate limiter keys on, and gets a
fresh budget on every request. The Dockerfile no longer depends on which
behaviour Caddy has; the `header_up` line above removes the question
entirely.

## Checking it

Replace `HOST` with the deployed host.

```bash
# 1. A spoofed forwarded header must not get a fresh rate-limit budget.
#    Send more requests than RATE_LIMIT allows in a minute, rotating the
#    header. Expect 429s to appear. Without the fix, every one returns 200.
for i in $(seq 1 80); do
  curl -s -o /dev/null -w "%{http_code} " \
    -H "X-Forwarded-For: 10.0.0.$i" \
    https://HOST/api/v1/health
done; echo

# 2. The body limit. Expect 413 from Caddy, well before 60 MB is uploaded.
head -c 60000000 /dev/zero > /tmp/big.pdf
curl -s -o /dev/null -w "%{http_code}\n" -F "file=@/tmp/big.pdf" \
  https://HOST/api/v1/documents/upload

# 3. Confirm the app sees a real client address rather than the gateway.
#    The rate-limit log line names the bucket it used; it should be a public
#    address, not 172.x.x.x.
docker logs rag-chatbot 2>&1 | grep rate_limit_exceeded | tail -5
```

If step 3 shows `172.*` as the client, uvicorn is not resolving the
forwarded address — check that the gateway falls inside the ranges in
`TRUSTED_PROXY_IPS` (`docker network inspect` will show the subnet).
