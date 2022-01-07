Run redis database in docker:

```
docker run --name redis_cont    \
    -e ALLOW_EMPTY_PASSWORD=yes \
    -p 6379:6379 bitnami/redis:latest
Unable to find image 'bitnami/redis:latest' locally
```