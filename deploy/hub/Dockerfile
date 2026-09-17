# syntax=docker/dockerfile:1
ARG HUB_IMAGE
ARG BASE_IMAGE
FROM ${HUB_IMAGE} AS runtime_package
FROM ${BASE_IMAGE}
ARG BASE_IMAGE
ARG HUB_IMAGE
ARG SOURCE_REV
RUN /app/.venv/bin/python -c 'import re,sys; assert all(re.fullmatch(r".+@sha256:[a-f0-9]{64}", v) for v in sys.argv[1:])' "${BASE_IMAGE}" "${HUB_IMAGE}"
COPY --from=runtime_package /opt/hub-runtime-wheel /tmp/hub-runtime-wheel
RUN uv pip install --python /app/.venv/bin/python --no-deps /tmp/hub-runtime-wheel/*.whl
WORKDIR /app
COPY hub_runtime /app/hub_runtime
ENV GRADIO_ANALYTICS_ENABLED=False PYTHONUNBUFFERED=1
LABEL org.opencontainers.image.source="https://github.com/FC-TTD/FireRedAudio" org.opencontainers.image.revision="${SOURCE_REV}"
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 CMD /app/.venv/bin/python -c 'import urllib.request;urllib.request.urlopen("http://127.0.0.1:8000/health",timeout=3).read()'
CMD ["/app/.venv/bin/python", "-m", "hub_runtime"]
