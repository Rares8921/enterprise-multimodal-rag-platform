import os


def init_otel() -> None:

    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") is None:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        service_name = os.getenv("OTEL_SERVICE_NAME", "monitoring")
        resource = Resource.create({"service.name": service_name})

        provider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter())
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        # Instrument common libs (FastAPI is instrumented once app exists)
        HTTPXClientInstrumentor().instrument()

        # Expose helper for app instrumentation
        init_otel.instrument_fastapi = FastAPIInstrumentor

    except Exception:
        # Don't fail service if tracing can't be initialized
        return
