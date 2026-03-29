import asyncio
import redis.asyncio as aioredis
import logging
import asyncpg, json
import aiohttp, time
import datetime, math
import random
import numpy as np
import mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
from prometheus_client import Gauge, Counter
DATA_DRIFT_SCORE = Gauge('data_drift_score', 'Data drift detection score', ['feature', 'tenant_id'])
CONCEPT_DRIFT_DETECTED = Counter('concept_drift_detected_total', 'Concept drift detections', ['tenant_id'])
MODEL_PERFORMANCE_SCORE = Gauge('model_user_feedback_score', 'Current user feedback score', ['tenant_id'])

from ..config import Settings
settings = Settings()

# Global state
redis_client: aioredis.Redis = None
pg_pool: asyncpg.Pool = None

from DriftDetector import DriftDetector

async def trigger_drift_alert(alert_type: str, tenant_id: str, result: dict):
    # webhook
    logger.critical(f"DRIFT ALERT [{alert_type}] for Tenant {tenant_id}: {result}")

    if not settings.slack_webhook_url:
        return

    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "text": f"(!!!) *Drift Alert: {alert_type}* (Tenant: `{tenant_id}`)\n```json\n{json.dumps(result, indent=2)}\n```"
            }
            await session.post(settings.slack_webhook_url, json=payload)
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")


async def get_sleep_time_until_next_run(interval_seconds: int) -> float:
    # exact sleep time to align executions
    now = datetime.now().timestamp()
    next_run = math.ceil(now / interval_seconds) * interval_seconds
    return next_run - now


async def monitor_drift_continuous():
    global redis_client, pg_pool

    redis_client = await aioredis.from_url(settings.redis_url)
    pg_pool = await asyncpg.create_pool(settings.postgres_url)
    detector = DriftDetector()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("production_drift_monitoring")

    logger.info("Starting production drift detection service")

    while True:
        try:
            sleep_time = await get_sleep_time_until_next_run(settings.drift_check_interval)
            await asyncio.sleep(sleep_time)

            logger.info("Executing scheduled drift checks...")

            tenants = await pg_pool.fetch(
                "SELECT DISTINCT tenant_id FROM monitoring_events WHERE timestamp >= NOW() - INTERVAL '24 hours'")

            with mlflow.start_run(run_name=f"drift_cycle_{int(time.time())}"):
                for tenant_record in tenants:
                    tenant_id = tenant_record['tenant_id']

                    recent_events = await pg_pool.fetch('''
                        SELECT query_length, embedding, user_feedback_score
                        FROM monitoring_events
                        WHERE tenant_id = $1 
                          AND timestamp >= NOW() - INTERVAL '24 hours'
                    ''', tenant_id)

                    baseline_events = await pg_pool.fetch('''
                        SELECT query_length, embedding, user_feedback_score
                        FROM monitoring_events
                        WHERE tenant_id = $1 
                          AND timestamp >= NOW() - INTERVAL '8 days'
                          AND timestamp < NOW() - INTERVAL '24 hours'
                    ''', tenant_id)

                    if not recent_events or not baseline_events:
                        continue

                    sample_size = min(1000, len(recent_events))
                    recent_sample = random.sample(recent_events, sample_size)
                    baseline_sample = random.sample(baseline_events, min(1000, len(baseline_events)))

                    # Data Drift
                    recent_lengths = [r['query_length'] for r in recent_sample if r['query_length'] is not None]
                    baseline_lengths = [b['query_length'] for b in baseline_sample if b['query_length'] is not None]

                    if recent_lengths and baseline_lengths:
                        data_drift = await detector.detect_data_drift('query_length', recent_lengths, baseline_lengths)
                        DATA_DRIFT_SCORE.labels(feature='query_length', tenant_id=tenant_id).set(data_drift['psi'])
                        mlflow.log_metric(f"psi_query_length_{tenant_id}", data_drift['psi'])

                        if data_drift['drift_detected'] and settings.alert_on_drift:
                            await trigger_drift_alert('Data Drift', tenant_id, data_drift)

                    # Concept Drift
                    recent_feedback = [r['user_feedback_score'] for r in recent_sample if
                                       r['user_feedback_score'] is not None]
                    baseline_feedback = [b['user_feedback_score'] for b in baseline_sample if
                                         b['user_feedback_score'] is not None]

                    if recent_feedback and baseline_feedback:
                        recent_avg = np.mean(recent_feedback)
                        baseline_avg = np.mean(baseline_feedback)

                        MODEL_PERFORMANCE_SCORE.labels(tenant_id=tenant_id).set(recent_avg)
                        mlflow.log_metric(f"feedback_score_{tenant_id}", recent_avg)

                        concept_drift = await detector.detect_concept_drift(recent_avg, baseline_avg,
                                                                            settings.concept_drift_threshold)
                        if concept_drift['drift_detected']:
                            CONCEPT_DRIFT_DETECTED.labels(tenant_id=tenant_id).inc()
                            mlflow.log_metric(f"accuracy_drop_{tenant_id}", concept_drift['accuracy_drop'])
                            if settings.alert_on_drift:
                                await trigger_drift_alert('Concept Drift', tenant_id, concept_drift)

                    # Embedding Drift
                    recent_embeddings = [json.loads(r['embedding']) for r in recent_sample if r['embedding']]
                    baseline_embeddings = [json.loads(b['embedding']) for b in baseline_sample if b['embedding']]

                    if recent_embeddings and baseline_embeddings:
                        emb_drift = await detector.detect_embedding_drift(np.array(recent_embeddings),
                                                                          np.array(baseline_embeddings))
                        mlflow.log_metric(f"embedding_centroid_distance_{tenant_id}", emb_drift['centroid_distance'])

                        if emb_drift['drift_detected'] and settings.alert_on_drift:
                            await trigger_drift_alert('Embedding Drift', tenant_id, emb_drift)

            logger.info("Completed tenant drift checks cycle.")

        except Exception as e:
            logger.error(f"Drift monitoring critical error: {str(e)}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(monitor_drift_continuous())