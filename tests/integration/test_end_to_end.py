import pytest
import httpx
from pathlib import Path

import time
import asyncio

@pytest.mark.asyncio
class TestEndToEnd:

    @pytest.fixture
    async def client(self):
        async with httpx.AsyncClient(base_url="http://localhost:8000", timeout=120.0) as client:
            yield client

    @pytest.fixture
    def sample_pdf(self):
        return Path("tests/data/sample_contract.pdf")

    async def test_document_upload_and_query(self, client, sample_pdf):
        tenant_id = "test-tenant-001"

        # Step 1: Upload document
        with open(sample_pdf, 'rb') as f:
            upload_response = await client.post(
                "/documents/upload",
                data={
                    'tenant_id': tenant_id,
                    'doc_type': 'legal_contract'
                },
                files={'file': f}
            )

        assert upload_response.status_code == 200
        doc_id = upload_response.json()['doc_id']
        print(f"Document uploaded: {doc_id}")

        # Step 2: Wait for processing
        max_wait = 120  # 2 minutes
        start_time = time.time()

        while time.time() - start_time < max_wait:
            status_response = await client.get(f"/documents/{doc_id}/status")
            status = status_response.json()['status']

            print(f"Status: {status}")

            if status == 'indexed':
                break
            elif status == 'failed':
                pytest.fail(f"Document processing failed: {status_response.json().get('error')}")

            await asyncio.sleep(5)

        assert status == 'indexed', "Document processing timed out"

        # Step 3: Query the document
        query_response = await client.post(
            "/query",
            json={
                'query': 'What are the key terms of this contract?',
                'tenant_id': tenant_id,
                'doc_id': doc_id,
                'top_k': 5
            }
        )

        assert query_response.status_code == 200
        query_data = query_response.json()

        assert 'answer' in query_data
        assert len(query_data['answer']) > 0
        assert 'citations' in query_data
        assert query_data['latency_ms'] < 1000  # < 1 second

        print(f"Query completed in {query_data['latency_ms']:.0f}ms")
        print(f"Answer: {query_data['answer'][:200]}...")

    async def test_multi_document_retrieval(self, client):
        tenant_id = "test-tenant-002"

        # Query without specifying doc_id (search all documents)
        query_response = await client.post(
            "/query",
            json={
                'query': 'What are common liability clauses?',
                'tenant_id': tenant_id,
                'doc_type': 'legal_contract',
                'top_k': 10
            }
        )

        assert query_response.status_code == 200
        data = query_response.json()

        # Should retrieve from multiple documents
        doc_ids = set(c['doc_id'] for c in data.get('citations', []))
        print(f"Retrieved from {len(doc_ids)} documents")

    async def test_cache_performance(self, client):
        tenant_id = "test-tenant-003"
        query = "What is the termination clause?"

        # First query (cold)
        response1 = await client.post(
            "/query",
            json={'query': query, 'tenant_id': tenant_id}
        )
        latency1 = response1.json()['latency_ms']

        # Second query (should be cached)
        response2 = await client.post(
            "/query",
            json={'query': query, 'tenant_id': tenant_id}
        )
        latency2 = response2.json()['latency_ms']

        print(f"Cold: {latency1:.0f}ms, Cached: {latency2:.0f}ms")

        # Cached should be significantly faster
        assert latency2 < latency1 * 0.5, "Cache not working effectively"

    async def test_document_deletion_gdpr(self, client, sample_pdf):
        tenant_id = "test-tenant-gdpr"

        # Upload
        with open(sample_pdf, 'rb') as f:
            upload_response = await client.post(
                "/documents/upload",
                data={'tenant_id': tenant_id, 'doc_type': 'legal_contract'},
                files={'file': f}
            )

        doc_id = upload_response.json()['doc_id']

        # Wait for indexing
        await asyncio.sleep(30)

        # Delete
        delete_response = await client.delete(
            f"/documents/{doc_id}",
            headers={'X-Tenant-Id': tenant_id}
        )

        assert delete_response.status_code == 200

        # Verify document no longer queryable
        query_response = await client.post(
            "/query",
            json={
                'query': 'test',
                'tenant_id': tenant_id,
                'doc_id': doc_id
            }
        )

        # Should return no results
        assert query_response.json().get('citations', []) == []


@pytest.mark.asyncio
async def test_health_checks():

    async with httpx.AsyncClient() as client:

        response = await client.get("http://localhost:8000/health")
        assert response.status_code == 200
        assert response.json()['api'] == 'healthy'
