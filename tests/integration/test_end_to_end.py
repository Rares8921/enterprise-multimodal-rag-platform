import pytest
import httpx
import time
import asyncio

@pytest.mark.asyncio
class TestEndToEnd:
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


@pytest.mark.asyncio
async def test_health_checks():

    async with httpx.AsyncClient() as client:

        response = await client.get("http://localhost:8000/health")
        assert response.status_code == 200
        assert response.json()['api'] == 'healthy'
