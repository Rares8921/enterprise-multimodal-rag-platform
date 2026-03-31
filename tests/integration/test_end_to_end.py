import pytest
import httpx

@pytest.mark.asyncio
async def test_health_checks():

    async with httpx.AsyncClient() as client:

        response = await client.get("http://localhost:8000/health")
        assert response.status_code == 200
        assert response.json()['api'] == 'healthy'
