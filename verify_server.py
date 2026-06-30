import subprocess
import time
import sys
import httpx

def test_endpoints():
    base_url = "http://127.0.0.1:8000"
    
    print("\n--- Starting Mock API Server Verification Tests ---")
    
    # 1. Get user details
    print("\n[Test 1] GET /api/users/1")
    res = httpx.get(f"{base_url}/api/users/1")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 200
    assert res.json()["name"] == "Sanjog"
    assert res.json()["base_language"] == "English"
    print("-> SUCCESS")

    # 2. Get event details
    print("\n[Test 2] GET /api/events/2")
    res = httpx.get(f"{base_url}/api/events/2")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 200
    assert res.json()["event_type"] == "Credit Expiry"
    print("-> SUCCESS")

    # 3. Get offer details
    print("\n[Test 3] GET /api/offers/3")
    res = httpx.get(f"{base_url}/api/offers/3")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 200
    assert "Fossil Watch" in res.json()["recommendations"]
    print("-> SUCCESS")

    # 4. Get non-existent user (expect 404)
    print("\n[Test 4] GET /api/users/999 (Non-existent)")
    res = httpx.get(f"{base_url}/api/users/999")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 404
    print("-> SUCCESS")

    # 5. POST WhatsApp notification
    print("\n[Test 5] POST /api/notify/whatsapp")
    payload_wa = {
        "customer_id": "1",
        "phone": "+1234567890",
        "message": "Happy Birthday Sanjog! Here is a 20% coupon: BIRTHDAY20"
    }
    res = httpx.post(f"{base_url}/api/notify/whatsapp", json=payload_wa)
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    print("-> SUCCESS")

    # 6. POST CRM ticket
    print("\n[Test 6] POST /api/tickets/crm")
    payload_crm = {
        "customer_id": "2",
        "issue_description": "User hung up during verification phase.",
        "priority": "high"
    }
    res = httpx.post(f"{base_url}/api/tickets/crm", json=payload_crm)
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.json()}")
    assert res.status_code == 200
    assert res.json()["status"] == "success"
    print("-> SUCCESS")

    print("\n--- All tests completed successfully! ---")

if __name__ == "__main__":
    # Start the FastAPI server using the virtual environment's uvicorn
    # Use python executable from active virtual env
    python_exe = sys.executable
    server_process = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "mock_server:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the server to spin up
    print("Spinning up mock API server...")
    time.sleep(2)
    
    try:
        test_endpoints()
    except Exception as e:
        print(f"\nTESTS FAILED: {e}")
        # Print server logs to help debugging
        stdout, stderr = server_process.communicate(timeout=1)
        print("--- Server stdout ---")
        print(stdout)
        print("--- Server stderr ---")
        print(stderr)
        sys.exit(1)
    finally:
        # Terminate server
        print("\nShutting down mock API server...")
        server_process.terminate()
        server_process.wait()
        print("Server shutdown completed.")
