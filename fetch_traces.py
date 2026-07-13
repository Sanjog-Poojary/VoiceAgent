import os
import json
from dotenv import load_dotenv
from langsmith import Client

def main():
    load_dotenv()
    
    print("Connecting to LangSmith API...")
    client = Client()
    
    # We didn't set LANGCHAIN_PROJECT, so traces go to "default"
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    
    print(f"Fetching 3 most recent root traces from project '{project_name}'...\n")
    
    try:
        # Fetch recent runs. execution_order=1 means root runs (the functions we directly decorated)
        runs = list(client.list_runs(
            project_name="default",
            execution_order=1
        ))
        
        if not runs:
            print("No runs found. Make sure the API key is correct and traces were sent.")
            return

        out = []
        for i, run in enumerate(runs, 1):
            out.append({
                "trace": i,
                "name": run.name,
                "inputs": run.inputs,
                "outputs": run.outputs
            })
        with open("traces.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print("Wrote traces.json")
            
    except Exception as e:
        print(f"Error fetching traces: {e}")

if __name__ == "__main__":
    main()
