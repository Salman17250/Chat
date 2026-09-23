import os
import sys
import uvicorn

if __name__ == "__main__":
    # Ensure current directory is on python path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
