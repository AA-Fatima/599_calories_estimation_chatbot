import uvicorn
from app.main import app
import os

if __name__ == "__main__": 
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Set to False for production
    )