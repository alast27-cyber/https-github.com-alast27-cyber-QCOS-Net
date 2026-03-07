import subprocess
import json
import os
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="QCOS Bridge Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    context: float
    energy: float

@app.post("/api/infer")
async def run_inference(req: InferenceRequest):
    try:
        # Calls the C++ inference stub
        cpp_executable = os.path.join(os.path.dirname(__file__), "..", "cpp", "build", "qcos_inference")
        
        if not os.path.exists(cpp_executable):
            # Fallback to python simulation if C++ binary is not built
            import torch
            from ..models.os_kernel_net import OSKernelNet
            model = OSKernelNet()
            v_score, _ = model(torch.tensor([[req.context, req.energy]]))
            return {"v_score": v_score.item(), "status": "simulated"}
            
        result = subprocess.run(
            [cpp_executable, str(req.context), str(req.energy)],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        output_lines = result.stdout.strip().split('\n')
        v_score = float(output_lines[-1].split(':')[-1].strip())
        
        return {"v_score": v_score, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
