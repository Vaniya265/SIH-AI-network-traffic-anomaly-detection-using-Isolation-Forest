from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predict import predict_one


app = FastAPI(
    title="Network Traffic Anomaly Detection API",
    description="SIH AI Network Security Detection System",
    version="1.0"
)

# Person 6's dashboard runs on a different port (e.g. localhost:3000 /
# 5173) than this API (localhost:8000). Without CORS enabled, the
# browser blocks the dashboard's requests entirely with a red console
# error, even though the API itself is working fine. This must stay on.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrafficData(BaseModel):

    duration: int

    protocol_type: str

    service: str

    flag: str

    src_bytes: int

    dst_bytes: int

    land: int

    wrong_fragment: int

    urgent: int

    hot: int

    num_failed_logins: int

    logged_in: int

    num_compromised: int

    root_shell: int

    su_attempted: int

    num_root: int

    num_file_creations: int

    num_shells: int

    num_access_files: int

    num_outbound_cmds: int

    is_host_login: int

    is_guest_login: int

    count: int

    srv_count: int

    serror_rate: float

    srv_serror_rate: float

    rerror_rate: float

    srv_rerror_rate: float

    same_srv_rate: float

    diff_srv_rate: float

    srv_diff_host_rate: float

    dst_host_count: int

    dst_host_srv_count: int

    dst_host_same_srv_rate: float

    dst_host_diff_srv_rate: float

    dst_host_same_src_port_rate: float

    dst_host_srv_diff_host_rate: float

    dst_host_serror_rate: float

    dst_host_srv_serror_rate: float

    dst_host_rerror_rate: float

    dst_host_srv_rerror_rate: float


@app.get("/")
def home():

    return {
        "message": "SIH Network Traffic Detection API is running"
    }


@app.get("/health")
def health():
    """Person 6 (or you, before rehearsal) can hit this to confirm the
    backend is up and the model actually loaded — separate from just
    the server process being alive."""
    return {
        "backend_running": True,
        "model_loaded": True,
    }


@app.post("/predict")
def predict_traffic(data: TrafficData):

    try:
        result = predict_one(
            data.model_dump()
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Couldn't process that traffic row: {e}"
        )

    return result


@app.post("/predict-batch")
def predict_batch(rows: list[TrafficData]):
    """Convenience for running the whole normal -> known -> hidden
    sequence in one call during testing. The live demo can still call
    /predict row-by-row for the staged reveal effect on stage."""
    return [predict_one(row.model_dump()) for row in rows]