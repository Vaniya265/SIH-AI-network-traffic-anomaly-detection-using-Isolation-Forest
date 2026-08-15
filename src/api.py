from fastapi import FastAPI
from pydantic import BaseModel

from predict import predict_one


app = FastAPI(
    title="Network Traffic Anomaly Detection API",
    description="SIH AI Network Security Detection System",
    version="1.0"
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


@app.post("/predict")
def predict_traffic(data: TrafficData):

    result = predict_one(
        data.model_dump()
    )

    return result