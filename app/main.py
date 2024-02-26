import os
import sys
from dotenv import load_dotenv
import uvicorn
import sentry_sdk
from sentry_sdk import set_user
import traceback
from fastapi import FastAPI, HTTPException
from loguru import logger
from utils.authentication.util import *

load_dotenv('conf/env')
from utils.text_similarity.fast_text_handler import fasttext_creator_object
from shared.app.MsgInterChangeFormat.fraud_detector_communication import *

app = FastAPI()
logger.remove()
logger.add(sys.stdout, level=os.getenv("LOG_LEVEL"))

try:
    # 從 Secret Manager 中取出 Sentry dsn
    # os.environ['SCX_SENTRY_DSN'] = _load_config_value_from_secret_manager('SCX_SENTRY_DSN')
    os.environ['SCX_SENTRY_DSN'] = os.getenv('SCX_SENTRY_DSN')
except Exception as e:
    logger.warning(f"load key: SCX_SENTRY_DSN from key manager failed: {e}")

if not os.getenv('SCX_SENTRY_DSN'):
    logger.info("system started without sentry")
else:
    logger.info("system started with sentry")
    sentry_sdk.init(
        dsn=os.getenv('SCX_SENTRY_DSN'),
        traces_sample_rate=float(os.getenv('SCX_SENTRY_TRACE_RATE')),
    )
    set_user({"email": "detector@example.com"})


@app.post("/fasttext")
async def start_fasttext(query: FDRequestModel, response_model_exclude_none=True):
    logger.info(f"QueryID: {str(query.QueryId)}")
    if not query:
        logger.warning(f"No query provided. Agent_name: 'fraud_detector', QueryID: {str(query.QueryId)}")
        raise HTTPException(status_code=400, detail="Bad Request")
    with sentry_sdk.start_transaction(
            name=f"/sms_intelligence",
            op="sms_intelligence",
            trace_id=query.Sentry.TraceId,
            parent_span_id=query.Sentry.SpanId,
    ) as transaction:
        with transaction.start_child(
                op="intelligence main",
                description=f"start to match intelligence"
        ) as span:
            try:
                with sentry_sdk.start_span(
                        op="detector pedict with fastttext",
                        description="建立fasttext任務"
                ) as span:
                    response_data = fasttext_creator_object.get_fasttext_response(query)
                    return response_data
            except (Exception, HTTPException) as e:
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/health-check")
async def health_check():
    return 'Alive'


if __name__ == '__main__':
    reload = load_bool_from_env('SCX_SERVER_RELOAD')
    uvicorn.run(
        app="main:app",
        host=os.getenv('SCX_SERVER_HOST'),
        port=int(os.getenv('SCX_SERVER_PORT')),
        reload=reload,
        workers=int(os.getenv('SCX_SERVER_WORKER'))
    )
