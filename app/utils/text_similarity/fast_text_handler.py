from loguru import logger
import sentry_sdk
import uuid
from datetime import datetime
from app.utils.text_similarity.fast_text import fasttext_object


class FastTextCreator:
    def __init__(self):
        pass

    @staticmethod
    def get_fasttext_response(query):
        logger.debug("Getting the response body...")
        with sentry_sdk.start_span(
                op="intelligence fast_text_handler",
                description="開始偵測流程，並等候預測結果"
        ) as span:
            query_text = query.QueryBody
            text, title, tag, link, summery, sourse, similarity = fasttext_object.fasttext(query_text)
            response_body = {
                "QueryId": str(query.QueryId),
                "ResponseId": str(uuid.uuid4()),
                "ResponseMessage": "Success",
                "ResponseBody": {
                    "Text": text,
                    "Title": title,
                    "Tag": tag,
                    "Link": link,
                    "Summery": summery,
                    "Sourse": sourse,
                    "Similarity": similarity
                },
                "ResponseDateTime": datetime.now()
            }
            logger.info(f'fasttext response: {response_body}')
            logger.debug("Finished getting the response body.")
            return response_body


fasttext_creator_object = FastTextCreator()
