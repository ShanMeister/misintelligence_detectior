import os
import fasttext.util
import pandas as pd
import numpy as np
from loguru import logger
import hashlib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from shared.app.library.mysql_implement import mysql_object
from shared.app.library.redis_implement import redis_object
ft = fasttext.load_model(os.getenv('FASTTEXT_MODEL'))
ft.get_dimension()


class FastText:
    def __init__(self):
        try:
            with mysql_object.get_connection() as conn:
                table_name = ['sms_tfc', 'sms_mgp']
                sql = f"SELECT * FROM {table_name[0]} UNION SELECT * FROM {table_name[1]};"
                self.current_intellignce = pd.read_sql(sql, conn.connection())
                logger.info(f'Launch system with {len(self.current_intellignce)} intellignce data from db.')
                test_data = self.current_intellignce['content'].tolist()
                self.intelligence_list = [ft.get_sentence_vector(str(row).strip().replace('\n', '')) for row in test_data]
        except Exception as e:
            self.current_intellignce = None
            logger.error(f'Launch system without intellignce data from db!')
        self.threshole = 0.8
        self.tfidf_thred = 0.6
        self.update_time = self.get_intelligence_update_times()
        print(self.update_time)
        pass


    def set_intelligence_update_times(self, table_name: str, timestamp: str):
        # 設定爬蟲情資更新時間
        message_hash = hashlib.sha256(f"intelligence_list".encode('utf-8')).hexdigest()
        logger.debug(f'Set intelligence list update time.')
        redis_object.conn.hset(message_hash, table_name, timestamp)

    def get_intelligence_update_times(self):
        # 取得爬蟲情資更新時間
        message_hash = hashlib.sha256("intelligence_list".encode('utf-8')).hexdigest()
        logger.debug(f'Get current intelligence list update time.')
        if redis_object.conn:
            timestamp_value = redis_object.conn.hget(message_hash, 'intelligence_list')
        else:
            timestamp_value = None
        if timestamp_value is not None:
            date_string = timestamp_value.decode('utf-8')
            timestamp_datetime = datetime.strptime(date_string, '%Y-%m-%d %H:%M')
            return timestamp_datetime
        else:
            return None

    def update_intelligence_list(self):
        current_update_time = self.get_intelligence_update_times()
        if current_update_time > self.update_time:
            try:
                with mysql_object.get_connection() as conn:
                    table_name = 'sms_factcheck'
                    sql = f"SELECT * FROM {table_name}"
                    self.current_intellignce = pd.read_sql(sql, conn.connection())
                    logger.debug(f'Update {len(self.current_intellignce)} intellignce data from db.')
                    test_data = self.current_intellignce['content'].tolist()
                    self.intelligence_list = [ft.get_sentence_vector(str(row).strip().replace('\n', '')) for row in test_data]
            except Exception as e:
                logger.error(f'Fail to update intellignce data from db!')
        else:
            logger.debug(f'Already got lastest intellignce data.')
            pass

    # @staticmethod
    # def generate_vector(sentence):
    #     return ft.get_sentence_vector(sentence)

    @staticmethod
    def cosine_sim(vector, matrix):
        dot_products = np.dot(vector, matrix.T)
        vector_norm = np.sqrt(np.sum(vector ** 2))
        matrix_norms = np.sqrt(np.sum(matrix ** 2, axis=1))
        matrix_norms[matrix_norms == 0] = 1  # Replace zeros with 1 to avoid division by zero
        matrix_norms = matrix_norms[np.newaxis, :]
        return (dot_products / (vector_norm * matrix_norms))[0]

    def fasttext(self, data):
        logger.info(f'Check fasttext with data - {data}')
        self.update_intelligence_list()
        df = self.current_intellignce
        target = ft.get_sentence_vector(data)
        score = self.cosine_sim(np.array([target]), np.array(self.intelligence_list))
        result_df = pd.DataFrame()
        df['similarity'] = score
        df['pred'] = 0
        df.loc[df['similarity'] >= self.threshole, 'pred'] = 1
        if (df['pred'] == 1).any():
            pass
        else:
            max_sim = df['similarity'].max()
            if max_sim > self.tfidf_thred:
                sentence1 = data
                sentence2 = df.loc[df['similarity'].idxmax(), 'content']
                # 預處理文本並創建 TF-IDF 向量
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([sentence1, sentence2])
                # 計算餘弦相似度
                similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
                similarity_score = similarity_matrix[0, 1]
                if similarity_score > 0:
                    df.loc[df['similarity'].idxmax(), 'pred'] = 1
        result = df.loc[df['pred'] == 1, 'content']
        result_df['text'] = result
        result_df['title'] = df.loc[df['pred'] == 1, 'title']
        result_df['tag'] = df.loc[df['pred'] == 1, 'tag']
        result_df['link'] = df.loc[df['pred'] == 1, 'link']
        result_df['summery'] = df.loc[df['pred'] == 1, 'summery']
        result_df['sourse'] = df.loc[df['pred'] == 1, 'sourse']
        result_df['similarity'] = df.loc[df['pred'] == 1, 'similarity']
        logger.info(f'Get fasttext result - {result_df.values}')
        if result_df['text'].empty:
            return None, None, None, None, None, None, None
        else:
            if len(result_df['text']) == 1:
                logger.info(f'Get fasttext result - {result_df["text"].tolist()[0]}:{result_df["similarity"].tolist()[0]}')
                return result_df['text'].tolist()[0], result_df['title'].tolist()[0], result_df['tag'].tolist()[0], result_df['link'].tolist()[0], result_df['summery'].tolist()[0], result_df['sourse'].tolist()[0], result_df['similarity'].tolist()[0]
            else:
                final_output = result_df.loc[result_df['similarity'].idxmax()]
                logger.info(f'Get fasttext result - {final_output} : {result_df.loc[result_df["similarity"].idxmax(), "similarity"]}')
                return final_output['text'], final_output['title'], final_output['tag'], final_output['link'], final_output['summery'], final_output['sourse'], final_output['similarity']


fasttext_object = FastText()
