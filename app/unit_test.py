import unittest
from unittest.mock import Mock
import asyncio
from main import *
from utils.text_similarity.fast_text import fasttext_object
from utils.text_similarity.fast_text_handler import fasttext_creator_object


query = FDRequestModel(
        QueryId='123e4567-e89b-12d3-a456-426655440000',
        QueryType='sms_fraud',
        QueryBody='夏威夷大火後，夏威夷獨立運動女領袖演講稱：我們不是美國人',
        QueryDateTime='2023-05-22T13:20:41.612Z',
        Sentry=SentryModel(
            TraceId='14b2e443e97e4e399e7d2b2c6a0eee2d',
            SpanId='aef45e9119e3a592',
            Op='task',
            Description='line bot task 2023-07-26 12:07'
        )
    )


class TestFastText(unittest.TestCase):
    def test_fasttext(self):
        expected_result = ('夏威夷大火後，夏威夷獨立運動女領袖演講稱：我們不是美國人，我們將以夏威夷人的身份去死！',
         '【錯誤】夏威夷大火後鬧獨立？女領袖發表演說否認自己是美國人？挪用多年前舊影片', '錯誤',
         'https://www.mygopen.com/2023/08/trask.html',
         "網傳「夏威夷獨立運動女領袖演講」的影片及訊息，內容聲稱夏威夷在 2023 年 8 月發生野火後，一名獨立運動女領袖發表演說，表示自己不是美國人，而是夏威夷人。經查證，傳言中的「獨立運動女領袖」是豪納妮凱．崔斯克（ Haunani Kay Trask ），她在 1993 年事逢夏威夷王國覆滅 100 周年之際，舉辦紀念遊行活動，並在檀香山「伊奧拉尼宮」（ 'Iolani Palace ）發表演說，並非近期發生的事，且崔斯克已於 2021 年 7 月過世，網傳訊息不正確。",
         'MyGoPen', 0.8372641205787659)
        self.result = fasttext_object.fasttext('夏威夷大火後，夏威夷獨立運動女領袖演講稱：我們不是美國人')
        self.assertEqual(self.result, expected_result)


class TestFastTextHandler(unittest.TestCase):
    def test_get_fasttext_response(self):
        text, title, tag, link, summery, sourse, similarity = fasttext_object.fasttext(
            '夏威夷大火後，夏威夷獨立運動女領袖演講稱：我們不是美國人')
        expected_result = (text, title, tag, link, summery, sourse, similarity)
        self.result = fasttext_creator_object.get_fasttext_response(query)
        body = self.result['ResponseBody']
        extracted_result = (body['Text'], body['Title'], body['Tag'], body['Link'], body['Summery'], body['Sourse'], body['Similarity'])
        self.assertEqual(extracted_result, expected_result)


class TestMain(unittest.TestCase):
    def test_start_fasttext(self):
        text, title, tag, link, summery, sourse, similarity = fasttext_object.fasttext('夏威夷大火後，夏威夷獨立運動女領袖演講稱：我們不是美國人，我們將以夏威夷人的身份去死！')
        current_time = datetime.now()
        response_data = {
            "QueryId":  "123e4567-e89b-12d3-a456-426655440000",
            "ResponseId": "9934e031-cd4d-4bed-8f04-529ef7db701d",
            "ResponseMessage": "Success",
            "ResponseBody": {
                "Text": text,
                "Title": title,
                "Tag": tag,
                "link": link,
                "Summery": summery,
                "Sourse": sourse,
                "Similarity": similarity
            },
            "ResponseDateTime": current_time
        }
        self.expected_result = {
            "QueryId":  "123e4567-e89b-12d3-a456-426655440000",
            "ResponseId": "9934e031-cd4d-4bed-8f04-529ef7db701d",
            "ResponseMessage": "Success",
            "ResponseBody": {
                "Text": "夏威夷大火後，夏威夷獨立運動女領袖演講稱：我們不是美國人，我們將以夏威夷人的身份去死！",
                "Title": "【錯誤】夏威夷大火後鬧獨立？女領袖發表演說否認自己是美國人？挪用多年前舊影片",
                "Tag": "錯誤",
                "link": "https://www.mygopen.com/2023/08/trask.html",
                "Summery": "網傳「夏威夷獨立運動女領袖演講」的影片及訊息，內容聲稱夏威夷在 2023 年 8 月發生野火後，一名獨立運動女領袖發表演說，表示自己不是美國人，而是夏威夷人。經查證，傳言中的「獨立運動女領袖」是豪納妮凱．崔斯克（ Haunani Kay Trask ），她在 1993 年事逢夏威夷王國覆滅 100 周年之際，舉辦紀念遊行活動，並在檀香山「伊奧拉尼宮」（ 'Iolani Palace ）發表演說，並非近期發生的事，且崔斯克已於 2021 年 7 月過世，網傳訊息不正確。",
                "Sourse": "MyGoPen",
                "Similarity": 1
            },
            "ResponseDateTime": current_time
        }
        mock_get_fasttext_response = Mock()
        mock_get_fasttext_response.return_value = response_data
        with unittest.mock.patch.object(fasttext_creator_object, 'get_fasttext_response', new=mock_get_fasttext_response):
            self.result = asyncio.run(start_fasttext(query))
        self.assertEqual(response_data, self.expected_result)

    def test_health_check(self):
        self.result = asyncio.run(health_check())
        self.assertEqual(self.result, 'Alive')


if __name__ == '__main__':
    test1 = unittest.TestLoader().loadTestsFromTestCase(TestMain)
    test2 = unittest.TestLoader().loadTestsFromTestCase(TestFastText)
    test3 = unittest.TestLoader().loadTestsFromTestCase(TestFastTextHandler)

    suite = unittest.TestSuite()
    suite.addTests(test1)
    suite.addTests(test2)
    suite.addTests(test3)
    unittest.TextTestRunner(verbosity=2).run(suite)