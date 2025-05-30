# -*- coding: utf-8 -*-
# This file is auto-generated, don't edit it. Thanks.
import os
import sys

from typing import List

from alibabacloud_tea_openapi.client import Client as OpenApiClient
from alibabacloud_credentials.client import Client as CredentialClient
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_openapi_util.client import Client as OpenApiUtilClient
from alibabacloud_darabonba_stream.client import Client as StreamClient
from alibabacloud_tea_util import models as util_models
from alibabacloud_tea_console.client import Client as ConsoleClient
from alibabacloud_tea_util.client import Client as UtilClient
from utils.utils import Utils


class Sample:
    def __init__(self):
        pass

    @staticmethod
    def create_client() -> OpenApiClient:
        """
        使用凭据初始化账号Client
        @return: Client
        @throws Exception
        """
        # 工程代码建议使用更安全的无AK方式，凭据配置方式请参见：https://help.aliyun.com/document_detail/378659.html。
        credential = CredentialClient()
        config = open_api_models.Config(
            credential=credential
        )
        # Endpoint 请参考 https://api.aliyun.com/product/ocr-api
        config.endpoint = f'ocr-api.cn-hangzhou.aliyuncs.com'
        return OpenApiClient(config)

    @staticmethod
    def create_api_info() -> open_api_models.Params:
        """
        API 相关
        @param path: string Path parameters
        @return: OpenApi.Params
        """
        params = open_api_models.Params(
            # 接口名称,
            action='RecognizeMultiLanguage',
            # 接口版本,
            version='2021-07-07',
            # 接口协议,
            protocol='HTTPS',
            # 接口 HTTP 方法,
            method='POST',
            auth_type='AK',
            style='V3',
            # 接口 PATH,
            pathname=f'/',
            # 接口请求体内容格式,
            req_body_type='json',
            # 接口响应体内容格式,
            body_type='json'
        )
        return params

    @staticmethod
    def main(args: List[str], image_path: str) -> None:
        client = Sample.create_client()
        params = Sample.create_api_info()
        # query params
        queries = {'Languages': OpenApiUtilClient.array_to_string_with_specified_style([
            'chn',
            'ja'
        ], 'Languages', 'simple'), 'OutputTable': True}
        # 需要安装额外的依赖库，直接点击下载完整工程即可看到所有依赖。
        body = StreamClient.read_from_file_path(image_path)
        # runtime options
        runtime = util_models.RuntimeOptions()
        request = open_api_models.OpenApiRequest(
            query=OpenApiUtilClient.query(queries),
            stream=body
        )
        # 复制代码运行请自行打印 API 的返回值
        # 返回值实际为 Map 类型，可从 Map 中获得三类数据：响应体 body、响应头 headers、HTTP 返回的状态码 statusCode。
        resp = client.call_api(params, request, runtime)
        if resp.get('statusCode') == 200:
            Utils.extract_resp(resp)
        else:
            print(resp.get('statusCode'))
        # ConsoleClient.log(UtilClient.to_jsonstring(resp))

    @staticmethod
    async def main_async(args: List[str],) -> None:
        client = Sample.create_client()
        params = Sample.create_api_info()
        # query params
        queries = {'Languages': OpenApiUtilClient.array_to_string_with_specified_style([
            'chn',
            'ja'
        ], 'Languages', 'simple'), 'OutputTable': True}
        # 需要安装额外的依赖库，直接点击下载完整工程即可看到所有依赖。
        body = StreamClient.read_from_file_path('<your-file-path>')
        # runtime options
        runtime = util_models.RuntimeOptions()
        request = open_api_models.OpenApiRequest(
            query=OpenApiUtilClient.query(queries),
            stream=body
        )
        # 复制代码运行请自行打印 API 的返回值
        # 返回值实际为 Map 类型，可从 Map 中获得三类数据：响应体 body、响应头 headers、HTTP 返回的状态码 statusCode。
        resp = await client.call_api_async(params, request, runtime)
        ConsoleClient.log(UtilClient.to_jsonstring(resp))


if __name__ == '__main__':
    # 替换为你的图片路径
    file_path = r'D:\Wrok\WebSiteProjects\BackEnd\QuizExtractor\pdf_pics\jp_n2_2016_page1.png'
    Sample.main(sys.argv[1:], file_path)

