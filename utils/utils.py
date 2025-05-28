import json
import os


class Utils:
    """
    工具类
    extract_resp:用于解析阿里云ocr识别的内容并存储到out_put.txt
    """

    # 识别内容并保存至临时文件中
    @staticmethod
    def extract_resp(resp):
        out_put_path = '../../output'
        # 创建路径
        if not os.path.exists(out_put_path):
            os.makedirs(out_put_path)

        try:
            # Step 1: 获取嵌套 JSON 字符串
            data_str = resp.get("body", {}).get("Data")
            if not data_str:
                print("❌ 没有找到 Data 字段。")
                return

            # Step 2: 反序列化
            data_json = json.loads(data_str)
            content = data_json.get("content")

            # Step 3: 写入文件
            if content:
                print(content)
                with open(out_put_path + '/ocr_text.txt', "a", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ 内容已保存至：{out_put_path}/ocr_text.txt")
            else:
                print("⚠️ content 字段不存在或内容为空。")

        except Exception as e:
            print("❌ 处理失败：", e)
