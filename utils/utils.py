import json
import os
import fitz
from PIL import Image


class Utils:
    """
    工具类
    extract_resp: 用于解析阿里云ocr识别的内容并存储到out_put.txt,
    pdf2image: 用于将pdf文件转换为png格式并存储到指定文件夹中
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
                print(content[:300] + '...')
                with open(out_put_path+'/ocr_text.txt', "a", encoding="utf-8") as f:
                    f.write(content)
                print(f"✅ 内容已保存至：{out_put_path}/ocr_text.txt")
            else:
                print("⚠️ content 字段不存在或内容为空。")

        except Exception as e:
            print("❌ 处理失败：", e)

    @staticmethod
    def pdf2img(pdf_path, pic_folder):
        dpi = 150   # 图片分辨率（DPI），建议300, 阿里云接口要求图像不能过大，修改为150
        os.makedirs(pic_folder, exist_ok=True)  # 创建输出文件夹
        doc = fitz.open(pdf_path)   # 构建对象
        base_name = os.path.splitext(pdf_path)[0].split('/')[-1]   # 获取PDF名称
        print(f"📄 正在处理：{base_name} (共 {len(doc)} 页)")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # 保存图片
            pic_folder = os.path.relpath(pic_folder)
            image_name = f"{base_name}_page{page_num + 1}.png"
            image_path = os.path.join(pic_folder, image_name)
            image.save(image_path, "PNG")
            print(f" ✅ 已保存：{pic_folder + '\\' + image_name}")
        print("\n 所有PDF已转换为PNG图片！输出目录：", pic_folder)


if __name__ == "__main__":
    file_path = "../pdf_pics/jp_n2_2016.pdf"
    Utils.pdf2img(file_path, '../pdf_pics')
