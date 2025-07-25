import cv2
import numpy as np
import requests
from insightface.app import FaceAnalysis
from collections import Counter

# 初始化 InsightFace（只执行一次）
face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

def download_image_from_url(url):
    """
    从URL下载图像并转换为OpenCV格式
    """
    try:
        resp = requests.get(url, timeout=10)
        img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return None

def extract_main_face_embedding(image_urls):
    """
    接收一组图像URL，识别图像中出现频率最高的主脸，并提取其嵌入向量等信息。
    返回结构：
    {
        embedding: [...],
        bbox: [...],
        gender: 'Male' or 'Female',
        age: 25
    }
    """
    all_faces = []
    face_to_embedding_map = {}

    for url in image_urls:
        img = download_image_from_url(url)
        if img is None:
            continue

        faces = face_app.get(img)
        for face in faces:
            face_id = tuple(np.round(face.embedding[:5], 2))  # 用前5个维度构造face_id用于聚类
            all_faces.append(face_id)
            face_to_embedding_map[face_id] = face

    if not all_faces:
        print("❌ No faces found in any image.")
        return None

    # 统计哪个face_id最常出现，认为它是“主脸”
    main_face_id = Counter(all_faces).most_common(1)[0][0]
    main_face = face_to_embedding_map[main_face_id]

    return {
        "embedding": main_face.embedding.tolist(),  # 转为 JSON 兼容格式
        "bbox": main_face.bbox.tolist(),
        "gender": main_face.sex,
        "age": int(main_face.age)
    }
