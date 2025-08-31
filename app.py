import os
from flask_cors import CORS  # ⬅️ 1. flask-cors 불러오기
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import cv2 # OpenCV

app = Flask(__name__)
CORS(app)  # ⬅️ 2. CORS를 앱에 적용하여 모든 외부 요청을 허용

# 각 머리색에 따른 두피색의 RGB 범위 (이 값을 조정하여 민감도를 조절할 수 있습니다)
# 형식: (R 최소, R 최대, G 최소, G 최대, B 최소, B 최대)
SCALP_COLOR_RANGES = {
    'black':      (160, 255, 110, 225, 80, 205),
    'dark_brown': (160, 255, 110, 225, 80, 205),
    'light_brown':(160, 255, 110, 225, 80, 205),
    'dyed':       (160, 255, 110, 225, 80, 205),
}

# 점수에 따른 결과 등급 데이터
RESULT_DATA = {
    (0, 20): {"grade": "[1단계]\n🌳 모(毛)의 왕국 🌳", "image": "rank1.png", "quote": "걱정은 저 멀리 던져버리세요.", "comment": "당신의 두피는 평화로운 숲과 같습니다.\n지금처럼만 유지해주세요!"},
    (21, 40): {"grade": "[2단계]\n🌱 평화로운 잔디밭 🌱", "image": "rank2.png", "quote": "아직 늦지 않았습니다. 희망을 가지세요.", "comment": "조금 비어 보이는 곳이 있지만,충분히 관리 가능합니다.\n긍정적인 마음이 중요해요!"},
    (41, 60): {"grade": "[3단계]\n🚨 고속도로 착공 시작 🚨", "image": "rank3.png", "quote": "AI는... 거짓말을 하지 않습니다.", "comment": "이제는 관리가 필요한 시점입니다.\n현실을 직시하고 대책을 세워보는 건 어떨까요?"},
    (61, 80): {"grade": "[4단계]\n🌬️ 바람의 언덕 🌬️", "image": "rank4.png", "quote": "괜찮아요...\n머리카락이 인생의 전부는 아니잖아요?", "comment": "두피가 휑한 바람을 느끼고 있습니다.\n당신의 매력은 머리숱에만 있는 것이 아닙니다!"},
    (81, 100): {"grade": "[5단계]\n💡 무념무상(無念無想)의 경지 💡", "image": "rank5.png", "quote": "해탈의 경지에 오르셨군요.", "comment": "모든 것을 내려놓은 당신,\n그 어떤 것에도 흔들리지 않는 평온함을 얻었습니다."}
}

def analyze_image(image_file, hair_color):
    """
    OpenCV 형태학 연산을 추가하여 머리카락이 덮인 두피 영역까지 분석합니다.
    """
    try:
        # 1. 이미지를 불러오고 중앙을 잘라내는 과정은 이전과 동일합니다.
        img = Image.open(image_file.stream).convert('RGB')
        width, height = img.size
        left, top, right, bottom = width * 0.25, height * 0.25, width * 0.75, height * 0.75
        img_cropped = img.crop((left, top, right, bottom))
        
        # 2. Pillow 이미지를 OpenCV가 사용할 수 있는 NumPy 배열로 변환합니다.
        frame = np.array(img_cropped)
        # 색상 순서를 RGB에서 BGR로 변경 (OpenCV 기본값) - 여기서는 RGB로 직접 처리해도 무방합니다.
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 3. 색상 범위를 이용해 '두피'에 해당하는 부분만 흰색으로 표시하는 흑백 마스크를 생성합니다.
        color_range = SCALP_COLOR_RANGES.get(hair_color, SCALP_COLOR_RANGES['black'])
        lower_bound = np.array([color_range[0], color_range[2], color_range[4]])
        upper_bound = np.array([color_range[1], color_range[3], color_range[5]])
        mask = cv2.inRange(frame, lower_bound, upper_bound)
        
        # 4. (핵심!) Morphology 연산으로 마스크의 노이즈를 제거하고 구멍을 메웁니다.
        # kernel은 연산을 적용할 영역의 크기를 정의합니다. '돋보기'와 비슷합니다.
        kernel = np.ones((5, 5), np.uint8)
        # MORPH_CLOSE 연산은 흰색 영역 안의 작은 검은색 구멍을 메워주는 역할을 합니다.
        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # 5. 후처리된 최종 마스크에서 두피(흰색 픽셀)의 개수를 계산합니다.
        scalp_pixel_count = cv2.countNonZero(closed_mask)
        total_pixels = closed_mask.shape[0] * closed_mask.shape[1]
        
        # 6. 점수 계산 로직은 이전과 동일합니다.
        score = min(100, int((scalp_pixel_count / total_pixels) * 100))
        
        return score
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

def get_result_by_score(score):
    """점수에 맞는 결과 데이터를 반환합니다."""
    for (min_score, max_score), data in RESULT_DATA.items():
        if min_score <= score <= max_score:
            return data
    return RESULT_DATA[(51, 100)] # 혹시 모를 예외 처리

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """사진 업로드를 처리하고 분석 결과를 JSON으로 반환합니다."""
    if 'photo' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['photo']
    hair_color = request.form.get('hair_color')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file and hair_color:
        score = analyze_image(file, hair_color)
        
        if score is not None:
            result_data = get_result_by_score(score)
            return jsonify({
                "score": score,
                "grade": result_data["grade"],
                "image_url": f"/static/img/{result_data['image']}",
                "quote": result_data["quote"],
                "comment": result_data["comment"]
            })
        else:
            return jsonify({"error": "Image analysis failed"}), 500

    return jsonify({"error": "Invalid request"}), 400

# if __name__ == '__main__':
#     # 'host=0.0.0.0'을 추가하면 같은 와이파이에 연결된 다른 기기(예: 스마트폰)에서도 접속 테스트가 가능합니다.
#     app.run(debug=True, host='0.0.0.0')