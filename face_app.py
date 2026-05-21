import os
import base64
import io
import numpy as np
from flask import Flask, request, jsonify
import face_recognition
from PIL import Image

application = Flask(__name__)

def decode_base64_image(base64_string):
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image.convert('RGB'))

@application.route('/verify_face', methods=['POST'])
def verify_face():
    data = request.json
    if not data or 'captured_image' not in data or 'reference_image_path' not in data:
        return jsonify({"status": "error", "message": "Missing data"}), 400

    try:
        # 1. معالجة صورة الكاميرا الحالية
        captured_img = decode_base64_image(data['captured_image'])
        captured_encodings = face_recognition.face_encodings(captured_img)
        
        if len(captured_encodings) == 0:
            return jsonify({"status": "failed", "message": "لم يتم العثور على وجه في صورة الكاميرا"}), 200
        
        captured_encoding = captured_encodings[0]
        
        # 2. تحميل صورة الموظف المرجعية المستقبلة من PHP
        ref_image_path = data['reference_image_path']
        
        # ملاحظة للسيرفر السحابي: بما أن مسار الصورة قادم من سيرفر PHP الخاص بك، 
        # سنقوم بتحميل الصورة عن بعد أو معالجتها محلياً إذا تم تمريرها كـ Base64 مستقبلاً.
        # حالياً، الكود يتوقع وجود الملف في مسار فيزيائي يمكن للـ Python الوصول إليه.
        if not os.path.exists(ref_image_path):
             return jsonify({"status": "error", "message": "الصورة المرجعية غير موجودة في المسار المحدد"}), 200

        ref_image = face_recognition.load_image_file(ref_image_path)
        ref_encodings = face_recognition.face_encodings(ref_image)

        if len(ref_encodings) == 0:
            return jsonify({"status": "error", "message": "الصورة المرجعية للموظف لا تحتوي على وجه واضح"}), 200
        
        ref_encoding = ref_encodings[0]
        
        # 3. المقارنة الرياضية لمعالم الوجهين
        results = face_recognition.compare_faces([ref_encoding], captured_encoding, tolerance=0.5)
        
        if results[0]:
            return jsonify({"status": "success", "match": True})
        else:
            return jsonify({"status": "failed", "match": False, "message": "الوجه لا يطابق الموظف"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    application.run()
