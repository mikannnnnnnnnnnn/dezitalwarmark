import os
import io
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename
app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG' , 'bmp'])
IMAGE_WIDTH =512
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']
        uimg_file = request.files['uimg_file']

        # 変なファイル弾き
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
        elif uimg_file and allowed_file(uimg_file.filename):
            filename = secure_filename(uimg_file.filename)
        else:
            return ''' <p>許可されていない拡張子です</p> '''

        # BytesIOで読み込んでOpenCVで扱える型にする
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        uf = uimg_file.stream.read()
        ubin_data = io.BytesIO(uf)
        ufile_bytes = np.asarray(bytearray(ubin_data.read()), dtype=np.uint8)
        uimg = cv2.imdecode(ufile_bytes, cv2.IMREAD_COLOR)

        # 保存する
        raw_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_'+filename)
        raw_uimg_url = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_'+filename)
        cv2.imwrite(raw_img_url, img)
        cv2.imwrite(raw_uimg_url, uimg)

        # なにがしかの加工
        R=8
        h,w=img.shape[:2]
        F=np.zeros((h,w),dtype=np.float32)
        uh,uw=uimg.shape[:2]
        for c in range(3):
            f[:,:]=img[:,:,c]
            for y in range(0,h,R):
                for x in range(0,w,R):
                    F[y:y+R,x:x+R]=cv2.dct(f[y:y+R,x:x+R])
            img[:,:,c]=F[:,:,]
        
        # 加工したものを保存する
        gray_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'gray_'+filename)
        cv2.imwrite(gray_img_url, gray_img)

        return render_template('index.html', raw_img_url=raw_img_url, gray_img_url=gray_img_url)

    else:
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()