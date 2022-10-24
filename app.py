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
        else:
            return ''' <p>許可されていない拡張子です</p> '''
        if uimg_file and allowed_file(uimg_file.filename):
            ufilename = secure_filename(uimg_file.filename)
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
        raw_uimg_url = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_'+ufilename)
        cv2.imwrite(raw_img_url, img)
        cv2.imwrite(raw_uimg_url, uimg)

        # 電子透かしを埋め込む
        pic = cv2.imread(f'{raw_img_url}')
        h,w=pic.shape[:2]
        ume = cv2.imread(f'{raw_uimg_url}')
        uh,uw=ume.shape[:2]
        R=8
        quan=np.zeros((uh,uw*4,3),dtype=np.float32)
        qc=np.zeros((8,8),dtype=np.float32)
        F=np.zeros((h,w),dtype=np.float32)
        DCTF=np.zeros((h,w),dtype=np.float32)
        uF=np.zeros((h,w),dtype=np.float32)
        qc[:,:]=np.loadtxt(f'quancsv/low.csv',delimiter=',',dtype=np.uint8)
        for c in range(3):
            for i in range(uh):
                for j in range(uw):
                    quan[i,4*j+3,c]=ume[i,j,c]%10
                    syo=ume[i,j,c]//10
                    sinsu = divmod(syo,3)
                    for r in range(2,-1,-1):
                        match sinsu[1]:
                            case 0:
                                quan[i,4*j+r,c]=-3
                            case 1:
                                quan[i,4*j+r,c]=0
                            case 2:
                                quan[i,4*j+r,c]=3
                        sinsu = divmod(sinsu[0],3)
            F[:,:]=pic[:,:,c]
            for y in range(0,h,R):
                for x in range(0,w,R):
                    DCTF[y:y+R,x:x+R]=cv2.dct(F[y:y+R,x:x+R])
            a=b=0
            for y in range(0,h,R):
                for x in range(0,w,R):
                    for u in range(R):
                        for v in range(R):
                            if qc[u,v]==1:
                                DCTF[y+u,x+v] += quan[a,b,c]
                                b+=1
                            if b>=uw*4:
                                b=0
                                a+=1
                    uF[y:y+R,x:x+R]=cv2.idct(DCTF[y:y+R,x:x+R])
            for y in range(h):
                for x in range(w):
                    if uF[y,x]<0:
                        uF[y,x]=0
                    elif uF[y,x]>255:
                        uF[y,x]=255
            pic[:,:,c]=uF[:,:]
        
        # 加工したものを保存する
        wate_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'watermark_'+filename)
        cv2.imwrite(wate_img_url, pic)

        return render_template('index.html', raw_img_url=raw_img_url, raw_uimg_url=raw_uimg_url, wate_img_url=wate_img_url)

    else:
        return redirect(url_for('index'))

@app.route('/take', methods=['GET', 'POST'])
def take():
    if request.method == 'POST':
        pic_file = request.files['pic_file']
        upic_file = request.files['upic_file']

        # 変なファイル弾き
        if pic_file and allowed_file(pic_file.filename):
            filename = secure_filename(pic_file.filename)
        else:
            return ''' <p>許可されていない拡張子です</p> '''
        if upic_file and allowed_file(upic_file.filename):
            ufilename = secure_filename(upic_file.filename)
        else:
            return ''' <p>許可されていない拡張子です</p> '''

        # BytesIOで読み込んでOpenCVで扱える型にする
        f = pic_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        uf = upic_file.stream.read()
        ubin_data = io.BytesIO(uf)
        ufile_bytes = np.asarray(bytearray(ubin_data.read()), dtype=np.uint8)
        uimg = cv2.imdecode(ufile_bytes, cv2.IMREAD_COLOR)

        # 保存する
        take_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'take_'+filename)
        take_uimg_url = os.path.join(app.config['UPLOAD_FOLDER'], 'take_'+ufilename)
        cv2.imwrite(take_img_url, img)
        cv2.imwrite(take_uimg_url, uimg)

        # 電子透かしを抽出
        pic = cv2.imread(f'{take_img_url}')
        h,w=pic.shape[:2]
        upic = cv2.imread(f'{take_uimg_url}')
        R=8
        uh=uw=128
        ume=np.zeros((uh,uw,3),dtype=np.float32)
        qc=np.zeros((8,8),dtype=np.float32)
        qc[:,:]=np.loadtxt(f'quancsv/low.csv',delimiter=',',dtype=np.uint8)
        F=np.zeros((h,w),dtype=np.float32)
        uF=np.zeros((h,w),dtype=np.float32)
        DCTF=np.zeros((h,w),dtype=np.float32)
        DCTuF=np.zeros((h,w),dtype=np.float32)
        quan=np.zeros((uh,uw*4,3),dtype=np.float32)

        for c in range(3):
            a=b=0
            F[:,:]=pic[:,:,c]
            uF[:,:]=upic[:,:,c]
            for y in range(0,h,R):
                for x in range(0,w,R):
                    DCTF[y:y+R,x:x+R]=cv2.dct(F[y:y+R,x:x+R])
                    DCTuF[y:y+R,x:x+R]=cv2.dct(uF[y:y+R,x:x+R])
                    for u in range(R):
                        for v in range(R):
                            if qc[u,v]==1:
                                quan[a,b,c]=DCTuF[y+u,x+v]-DCTF[y+u,x+v]
                                b+=1
                            if b==uw*4:
                                b=0
                                a+=1
            for i in range(uh):
                for j in range(uw):
                    ume[i,j,c]=quan[i,4*j+3,c]
                    if  -1<= quan[i,4*j+2,c] <= 1:
                        ume[i,j,c]+=10
                    elif 2 <= quan[i,4*j+2,c]:
                        ume[i,j,c]+=20
                    if -1 <= quan[i,4*j+1,c] <= 1:
                        ume[i,j,c]+=30
                    elif 2 <= quan[i,4*j+1,c]:
                        ume[i,j,c]+=60
                    if -1 <= quan[i,4*j,c] <= 1:
                        ume[i,j,c]+=90
                    elif 2 <= quan[i,4*j,c]:
                        ume[i,j,c]+=180

        # 加工したものを保存する
        ume_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'ume_'+filename)
        cv2.imwrite(ume_img_url, ume)

        return render_template('index.html', take_img_url=take_img_url, take_uimg_url=take_uimg_url, ume_img_url=ume_img_url)

    else:
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()