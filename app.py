import io
import json

from PIL import Image
from flask import Flask, jsonify, request,abort
from painter import Painter
from service_streamer import ThreadedStreamer
from io import BytesIO
from flask import send_file


app = Flask(__name__)
paint_streamer = None


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=90)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@app.route('/paint', methods=['POST'])
def painter():
    if request.method == 'POST':
        image = request.files['image']
        mask =  request.files['mask']
        img_bytes = image.read()
        mask_bytes=mask.read()
        batch={'image':img_bytes,'mask':mask_bytes}
        result = paint_streamer.predict([batch])[0]
        if result is None:
             return abort(404)
        else:
            return serve_pil_image(result)



if __name__ == '__main__':
    painter=Painter()

    paint_streamer = ThreadedStreamer(painter.predict, batch_size=64, max_latency=0.1)
    app.run(host="0.0.0.0",port=5005, debug=False)