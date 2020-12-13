#!/usr/bin/env python
from importlib import import_module
from flask import Flask, render_template, Response


app = Flask(__name__)

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/Dynamic_detection')
def index_dd():
    return render_template('index_dd.html')
@app.route('/video_feed_dd')
def video_feed_dd():
    Camera = import_module('Dynamic_detection').Camera
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/Face_recognition')
def index_fr():
    return render_template('index_fr.html')
@app.route('/video_feed_fr')
def video_feed_fr():
    Camera = import_module('Face_recognition').Camera
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
        
@app.route('/Danger_warning')
def index_dw():
    return render_template('index_dw.html')
@app.route('/video_feed_dw')
def video_feed_dw():
    Camera = import_module('Danger_warning').Camera
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
        
        
        
if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)