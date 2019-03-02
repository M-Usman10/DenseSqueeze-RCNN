from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)


from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
from detectron.core.construct_test_model import DensePoseModel

from detectron.core.test_texture import *
from detectron.core.test_tools import *
from flask import url_for, send_from_directory, request
from werkzeug import secure_filename
from caffe2.python import workspace
cv2.ocl.setUseOpenCL(False)


config = load_config(r'server/configs.yaml')
map_t = Texture(config)
app = make_flask_app(config)

def process_video(saved_path,video_name,flag=0):


    with Cap(saved_path,step_size=1) as cap:
        images = cap.read_all()

    iuvs=dpmodel.predict_iuvs(images)

    assert len(iuvs)==len(images),"Number of frames of IUV video and sent video not equal"

    if flag==0:
        result_filename="texture_result.mp4" if len(images)>1 else "texture_result.jpg"
        result_save_file = os.path.join(app.config['UPLOAD_FOLDER'],result_filename)
        out=map_t.transfer_texture_on_video(images,iuvs)
        if len(images)>1:
            print ("saving video at {}".format(result_save_file))
            save_video(out,result_save_file)
        else:
            cv2.imwrite(result_save_file,out[0])
        return result_filename

    else:
        result_filename="texture_result.jpg"
        result_save_file = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        map_t.extract_texture_from_video(images,iuvs,result_save_file)
        return result_filename


@app.route('/retreive_texture', methods = ['POST'])
def retreive_texture():
    print ("request for texture retreival received")
    app.logger.info(app.config['UPLOAD_FOLDER'])
    video = request.files['extract']
    video_name = secure_filename(video.filename)
    create_new_folder(app.config['UPLOAD_FOLDER'])
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
    app.logger.info("saving {}".format(saved_path))
    video.save(saved_path)
    filename=process_video(saved_path, video_name, flag=1)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/transfer_texture', methods = ['POST'])
def transfer_texture():
    print ("request for texture transfer received")
    app.logger.info(app.config['UPLOAD_FOLDER'])
    video = request.files['transfer']
    video_name = secure_filename(video.filename)
    print ("Video/image to process is {}".format(video_name))
    create_new_folder(app.config['UPLOAD_FOLDER'])
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], video_name)
    app.logger.info("saving {}".format(saved_path))
    video.save(saved_path)
    filename = process_video(saved_path, video_name)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

@app.route("/",methods=['POST','GET'])
def index_fn():
    return "hello world"




if __name__ == '__main__':
    # config = load_config(r'server/configs.yaml')
    # map_t = Texture(config)
    # app = make_flask_app(config)

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)

    logger = logging.getLogger(__name__)
    merge_cfg_from_file(config['model_config_file'])
    cfg.NUM_GPUS = 1

    weights = cache_url(config['weights'], cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)



    dpmodel=DensePoseModel(weights)

    app.run(host='0.0.0.0',port=9090, debug=False)
