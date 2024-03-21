# E3PO, an open platform for 360˚ video streaming simulation and evaluation.
# Copyright 2023 ByteDance Ltd. and/or its affiliates
#
# This file is part of E3PO.
#
# E3PO is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# E3PO is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see:
#    <https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html>
import copy
import glob
import os
import os.path as osp
import re

import cv2
import numpy as np
import shutil
import yaml
import math
import torch
from sklearn.linear_model import Ridge
from e3po.approaches.Bingo.SR_models import SRCNN
from e3po.approaches.Bingo.SR_utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb
from e3po import get_logger
from e3po.utils.data_utilities import remove_temp_files, transcode_video, resize_video
from e3po.utils.decision_utilities import generate_dl_list
from e3po.utils.json import get_video_json_size
#from e3po.utils.projection_utilities import fov_to_3d_polar_coord, _3d_polar_coord_to_pixel_coord, \
    #pixel_coord_to_tile, pixel_coord_to_relative_tile_coord


def video_analysis(user_data, video_info):
    """
    This API allows users to analyze the full 360 video (if necessary) before the pre-processing starts.
    Parameters
    ----------
    user_data: is initially set to an empy object and users can change it to any structure they need.
    video_info: is a dictionary containing the required video information.

    Returns
    -------
    user_data:
        user should return the modified (or unmodified) user_data as the return value.
        Failing to do so will result in the loss of the information stored in the user_data object.
    """

    user_data = user_data or {}
    user_data["video_analysis"] = []

    return user_data


def init_user(user_data, video_info, dst_video_folder=None, pre_flag=False, eva_flag=False):
    """
    Initialization function, users initialize their parameters based on the content passed by E3PO

    Parameters
    ----------
    eva_flag: 如果是evaluation则user data中会有部分与evaluation相关的信息
    dst_video_folder:存储tile的文件夹，会在preprocess_video/生成结果图过程中用到
    pre_flag: 如果是preprocess则user data中会有部分与pre相关的信息
    user_data: None
        the initialized user_data is none, where user can store their parameters
    video_info: dict
        video information of original video, user can perform preprocessing according to their requirement

    Returns
    -------
    user_data: dict
        the updated user_data
    """

    user_data = user_data or {}
    user_data["video_info"] = video_info
    user_data["config_params"] = read_config()
    user_data["chunk_idx"] = -1
    user_data["encoding_params"] = set_encoding_params()    # add
    user_data['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if pre_flag:
        # 创建相关文件夹
        # 创建存储转换投影格式后的tile图片的文件夹，即后续mse比较时的原图
        projection_org_folder = osp.join(
            os.path.abspath(os.path.dirname(dst_video_folder)), 'projection_org_folder')
        try:
            if os.path.exists(projection_org_folder) and os.path.isdir(projection_org_folder):
                shutil.rmtree(projection_org_folder)
            os.makedirs(projection_org_folder, exist_ok=True)
        except Exception as e:
            print(f"An error occurred while deleting the folder {projection_org_folder}: {e}")

        # 创建存储不同分辨率视频的文件夹
        resize_video_folder_1080p = osp.join(
            os.path.abspath(os.path.dirname(dst_video_folder)), 'resize_video_folder_1080p')
        try:
            if os.path.exists(resize_video_folder_1080p) and os.path.isdir(resize_video_folder_1080p):
                shutil.rmtree(resize_video_folder_1080p)
            os.makedirs(resize_video_folder_1080p, exist_ok=True)
        except Exception as e:
            print(f"An error occurred while deleting the folder {resize_video_folder_1080p}: {e}")

        user_data['resize_video_folder_1080p'] = resize_video_folder_1080p
        user_data['projection_org_folder'] = projection_org_folder

    if pre_flag or eva_flag:    # 加载超分模型
        device = user_data['device']

        model1080p_path = osp.join(
            os.path.abspath(os.path.dirname(os.path.abspath(__file__))), 'SR_1080p28k.pth')
        model1080p = SRCNN().to(device)
        state1080p_dict = model1080p.state_dict()
        for n, p in torch.load(model1080p_path, map_location=lambda storage, loc: storage).items():
            if n in state1080p_dict.keys():
                state1080p_dict[n].copy_(p)
            else:
                raise KeyError(n)

        user_data['model1080p'] = model1080p

    return user_data


def read_config():
    """
    read the user-customized configuration file as needed

    Returns
    -------
    config_params: dict
        the corresponding config parameters
    """

    config_path = os.path.dirname(os.path.abspath(__file__)) + "/Bingo.yml"
    with open(config_path, 'r', encoding='UTF-8') as f:
        opt = yaml.safe_load(f.read())['approach_settings']

    background_flag = opt['background']['background_flag']
    converted_height = opt['video']['converted']['height']
    converted_width = opt['video']['converted']['width']
    background_height = opt['background']['height']
    background_width = opt['background']['width']
    tile_height_num = opt['video']['tile_height_num']
    tile_width_num = opt['video']['tile_width_num']
    total_tile_num = tile_height_num * tile_width_num
    tile_width = int(opt['video']['converted']['width'] / tile_width_num)
    tile_height = int(opt['video']['converted']['height'] / tile_height_num)
    if background_flag:
        background_info = {
            "width": opt['background']['width'],
            "height": opt['background']['height'],
            "background_projection_mode": opt['background']['projection_mode']
        }
    else:
        background_info = {}

    motion_history_size = opt['video']['hw_size'] * 1000
    motino_prediction_size = opt['video']['pw_size']
    ffmpeg_settings = opt['ffmpeg']
    if not ffmpeg_settings['ffmpeg_path']:
        assert shutil.which('ffmpeg'), '[error] ffmpeg doesn\'t exist'
        ffmpeg_settings['ffmpeg_path'] = shutil.which('ffmpeg')
    else:
        assert os.path.exists(ffmpeg_settings['ffmpeg_path']), \
            f'[error] {ffmpeg_settings["ffmpeg_path"]} doesn\'t exist'
    projection_mode = opt['approach']['projection_mode']
    converted_projection_mode = opt['video']['converted']['projection_mode']

    config_params = {
        "background_flag": background_flag,
        "converted_height": converted_height,
        "converted_width": converted_width,
        "background_height": background_height,
        "background_width": background_width,
        "tile_height_num": tile_height_num,
        "tile_width_num": tile_width_num,
        "total_tile_num": total_tile_num,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "org_tile_width": tile_width,
        "org_tile_height": tile_height,
        "background_info": background_info,
        "motion_history_size": motion_history_size,
        "motion_prediction_size": motino_prediction_size,
        "ffmpeg_settings": ffmpeg_settings,
        "projection_mode": projection_mode,
        "converted_projection_mode": converted_projection_mode
    }

    return config_params


def calculate_tile_mse(tile_pics, user_data, resize_resolution):    # add
    '''
    :param tile_pics: tile的图片
    :param user_data:
    :param resize_resolution: 分辨率index：'converted', '4k', '2k', '1080p'
    :function: 根据切好的tile pics进行编码，获得编码后的psnr，存储到user_data的tile_info中，删除临时编码文件
    :return: tile_psnr: 当前tile的psnr
    '''

    result_video_name = 'temp.mp4'
    dst_video_uri = osp.join(tile_pics, result_video_name)    # 存储编码后的视频
    source_pics_path = os.path.join(tile_pics, f"%d.png")    # 获取切完的tile图像
    cmd = f"{user_data['config_params']['ffmpeg_settings']['ffmpeg_path']} " \
          f"-r {user_data['encoding_params']['video_fps']} " \
          f"-start_number 0 " \
          f"-i {source_pics_path} " \
          f"-threads {user_data['config_params']['ffmpeg_settings']['thread']} " \
          f"-preset {user_data['encoding_params']['preset']} " \
          f"-c:v {user_data['encoding_params']['encoder']} " \
          f"-g {user_data['encoding_params']['gop']} " \
          f"-bf {user_data['encoding_params']['bf']} " \
          f"-qp {user_data['encoding_params']['qp_list'][0]} " \
          f"-y {dst_video_uri} " \
          f"-loglevel {user_data['config_params']['ffmpeg_settings']['loglevel']} "
    os.system(cmd)

    temp_idx = (user_data['tile_idx']-1) % 72
    width = user_data['config_params']['org_tile_width']
    height = user_data['config_params']['org_tile_height']
    device = user_data['device']

    # 加载不同分辨率的模型
    if resize_resolution == '1080p':
        model = user_data['model1080p']
        model.eval()

    projection_org_folder = user_data['projection_org_folder']
    projection_org_folder_tileidx = osp.join(
        os.path.abspath(projection_org_folder), 'tile_' + str(temp_idx))
    pro_tile_image_list = glob.glob(projection_org_folder_tileidx + '/*.png')   # 获取转换完投影格式后的图片
    pro_tile_image_list.sort(key=lambda l: int(re.findall('\d+', l)[-1]))  # 找出字符串中的数字并依据其整形进行排序
    index = 0
    video_temp = cv2.VideoCapture(dst_video_uri)   # 读取编码后的视频
    frame_count = video_temp.get(cv2.CAP_PROP_FRAME_COUNT)
    total_mse = 0
    total_sr_mse = 0
    if video_temp.isOpened():
        while True:
            ret, img_after_codec = video_temp.read()  # img 就是一帧图片
            if not ret:
                break  # 当获取完最后一帧就结束
            pro_tile_image_path = pro_tile_image_list[index]    # 获取原图，即转换完投影格式后的图片
            pro_tile_image = torch.from_numpy(cv2.imread(pro_tile_image_path))
            if resize_resolution != 'converted':  # 如果做了resize，则需要超分
                inter_img = cv2.resize(img_after_codec, (width, height), interpolation=cv2.INTER_LINEAR)
                img3_ = torch.from_numpy(inter_img).to(dtype=torch.float64, device=device)
                # 超分
                inter_img = np.array(inter_img).astype(np.float32)
                ycbcr = convert_rgb_to_ycbcr(inter_img)
                y = ycbcr[..., 0]
                y /= 255.
                y = torch.from_numpy(y).to(device)
                y = y.unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    preds = model(y).clamp(0.0, 1.0)    # 超分后的结果
                preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
                output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
                sr_output = torch.from_numpy(output)
                img1_ = pro_tile_image.to(dtype=torch.float64, device=device)    # 原图
                img2_ = sr_output.to(dtype=torch.float64, device=device)    # 超分
                sr_mse = torch.mean((img1_ - img2_) ** 2, dim=[0, 1, 2])      # 超分后mse
                mse = torch.mean((img1_ - img3_) ** 2, dim=[0, 1, 2])      # 插值mse
            else:     # 如果没有resize则不需要插值，直接计算mse
                img_after_codec = torch.from_numpy(img_after_codec)
                img1_ = pro_tile_image.to(dtype=torch.float64, device=device)
                img2_ = img_after_codec.to(dtype=torch.float64, device=device)
                mse = torch.mean((img1_ - img2_) ** 2, dim=[0, 1, 2])
                sr_mse = mse      # 原分辨率没有SR
            index += 1
            total_mse += float(mse)
            total_sr_mse += float(sr_mse)
    else:
        print('video open fail!')

    video_temp.release()
    os.remove(dst_video_uri)   # 删除临时存储的视频

    tile_mse = total_mse / frame_count
    tile_mse = round(tile_mse + 0.0001, 3)    # 保留三位小数

    # 超分后的mse
    tile_sr_mse = total_sr_mse / frame_count
    tile_sr_mse = round(tile_sr_mse + 0.0001, 3)

    return tile_sr_mse, tile_mse

def segment_video(ffmpeg_settings, source_video_uri, dst_video_folder, segmentation_info,
                  tile_info, user_data, org_flag=False):
    f"""
    Segment video tile from the original video

    Parameters
    ----------
    user_data
    org_flag:指示是否是原分辨率
    tile_info
    ffmpeg_settings: dict
        ffmpeg related information
    source_video_uri: str
        resize video uri of original video, store the resize video as a picture
    dst_video_folder: str
        folder path of the segmented video tile
    segmentation_info: dict
        tile information

    Returns
    -------
        None
    """

    out_w = segmentation_info['segment_out_info']['width']
    out_h = segmentation_info['segment_out_info']['height']
    start_w = segmentation_info['start_position']['width']
    start_h = segmentation_info['start_position']['height']

    result_frame_path = osp.join(dst_video_folder, f"%d.png")

    if org_flag:    # 如果是原分辨率
        cmd = f"{ffmpeg_settings['ffmpeg_path']} " \
              f"-i {source_video_uri} " \
              f"-threads {ffmpeg_settings['thread']} " \
              f"-vf \"crop={out_w}:{out_h}:{start_w}:{start_h}\" " \
              f"-q:v 2 -f image2 {result_frame_path} " \
              f"-loglevel {ffmpeg_settings['loglevel']}"

        os.system(cmd)

        tile_idx = tile_info['tile_idx']
        projection_org_folder = user_data['projection_org_folder']
        # 在projection_org_folder中存储原视频的tile图像便于后续计算mse
        projection_org_folder_tileidx = osp.join(
            os.path.abspath(projection_org_folder), 'tile_' + str(tile_idx))
        if os.path.exists(projection_org_folder_tileidx) and os.path.isdir(projection_org_folder_tileidx):
            shutil.rmtree(projection_org_folder_tileidx)
        # 将切好的转换格式的图像复制到projection_org_folder_tileidx文件夹中
        shutil.copytree(dst_video_folder, projection_org_folder_tileidx,
                        ignore=shutil.ignore_patterns("*.mp4", "*.h264",  "*.264"))

    else:    # 其他分辨率
        pic_path = source_video_uri + '/%d.png'  # 获取转换投影格式/转码分辨率后的图片
        cmd = f"{ffmpeg_settings['ffmpeg_path']} " \
              f"-i {pic_path} " \
              f"-threads {ffmpeg_settings['thread']} " \
              f"-vf \"crop={out_w}:{out_h}:{start_w}:{start_h}\" " \
              f"-q:v 2 -f image2 {result_frame_path} " \
              f"-loglevel {ffmpeg_settings['loglevel']}"

        os.system(cmd)





def preprocess_video(source_video_uri, dst_video_folder, chunk_info, user_data, video_info):
    """
    Self defined preprocessing strategy

    Parameters
    ----------
    source_video_uri: str
        the video uri of source video
    dst_video_folder: str
        the folder to store processed video
    chunk_info: dict
        chunk information
    user_data: dict
        store user-related parameters along with their required content
    video_info: dict
        store video information

    Returns
    -------
    user_video_spec: dict
        a dictionary storing user specific information for the preprocessed video
    user_data: dict
        updated user_data
    """

    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info, dst_video_folder=dst_video_folder, pre_flag=True)

    config_params = user_data['config_params']
    video_info = user_data['video_info']

    # 获取存储不同分辨率视频的文件夹
    resize_video_folder_1080p = user_data['resize_video_folder_1080p']

    # 存储转码分辨率信息的词典   add
    resize_dic = {
        "1080p": {"width": 1920, "height": 1080}
    }

    # update related information
    if user_data['chunk_idx'] == -1:
        user_data['chunk_idx'] = chunk_info['chunk_idx']
        user_data['tile_idx'] = 0
        user_data['transcode_video_uri'] = source_video_uri
    else:
        if user_data['chunk_idx'] != chunk_info['chunk_idx']:
            user_data['chunk_idx'] = chunk_info['chunk_idx']
            user_data['tile_idx'] = 0
            user_data['transcode_video_uri'] = source_video_uri

    # transcoding
    src_projection = video_info['projection']
    dst_projection = config_params['converted_projection_mode']
    if src_projection != dst_projection and user_data['tile_idx'] == 0:
        src_resolution = [video_info['height'],video_info['width']]
        dst_resolution = [config_params['converted_height'], config_params['converted_width']]
        user_data['transcode_video_uri'] = transcode_video(
            source_video_uri, src_projection, dst_projection, src_resolution, dst_resolution,
            dst_video_folder, chunk_info, config_params['ffmpeg_settings']
        )
    else:
        pass
    transcode_video_uri = user_data['transcode_video_uri']

    # 转换不同分辨率
    if len(os.listdir(resize_video_folder_1080p)) == 0:
        resize_video(config_params['ffmpeg_settings'], transcode_video_uri, resize_video_folder_1080p, resize_dic['1080p'])

    # 根据不同分辨率设定tile的width和height
    if user_data['tile_idx'] == 0:  # 如果是第0个tile，则是原始分辨率的大小，此时tile的宽高不需要改
        user_data['config_params']['tile_width'] = user_data['config_params']['org_tile_width']
        user_data['config_params']['tile_height'] = user_data['config_params']['org_tile_height']
    elif user_data['tile_idx'] == 216:  # 如果是第216个tile，则是1080p分辨率的大小，需要更新tile的宽高
        user_data['config_params']['tile_width'] = \
            int(resize_dic['1080p']['width'] / user_data['config_params']['tile_width_num'])
        user_data['config_params']['tile_height'] = \
            int(resize_dic['1080p']['height'] / user_data['config_params']['tile_height_num'])
    # end add

    # segmentation
    if user_data['tile_idx'] < config_params['total_tile_num'] * 1:
        tile_info, segment_info = tile_segment_info(chunk_info, user_data)
        segment_video(user_data["config_params"]['ffmpeg_settings'], transcode_video_uri, dst_video_folder,
                      segment_info, tile_info, user_data, org_flag=True)
        user_data['tile_idx'] += 1
        tile_sr_mse, tile_mse = calculate_tile_mse(dst_video_folder, user_data, 'converted')  # 计算mse
        tile_info['INTER_mse'] = tile_mse
        tile_info['SR_mse'] = tile_sr_mse
        user_video_spec = {'segment_info': segment_info, 'tile_info': tile_info}

        if user_data['tile_idx'] == 72:     # 改tile index
            user_data['tile_idx'] = 216

    elif user_data['tile_idx'] < config_params['total_tile_num'] * 4:
        tile_info, segment_info = tile_segment_info(chunk_info, user_data)
        segment_video(user_data["config_params"]['ffmpeg_settings'], resize_video_folder_1080p, dst_video_folder,
                      segment_info, tile_info, user_data)
        user_data['tile_idx'] += 1
        tile_sr_mse, tile_mse = calculate_tile_mse(dst_video_folder, user_data, '1080p')
        tile_info['INTER_mse'] = tile_mse
        tile_info['SR_mse'] = tile_sr_mse
        user_video_spec = {'segment_info': segment_info, 'tile_info': tile_info}

    else:
        projection_org_folder = user_data['projection_org_folder']
        remove_temp_files(projection_org_folder)  # 更换chunk后删除掉上个chunk的所有原图
        remove_temp_files(resize_video_folder_1080p)
        remove_temp_files(user_data['transcode_video_uri'])
        user_video_spec = None

    return user_video_spec, user_data



def predict_motion_tile(motion_history, motion_history_size, motion_prediction_size, segmentid, chunk_duration, min_re_download, is_last_prediction):
    """
    Predicting motion with given historical information and prediction window size.
    (As an example, users can implement their customized function.)

    Parameters
    ----------
    motion_history: dict
        a dictionary recording the historical motion, with the following format:

    motion_history_size: int
        the size of motion history to be used for predicting2000
    motion_prediction_size: int
        the size of motion to be predicted1

    Returns
    -------
    list
        The predicted record list, which sequentially store the predicted motion of the future pw chunks.
         Each motion dictionary is stored in the following format:
            {'yaw ': yaw,' pitch ': pitch,' scale ': scale}
    """
    # Use exponential smoothing to predict the angle of each motion within pw for yaw and pitch.
    hw = [d['motion_record'] for d in motion_history]
    time = [d['motion_ts'] for d in motion_history]
    if len(time) < 3:
        motion_frequency = 1000
    else:
        motion_frequency = 1000 / (time[-1] - time[-2])
    first_time = segmentid * chunk_duration * 1000  # ms
    end_time = (segmentid + 1) * chunk_duration * 1000  # ms
    if is_last_prediction == 0:
        predict_count = 2
    else:
        predict_count = int((end_time - time[-1]) / 40) + 1
    for i in range(predict_count):
        if i == 0:
            if first_time <= time[-1]:
                first_time = time[-1] + min_re_download
                predict_size = min_re_download
            else:
                if time[-1] + min_re_download > first_time:
                    predict_size = min_re_download
                    first_time = time[-1] + min_re_download
                else:
                    predict_size = first_time - time[-1]
        else:
            predict_size = first_time - time[-1]
        history_size = 0.5 * predict_size
        motion = hw[-int(history_size / (1000 / motion_frequency)):]
        hw_yaw = [motion[0]['yaw']]
        hw_pitch = [motion[0]['pitch']]
        for i in range(len(motion)-1):
            if motion[i + 1]['yaw'] - hw_yaw[-1] > math.pi:
                motion[i + 1]['yaw'] -= 2 * math.pi
            if motion[i + 1]['yaw'] - hw_yaw[-1] < -math.pi:
                motion[i + 1]['yaw'] += 2 * math.pi
            hw_yaw.append(motion[i + 1]['yaw'])
            hw_pitch.append(motion[i + 1]['pitch'])
        play_time = time[-int(history_size / (1000 / motion_frequency)):]
        play_time = np.array(play_time).reshape(len(play_time), 1)
        hw_yaw = np.array(hw_yaw).reshape(len(hw_yaw), )
        hw_pitch = np.array(hw_pitch).reshape(len(hw_pitch), )
        model_yaw, model_pitch = build_model(play_time, hw_yaw, hw_pitch)
        predicted_record = []
        if first_time < end_time:
            yaw, pitch = predict_view(model_yaw, model_pitch, first_time)
            if len(hw) < 5:
                yaw, pitch = hw_yaw[-1], hw_pitch[-1]
            if yaw < 0:
                yaw += math.pi * 2
            elif yaw > math.pi * 2:
                yaw -= math.pi * 2
            if pitch > math.pi * 0.5:
                pitch = math.pi * 0.5
            elif pitch < -math.pi * 0.5:
                pitch = -math.pi * 0.5
            if first_time >= segmentid * chunk_duration * 1000:
                predicted_record.append({'yaw': yaw, 'pitch': pitch})
            first_time += 40

    return predicted_record


def build_model(play_time, hw_yaw, hw_pitch):
    model_yaw = Ridge()
    model_yaw.fit(play_time, hw_yaw)
    model_pitch = Ridge()
    model_pitch.fit(play_time, hw_pitch)
    return model_yaw, model_pitch


def predict_view(model_yaw, model_pitch, time):
    yaw = model_yaw.predict([[time]])
    if yaw > (2*math.pi):
        yaw = yaw - 2 * math.pi
    elif yaw < 0:
        yaw = yaw + 2 * math.pi
    if yaw < 0:
        yaw = yaw + 2 * math.pi
    pitch = model_pitch.predict([[time]])
    if pitch > (math.pi/2):
        pitch = math.pi/2
    elif pitch < (-math.pi/2):
        pitch = -math.pi/2
    return yaw , pitch

def download_decision(network_stats, motion_history, video_size, curr_ts, user_data, video_info):
    """
    Self defined download strategy

    Parameters
    ----------
    network_stats: list
        a list represents historical network status
    motion_history: list
        a list represents historical motion information
    video_size: dict
        video size of preprocessed video
    curr_ts: int
        current system timestamp
    user_data: dict
        user related parameters and information
    video_info: dict
        video information for decision module


    Returns
    -------
    dl_list: list
        the list of tiles which are decided to be downloaded
    user_data: dict
        updated user_date
    """


    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info)

    config_params = user_data['config_params']
    video_info = user_data['video_info']

    if curr_ts == 0:  # initialize the related parameters
        user_data['next_download_idx'] = 0
        user_data['latest_decision'] = []
    dl_list = []
    chunk_idx = user_data['next_download_idx']
    latest_decision = user_data['latest_decision']
    chunk_duration = video_info['chunk_duration']

    if user_data['next_download_idx'] >= int(video_info['duration'] / video_info['chunk_duration']):
        return dl_list, user_data
    download_size_next = 0
    download_size_now = 0
    bandwidth = network_stats[0]['bandwidth']
    rtt = network_stats[0]['rtt']
    for i in range(72):
        if user_data['next_download_idx'] == int(video_info['duration'] / video_info['chunk_duration']) - 1:
            download_size_next = 0
        else:
            download_size_next += get_video_json_size(video_size, chunk_idx + 1, f"chunk_{str(chunk_idx+1).zfill(4)}_tile_{str(i).zfill(3)}")
        download_size_now += get_video_json_size(video_size, chunk_idx, f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(i).zfill(3)}")
    download_max_next = download_size_next / bandwidth / 1000
    max_pre_time = download_max_next * 3 / 4
    download_max_now = download_size_now / bandwidth / 1000
    max_re_time = download_max_now * 0.15
    min_pre_download = np.ceil((max_pre_time + rtt + 10) / 10) * 10
    min_re_download = np.ceil((rtt + max_re_time + 10) / 10) * 10

    is_last_prediction = 0
    if motion_history[-1]['motion_ts'] == ((chunk_idx + 1) * chunk_duration * 1000 - min_pre_download - 10):
        is_last_prediction = 1
    predicted_record = predict_motion_tile(motion_history, config_params['motion_history_size'], config_params['motion_prediction_size'], chunk_idx, chunk_duration, min_re_download, is_last_prediction)  # motion prediction
    tile_record = tile_decision(predicted_record, video_size, video_info['range_fov'], chunk_idx, user_data, curr_ts, chunk_duration, motion_history[-1]['motion_ts'] + min_re_download, is_last_prediction)     # tile decision

    dl_list = generate_dl_list(chunk_idx, tile_record, latest_decision, dl_list)

    user_data = update_decision_info(user_data, tile_record, curr_ts, chunk_duration, min_pre_download)  # update decision information

    return dl_list, user_data

def generate_display_result(curr_display_frames, current_display_chunks, curr_fov, dst_video_frame_uri, frame_idx, video_size, user_data, video_info):
    """
    Generate fov images corresponding to different approaches

    Parameters
    ----------
    curr_display_frames: list
        current available video tile frames
    current_display_chunks: list
        current available video chunks
    curr_fov: dict
        current fov information, with format {"curr_motion", "range_fov", "fov_resolution"}
    dst_video_frame_uri: str
        the uri of generated fov frame
    frame_idx: int
        frame index of current display frame
    video_size: dict
        video size of preprocessed video
    user_data: dict
        user related parameters and information
    video_info: dict
        video information for evaluation

    Returns
    -------
    user_data: dict
        updated user_data
    """

    get_logger().debug(f'[evaluation] start get display img {frame_idx}')

    if user_data is None or "video_info" not in user_data:
        user_data = init_user(user_data, video_info, eva_flag=True)

    video_info = user_data['video_info']
    config_params = user_data['config_params']
    total_tile_num = config_params['total_tile_num']
    device = user_data['device']

    chunk_idx = int(frame_idx * (1000 / video_info['video_fps']) // (video_info['chunk_duration'] * 1000))  # frame idx starts from 0
    if chunk_idx <= len(current_display_chunks) - 1:
        tile_list = current_display_chunks[chunk_idx]['tile_list']
    else:
        tile_list = current_display_chunks[-1]['tile_list']

    # 将所有72之后的tile_idx都改成0~71之间的，同时把对应的frame进行resize
    org_tile_list = []  # 将所有的tile idx都改成0~71内的tile info
    org_frame_list = []  # 将tile idx大于等于72的frame插值
    for i in range(len(tile_list)):
        tile_id = tile_list[i]['tile_id']
        temp_idx = video_size[tile_id]['user_video_spec']['tile_info']['tile_idx']
        temp_frame = curr_display_frames[i]  # 获得这个temp idx对应的帧
        tile_idx = temp_idx % 72
        org_tile_width = user_data['config_params']['org_tile_width']
        org_tile_height = user_data['config_params']['org_tile_height']

        if temp_idx >= total_tile_num:      # 有低分辨率的tile了
            if 216 <= temp_idx <= 287:
                model = user_data['model1080p']
                model.eval()

                # 超分
            inter_img = cv2.resize(temp_frame, (org_tile_width, org_tile_height),
                                       interpolation=cv2.INTER_LINEAR)  # 插值
            if (video_size[tile_id]['user_video_spec']['tile_info']["SR_mse"]
                    < video_size[tile_id]['user_video_spec']['tile_info']["INTER_mse"]):  # 如果超分的mse更小则进行超分，否则直接使用插值的图片即可
                inter_img = np.array(inter_img).astype(np.float32)
                ycbcr = convert_rgb_to_ycbcr(inter_img)
                y = ycbcr[..., 0]
                y /= 255.
                y = torch.from_numpy(y).to(device)
                y = y.unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    preds = model(y).clamp(0.0, 1.0)  # 超分后的结果
                preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
                output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
                output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
                # end 超分
                org_frame_list.append(output)
            else:
                org_frame_list.append(inter_img)
        else:
            org_frame_list.append(temp_frame)
        org_tile_list.append(tile_idx)

    # calculating fov_uv parameters
    fov_ypr = [float(curr_fov['curr_motion']['yaw']), float(curr_fov['curr_motion']['pitch']), 0]
    _3d_polar_coord = fov_to_3d_polar_coord(fov_ypr, curr_fov['range_fov'], curr_fov['fov_resolution'])
    pixel_coord = _3d_polar_coord_to_pixel_coord(_3d_polar_coord, config_params['projection_mode'], [config_params['converted_height'], config_params['converted_width']])

    coord_tile_list = pixel_coord_to_tile(pixel_coord, config_params['total_tile_num'], video_size, chunk_idx)
    relative_tile_coord = pixel_coord_to_relative_tile_coord(pixel_coord, coord_tile_list, video_size, chunk_idx)
    org_coord_tile_list = copy.deepcopy(coord_tile_list)   # copy.deepcopy()的用法是将某一个变量的值赋值给另一个变量(此时两个变量地址不同)，因为地址不同，所以可以防止变量间相互干扰

    display_img = np.full((org_coord_tile_list.shape[0], org_coord_tile_list.shape[1], 3), [128, 128, 128], dtype=np.float32)  # create an empty matrix for the final image

    for i, tile_idx in enumerate(org_tile_list):
        hit_coord_mask = (org_coord_tile_list == tile_idx)
        if not np.any(hit_coord_mask):  # if no pixels belong to the current frame, skip
            continue
        dstMap_u, dstMap_v = cv2.convertMaps(relative_tile_coord[0].astype(np.float32),
                                             relative_tile_coord[1].astype(np.float32), cv2.CV_16SC2)

        remapped_frame = cv2.remap(org_frame_list[i], dstMap_u, dstMap_v, cv2.INTER_LINEAR)
        display_img[hit_coord_mask] = remapped_frame[hit_coord_mask]

    cv2.imwrite(dst_video_frame_uri, display_img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    get_logger().debug(f'[evaluation] end get display img {frame_idx}')

    return user_data


def update_decision_info(user_data, tile_record, curr_ts, chunk_duration,min_pre_download):
    """
    update the decision information

    Parameters
    ----------
    user_data: dict
        user related parameters and information
    tile_record: list
        recode the tiles should be downloaded
    curr_ts: int
        current system timestamp

    Returns
    -------
    user_data: dict
        updated user_data
    """

    # update latest_decision
    for i in range(len(tile_record)):
        if tile_record[i] not in user_data['latest_decision']:
            user_data['latest_decision'].append(tile_record[i])
    if user_data['config_params']['background_flag']:
        if -1 not in user_data['latest_decision']:
            user_data['latest_decision'].append(-1)

    # update chunk_idx & latest_decision
    # if curr_ts == 0 or curr_ts >= user_data['video_info']['pre_download_duration'] + user_data['next_download_idx'] * user_data['video_info']['chunk_duration'] * 1000 + chunk_duration * 1000 * 0.75:
    if curr_ts >= user_data['video_info']['pre_download_duration'] + user_data['next_download_idx'] * user_data['video_info']['chunk_duration'] * 1000 + (chunk_duration * 1000 - min_pre_download - 10):
        user_data['next_download_idx'] += 1
        user_data['latest_decision'] = []
        user_data['tile_idx_temp'] = []
        #print("----------------------------------chunk" + str(user_data['next_download_idx']))

    return user_data

# erp
def tile_segment_info(chunk_info, user_data):
    """
    Generate the information for the current tile, with required format
    Parameters
    ----------
    chunk_info: dict
        chunk information
    user_data: dict
        user related information

    Returns
    -------
    tile_info: dict
        tile related information, with format {chunk_idx:, tile_idx:}
    segment_info: dict
        segmentation related information, with format
        {segment_out_info:{width:, height:}, start_position:{width:, height:}}
    """

    tile_idx = user_data['tile_idx']
    temp_idx = tile_idx % 72  # add

    index_width = temp_idx % user_data['config_params']['tile_width_num']        # determine which col
    index_height = temp_idx // user_data['config_params']['tile_width_num']      # determine which row

    segment_info = {
        'segment_out_info': {
            'width': user_data['config_params']['tile_width'],
            'height': user_data['config_params']['tile_height']
        },
        'start_position': {
            'width': index_width * user_data['config_params']['tile_width'],
            'height': index_height * user_data['config_params']['tile_height']
        }
    }

    tile_info = {
        'chunk_idx': user_data['chunk_idx'],
        'tile_idx': tile_idx
    }

    return tile_info, segment_info

def tile_decision(predicted_record, video_size, range_fov, chunk_idx, user_data, curr_ts, chunk_duration,predict_ts,is_last_prediction):
    """
    Deciding which tiles should be transmitted for each chunk, within the prediction window
    (As an example, users can implement their customized function.)

    Parameters
    ----------
    predicted_record: dict
        the predicted motion, with format {yaw: , pitch: , scale:}, where
        the parameter 'scale' is used for transcoding approach
    video_size: dict
        the recorded whole video size after video preprocessing
    range_fov: list
        degree range of fov, with format [height, width]
    chunk_idx: int
        index of current chunk
    user_data: dict
        user related data structure, necessary information for tile decision

    Returns
    -------
    tile_record: list
        the decided tile list of current update, each item is the chunk index
    """
    # The current tile decision method is to sample the fov range corresponding to the predicted motion of each chunk,
    # and the union of the tile sets mapped by these sampling points is the tile set to be transmitted.
    config_params = user_data['config_params']
    tile_record = []
    sampling_size = [50, 50]
    converted_width = user_data['config_params']['converted_width']
    converted_height = user_data['config_params']['converted_height']
    tile_info = read_tile_info(video_size, chunk_idx)
    # download_max = 0
    # for i in range(72):
    #     download_max += tile_info[i]['size'] / 1000000 / 100 * 1000
    # print("chunk：",chunk_idx,"最大下载时间:",download_max)
    for predicted_motion in predicted_record:
        _3d_polar_coord = fov_to_3d_polar_coord([float(predicted_motion['yaw']), float(predicted_motion['pitch']), 0], range_fov, sampling_size)
        pixel_coord = _3d_polar_coord_to_pixel_coord(_3d_polar_coord, config_params['projection_mode'], [converted_height, converted_width])
        coord_tile_list = pixel_coord_to_tile(pixel_coord, config_params['total_tile_num'], video_size, chunk_idx)
        unique_tile_list = [int(item) for item in np.unique(coord_tile_list)]
        for i in range(len(unique_tile_list)):
            if unique_tile_list[i] not in tile_record:
                tile_record.append(unique_tile_list[i])


    tile_record_unique = [int(item) for item in np.unique(tile_record)]
    tile_record_after = []
    download_time = 0
    real_download_time = 0
    # for tile_idx in tile_record_unique:
    for tile_idx in tile_record_unique:
        size = tile_info[tile_idx]['size']
        download_time += size / 1000000 / 100 * 1000  #ms

        if tile_idx not in user_data['latest_decision']:
            # 单分辨率代码
            tile_record_after.append(tile_idx)
            real_download_time += tile_info[tile_idx]['size'] / 1000000 / 100 * 1000

    tile_record = []
    if config_params['background_flag']:
        if -1 not in user_data['latest_decision']:
            tile_record_after.append(-1)
    else:
        for predicted_motion in predicted_record:
            if curr_ts >= 330:
                range_fov_background = [range_fov[0] + 10, range_fov[1] + 20]
            else:
                range_fov_background = [range_fov[0] + 60, range_fov[1] + 60]
            _3d_polar_coord = fov_to_3d_polar_coord([float(predicted_motion['yaw']), float(predicted_motion['pitch']), 0], range_fov_background, sampling_size)
            pixel_coord = _3d_polar_coord_to_pixel_coord(_3d_polar_coord, config_params['projection_mode'],[converted_height, converted_width])
            coord_tile_list = pixel_coord_to_tile(pixel_coord, config_params['total_tile_num'], video_size, chunk_idx)
            unique_tile_list = [int(item) for item in np.unique(coord_tile_list)]
            for i in range(len(unique_tile_list)):
                if unique_tile_list[i] not in tile_record:
                    tile_record.append(unique_tile_list[i])
        tile_record_unique_expend = [int(item) for item in np.unique(tile_record)]
        for tile_idx in tile_record_unique_expend:
            if tile_idx not in tile_record_unique:
                download_time += tile_info[tile_idx + 72 * 3]['size'] / 1000000 / 100 * 1000
            if tile_idx not in tile_record_unique and tile_idx not in user_data['latest_decision']:
                tile_record_after.append(tile_idx + 72 * 3)
                real_download_time += tile_info[tile_idx + 72 * 3]['size'] / 1000000 / 100*1000
        # if real_download_time / download_max > 0.1:
        #     print("chunk：", chunk_idx, "下载总时间：", download_time, "实际下载时间占比：", real_download_time / download_max)

    return tile_record_after


def read_tile_info(video_size, chunk_idx):
    tile_info = {}
    for tile_idx in range(72):
        for i in range(4):
            if(i==0 or i ==3):
               mse_INTER = video_size[f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(tile_idx + 72 * i).zfill(3)}"]['user_video_spec']["tile_info"]["INTER_mse"]
               mse_SR = video_size[f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(tile_idx + 72 * i).zfill(3)}"]['user_video_spec']["tile_info"]["SR_mse"]
               tile_size = video_size[f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(tile_idx + 72 * i).zfill(3)}"]["video_size"]
               tile_info[tile_idx + 72 * i] = {'mse_INTER': mse_INTER, 'mse_SR': mse_SR, 'size': tile_size}
    return tile_info

def set_encoding_params():
    encoding_params = {
        "encoder": 'libx264',
        "qp_list": [29],
        "preset": 'faster',
        "video_fps": 30,
        "gop": 30,
        "bf": 0
    }
    return encoding_params
def fov_to_3d_polar_coord(fov_direction, fov_range, fov_resolution):
    """
    Given fov information, convert it to 3D polar coordinates.

    Parameters
    ----------
    fov_direction: dict
        The orientation of fov, with format {yaw: , pitch: , roll:}
    fov_range: list
        The angle range corresponding to fov, expressed in degrees
    fov_resolution: list
        the fov resolution, with format [height, width]

    Returns
    -------
    _3d_polar_coord: array
        3D polar coordinates of fov

    """
    try:
        fov_w, fov_h = fov_range[1] * np.pi / 180, fov_range[0] * np.pi / 180
    except:
        fov_w, fov_h = 90 * np.pi / 180, 90 * np.pi / 180

    # calculate the 3d cartesian coordinates
    vp_yaw, vp_pitch, vp_roll = fov_direction
    _3d_cartesian_coord = calcualte_3d_cartesian_coord(fov_w, fov_h, vp_yaw, vp_pitch, vp_roll, fov_resolution)

    # calculate the 3d polar coordinates
    _3d_polar_coord = calculate_3d_polar_coord(_3d_cartesian_coord)

    return _3d_polar_coord


def calcualte_3d_cartesian_coord(fov_w, fov_h, vp_yaw, vp_pitch, vp_roll, fov_resolution):
    """
    Calculate the 3d spherical coordinates of fov

    Parameters
    ----------
    fov_w: float
        width of fov, in radian
    fov_h: float
        height of fov, in radian
    vp_yaw: float
        yaw of viewport, in radian
    vp_pitch: float
        pitch of viewport, in radian
    vp_roll: float
        roll of viewport, in radian
    fov_resolution: list
        the fov resolution, with format [height, width]

    Returns
    -------
    cartesian_coord: array
        spherical cartesian coordinate of the fov
    """

    m = np.linspace(0, fov_resolution[1] - 1, fov_resolution[1])
    n = np.linspace(0, fov_resolution[0] - 1, fov_resolution[0])

    u = (m + 0.5) * 2 * np.tan(fov_w/2) / fov_resolution[1]
    v = (n + 0.5) * 2 * np.tan(fov_h/2) / fov_resolution[0]

    # calculate the corresponding three-dimensional coordinates (x, y, z) mapped from the positive X-axis.
    x = 1.0
    y = u - np.tan(fov_w/2)
    z = -v + np.tan(fov_h/2)

    # coordinate point dimension expansion
    x_scale = np.ones((fov_resolution[0], fov_resolution[1]))
    y_scale = np.tile(y, (fov_resolution[0], 1))
    z_scale = np.tile(z, (fov_resolution[1], 1)).transpose()

    # unit sphere
    x_hat = x_scale / np.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)
    y_hat = y_scale / np.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)
    z_hat = z_scale / np.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)

    # rotation
    a = vp_yaw              # yaw, phi
    b = vp_pitch            # pitch, theta
    r = vp_roll             # roll, psi

    rot_a = np.array([np.cos(a)*np.cos(b), np.cos(a)*np.sin(b)*np.sin(r) - np.sin(a)*np.cos(r), np.cos(a)*np.sin(b)*np.cos(r) + np.sin(a)*np.sin(r)])
    rot_b = np.array([np.sin(a)*np.cos(b), np.sin(a)*np.sin(b)*np.sin(r) + np.cos(a)*np.cos(r), np.sin(a)*np.sin(b)*np.cos(r) - np.cos(a)*np.sin(r)])
    rot_c = np.array([-np.sin(b), np.cos(b)*np.sin(r), np.cos(b)*np.cos(r)])

    xx = rot_a[0] * x_hat + rot_a[1] * y_hat + rot_a[2] * z_hat
    yy = rot_b[0] * x_hat + rot_b[1] * y_hat + rot_b[2] * z_hat
    zz = rot_c[0] * x_hat + rot_c[1] * y_hat + rot_c[2] * z_hat

    xx = np.clip(xx, -1, 1)
    yy = np.clip(yy, -1, 1)
    zz = np.clip(zz, -1, 1)

    cartesian_coord = np.array([xx, yy, zz])

    return cartesian_coord


def calculate_3d_polar_coord(_3d_cartesian_coord):
    """
    Calculate the corresponding 3d polar coordinates

    Parameters
    ----------
    _3d_cartesian_coord: array
         spherical cartesian coordinate
    Returns
    -------
    _3d_polar_coord: array
        spherical polar coordinate, with foramt [phi, theta]

    """

    x, y, z = _3d_cartesian_coord[0], _3d_cartesian_coord[1], _3d_cartesian_coord[2]
    phi = np.arctan2(y, x)                      # range is [-pi, pi]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(z, r)                    # range from bottom to top is [-np.pi/2, np.pi/2]

    _3d_polar_coord = np.concatenate([phi, theta], axis=-1)

    return _3d_polar_coord
def _3d_polar_coord_to_pixel_coord(_3d_polar_coord, projection_type, src_resolution):
    """
    Given the 3d polar coordinates, convert it to pixel coordinates in the corresponding projection

    Parameters
    ----------
    _3d_polar_coord: array
         spherical polar coordinate, with foramt [phi, theta]
    projection_type: str
        projection format
    src_resolution: list
        source resolution, with format [height, width]

    Returns
    -------
    pixel_coord: array
        the pixel coordinates in the source projection format
    """

    if projection_type == "erp":
        pixel_coord = _3d_polar_coord_to_erp(_3d_polar_coord, src_resolution)
    elif projection_type == "cmp":
        pixel_coord = _3d_polar_coord_to_cmp(_3d_polar_coord, src_resolution)
    elif projection_type == "eac":
        pixel_coord = _3d_polar_coord_to_eac(_3d_polar_coord, src_resolution)
    else:
        raise Exception(f"the projection {projection_type} is not supported currently in e3po")

    return pixel_coord
def _3d_polar_coord_to_erp(polar_coord, src_resolution):
    """
    Convert polar coordinates to pixel coordinates in ERP format

    Parameters
    ----------
    polar_coord: array
        polar coord, with format [phi, theta]
    src_resolution: list
        source resolution, with format [height, width]

    Returns
    -------
    pixel_coord: array
        the corresponding pixel coordinates in ERP format
    """

    erp_height, erp_width = src_resolution[0], src_resolution[1]

    phi, theta = np.split(polar_coord, 2, axis=-1)
    phi = phi.reshape(phi.shape[:2])
    theta = theta.reshape(theta.shape[:2])

    u = phi / (2 * np.pi) + 0.5
    v = 0.5 - theta / np.pi

    coor_x = u * erp_width - 0.5
    coor_y = v * erp_height - 0.5

    coor_x = np.clip(coor_x, 0, erp_width - 1)
    coor_y = np.clip(coor_y, 0, erp_height - 1)

    pixel_coord = [coor_x, coor_y]

    return pixel_coord


def _3d_polar_coord_to_cmp(polar_coord, src_resolution):
    """
    Convert polar coordinates to pixel coordinates in CMP format

    Parameters
    ----------
    polar_coord: array
        polar coord, with format [phi, theta]
    src_resolution: list
        source resolution, with format [height, width]

    Returns
    -------
    pixel_coord: array
        the corresponding pixel coordinates in CMP format
    """

    cmp_height, cmp_width = src_resolution[0], src_resolution[1]
    u, v = np.split(polar_coord, 2, axis=-1)
    u = u.reshape(u.shape[:2])
    v = v.reshape(v.shape[:2])

    face_size_w = cmp_width // 3
    face_size_h = cmp_height // 2
    assert (face_size_w == face_size_h)  # ensure the ratio of w:h is 3:2

    x_sphere = np.round(np.cos(v) * np.cos(u), 9)
    y_sphere = np.round(np.cos(v) * np.sin(u), 9)
    z_sphere = np.round(np.sin(v), 9)

    dst_h, dst_w = np.shape(u)[:2]
    coor_x = np.zeros((dst_h, dst_w))
    coor_y = np.zeros((dst_h, dst_w))

    for i in range(6):
        if i == 0:
            temp_index1 = np.where(y_sphere < 0, 1, -1)
            temp_index2 = np.where(abs(y_sphere) >= abs(x_sphere), 1, -2)
            temp_index3 = np.where(abs(y_sphere) >= abs(z_sphere), 1, -3)
            temp_index = (temp_index1 == np.where(temp_index2 == temp_index3, 1, -2))
            u_cub = x_sphere[temp_index] / abs(y_sphere[temp_index])
            v_cub = -z_sphere[temp_index] / abs(y_sphere[temp_index])
        elif i == 1:
            temp_index1 = np.where(x_sphere > 0, 1, -1)
            temp_index2 = np.where(abs(x_sphere) >= abs(y_sphere), 1, -2)
            temp_index3 = np.where(abs(x_sphere) >= abs(z_sphere), 1, -3)
            temp_index = (temp_index1 == np.where(temp_index2 == temp_index3, 1, -2))
            u_cub = y_sphere[temp_index] / abs(x_sphere[temp_index])
            v_cub = -z_sphere[temp_index] / abs(x_sphere[temp_index])
        elif i == 2:
            temp_index1 = np.where(y_sphere > 0, 1, -1)
            temp_index2 = np.where(abs(y_sphere) >= abs(x_sphere), 1, -2)
            temp_index3 = np.where(abs(y_sphere) >= abs(z_sphere), 1, -3)
            temp_index = (temp_index1 == np.where(temp_index2 == temp_index3, 1, -2))
            u_cub = -x_sphere[temp_index] / abs(y_sphere[temp_index])
            v_cub = -z_sphere[temp_index] / abs(y_sphere[temp_index])
        elif i == 3:
            temp_index1 = np.where(z_sphere < 0, 1, -1)
            temp_index2 = np.where(abs(z_sphere) >= abs(x_sphere), 1, -2)
            temp_index3 = np.where(abs(z_sphere) >= abs(y_sphere), 1, -3)
            temp_index = (temp_index1 == np.where(temp_index2 == temp_index3, 1, -2))
            u_cub = -x_sphere[temp_index] / abs(z_sphere[temp_index])
            v_cub = -y_sphere[temp_index] / abs(z_sphere[temp_index])
        elif i == 4:
            temp_index1 = np.where(x_sphere < 0, 1, -1)
            temp_index2 = np.where(abs(x_sphere) >= abs(y_sphere), 1, -2)
            temp_index3 = np.where(abs(x_sphere) >= abs(z_sphere), 1, -3)
            temp_index = (temp_index1 == np.where(temp_index2 == temp_index3, 1, -2))
            u_cub = z_sphere[temp_index] / abs(x_sphere[temp_index])
            v_cub = -y_sphere[temp_index] / abs(x_sphere[temp_index])
        elif i == 5:
            temp_index1 = np.where(z_sphere > 0, 1, -1)
            temp_index2 = np.where(abs(z_sphere) >= abs(x_sphere), 1, -2)
            temp_index3 = np.where(abs(z_sphere) >= abs(y_sphere), 1, -3)
            temp_index = (temp_index1 == np.where(temp_index2 == temp_index3, 1, -2))
            u_cub = x_sphere[temp_index] / abs(z_sphere[temp_index])
            v_cub = -y_sphere[temp_index] / abs(z_sphere[temp_index])

        face_index = i
        m_cub = (u_cub + 1) * face_size_w / 2 - 0.5
        n_cub = (v_cub + 1) * face_size_h / 2 - 0.5
        coor_x[temp_index] = (face_index % 3) * face_size_w + m_cub
        coor_y[temp_index] = (face_index // 3) * face_size_h + n_cub

        coor_x[temp_index] = np.clip(coor_x[temp_index], (face_index % 3) * face_size_w,
                                     (face_index % 3 + 1) * face_size_w - 1)
        coor_y[temp_index] = np.clip(coor_y[temp_index], (face_index // 3) * face_size_h,
                                     (face_index // 3 + 1) * face_size_h - 1)

    pixel_coord = [coor_x, coor_y]

    return pixel_coord


def _3d_polar_coord_to_eac(polar_coord, src_resolution):
    """
    Convert polar coordinates to pixel coordinates in EAC format

    Parameters
    ----------
    polar_coord: array
        polar coord, with format [phi, theta]
    src_resolution: list
        source resolution, with format [height, width]

    Returns
    -------
    pixel_coord: array
        the corresponding pixel coordinates in EAC format
    """

    eac_height, eac_width = src_resolution[0], src_resolution[1]
    u, v = np.split(polar_coord, 2, axis=-1)
    u = u.reshape(u.shape[:2])
    v = v.reshape(v.shape[:2])

    face_size_w = eac_width // 3
    face_size_h = eac_height // 2
    assert (face_size_w == face_size_h)     # ensure the ratio of w:h is 3:2

    x_sphere = np.round(np.cos(v) * np.cos(u), 9)
    y_sphere = np.round(np.cos(v) * np.sin(u), 9)
    z_sphere = np.round(np.sin(v), 9)

    dst_h, dst_w = np.shape(u)[:2]
    coor_x = np.zeros((dst_h, dst_w))
    coor_y = np.zeros((dst_h, dst_w))

    for i in range(6):
        if i == 0:
            temp_index1 = np.where(y_sphere < 0, 1, -1)
            temp_index2 = np.where(abs(y_sphere) >= abs(x_sphere), 1, -2)
            temp_index3 = np.where(abs(y_sphere) >= abs(z_sphere), 1, -3)
            temp_index = (temp_index1 == np.where(temp_index2 == temp_index3, 1, -2))
            u_cub = x_sphere[temp_index] / abs(y_sphere[temp_index])
            v_cub = -z_sphere[temp_index] / abs(y_sphere[temp_index])
        elif i == 1:
            temp_index1 = np.where(x_sphere > 0, 1, -1)
            temp_index2 = np.where(abs(x_sphere) >= abs(y_sphere), 1, -2)
            temp_index3 = np.where(abs(x_sphere) >= abs(z_sphere), 1, -3)
            temp_index = (temp_index1 == np.where(temp_index2 == temp_index3, 1, -2))
            u_cub = y_sphere[temp_index] / abs(x_sphere[temp_index])
            v_cub = -z_sphere[temp_index] / abs(x_sphere[temp_index])
        elif i == 2:
            temp_index1 = np.where(y_sphere > 0, 1, -1)
            temp_index2 = np.where(abs(y_sphere) >= abs(x_sphere), 1, -2)
            temp_index3 = np.where(abs(y_sphere) >= abs(z_sphere), 1, -3)
            temp_index = (temp_index1 == np.where(temp_index2 == temp_index3, 1, -2))
            u_cub = -x_sphere[temp_index] / abs(y_sphere[temp_index])
            v_cub = -z_sphere[temp_index] / abs(y_sphere[temp_index])
        elif i == 3:
            temp_index1 = np.where(z_sphere < 0, 1, -1)
            temp_index2 = np.where(abs(z_sphere) >= abs(x_sphere), 1, -2)
            temp_index3 = np.where(abs(z_sphere) >= abs(y_sphere), 1, -3)
            temp_index = (temp_index1 == np.where(temp_index2 == temp_index3, 1, -2))
            u_cub = -x_sphere[temp_index] / abs(z_sphere[temp_index])
            v_cub = -y_sphere[temp_index] / abs(z_sphere[temp_index])
        elif i == 4:
            temp_index1 = np.where(x_sphere < 0, 1, -1)
            temp_index2 = np.where(abs(x_sphere) >= abs(y_sphere), 1, -2)
            temp_index3 = np.where(abs(x_sphere) >= abs(z_sphere), 1, -3)
            temp_index = (temp_index1 == np.where(temp_index2 == temp_index3, 1, -2))
            u_cub = z_sphere[temp_index] / abs(x_sphere[temp_index])
            v_cub = -y_sphere[temp_index] / abs(x_sphere[temp_index])
        elif i == 5:
            temp_index1 = np.where(z_sphere > 0, 1, -1)
            temp_index2 = np.where(abs(z_sphere) >= abs(x_sphere), 1, -2)
            temp_index3 = np.where(abs(z_sphere) >= abs(y_sphere), 1, -3)
            temp_index = (temp_index1 == np.where(temp_index2 == temp_index3, 1, -2))
            u_cub = x_sphere[temp_index] / abs(z_sphere[temp_index])
            v_cub = -y_sphere[temp_index] / abs(z_sphere[temp_index])

        face_index = i
        u_cub = np.arctan(u_cub) * 4 / np.pi
        v_cub = np.arctan(v_cub) * 4 / np.pi
        m_cub = (u_cub + 1) * face_size_w / 2 - 0.5
        n_cub = (v_cub + 1) * face_size_h / 2 - 0.5
        coor_x[temp_index] = (face_index % 3) * face_size_w + m_cub
        coor_y[temp_index] = (face_index // 3) * face_size_h + n_cub

        coor_x[temp_index] = np.clip(coor_x[temp_index], (face_index % 3) * face_size_w,
                                     (face_index % 3 + 1) * face_size_w - 1)
        coor_y[temp_index] = np.clip(coor_y[temp_index], (face_index // 3) * face_size_h,
                                     (face_index // 3 + 1) * face_size_h - 1)
    pixel_coord = [coor_x, coor_y]

    return pixel_coord
def pixel_coord_to_relative_tile_coord(pixel_coord, coord_tile_list, video_info, chunk_idx):
    """
    Calculate the relative position of the pixel_coord coordinates on each tile.

    Parameters
    ----------
    pixel_coord: array
        pixel coordinates
    coord_tile_list: list
        calculated tile list
    video_info: dict
    chunk_idx: int
        chunk index

    Returns
    -------
    relative_tile_coord: array
        the relative tile coord for the given pixel coordinates
    """

    relative_tile_coord = copy.deepcopy(pixel_coord)
    unique_tile_list = np.unique(coord_tile_list)
    for i in unique_tile_list:
        tile_id = f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(i).zfill(3)}"
        tile_start_width = video_info[tile_id]['user_video_spec']['segment_info']['start_position']['width']
        tile_start_height = video_info[tile_id]['user_video_spec']['segment_info']['start_position']['height']
        tile_width = video_info[tile_id]['user_video_spec']['segment_info']['segment_out_info']['width']
        tile_height = video_info[tile_id]['user_video_spec']['segment_info']['segment_out_info']['height']

        hit_coord_mask = (coord_tile_list == i)

        # update the relative position
        relative_tile_coord[0][hit_coord_mask] = np.clip(relative_tile_coord[0][hit_coord_mask] - tile_start_width, 0, tile_width - 1)
        relative_tile_coord[1][hit_coord_mask] = np.clip(relative_tile_coord[1][hit_coord_mask] - tile_start_height, 0, tile_height - 1)

    return relative_tile_coord
def pixel_coord_to_tile(pixel_coord, total_tile_num, video_size, chunk_idx):
    """
    Calculate the corresponding tile, for given pixel coordinates

    Parameters
    ----------
    pixel_coord: array
        pixel coordinates
    total_tile_num: int
        total num of tiles for different approach
    video_size: dict
        video size of preprocessed video
    chunk_idx: int
        chunk index

    Returns
    -------
    coord_tile_list: list
        the calculated tile list, for the given pixel coordinates
    """

    coord_tile_list = np.full(pixel_coord[0].shape, 0)
    for i in range(total_tile_num):
        tile_id = f"chunk_{str(chunk_idx).zfill(4)}_tile_{str(i).zfill(3)}"
        if tile_id not in video_size:
            continue
        tile_idx = video_size[tile_id]['user_video_spec']['tile_info']['tile_idx']
        tile_start_width = video_size[tile_id]['user_video_spec']['segment_info']['start_position']['width']
        tile_start_height = video_size[tile_id]['user_video_spec']['segment_info']['start_position']['height']
        tile_width = video_size[tile_id]['user_video_spec']['segment_info']['segment_out_info']['width']
        tile_height = video_size[tile_id]['user_video_spec']['segment_info']['segment_out_info']['height']

        # Create a Boolean mask to check if the coordinates are within the tile range
        mask_width = (tile_start_width <= pixel_coord[0]) & (pixel_coord[0] < tile_start_width + tile_width)
        mask_height = (tile_start_height <= pixel_coord[1]) & (pixel_coord[1] < tile_start_height + tile_height)

        # find coordinates that satisfy both width and height conditions
        hit_coord_mask = mask_width & mask_height

        # update coord_tile_list
        coord_tile_list[hit_coord_mask] = tile_idx

    return coord_tile_list







