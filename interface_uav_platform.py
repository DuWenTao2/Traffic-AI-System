import requests
import cv2
import numpy as np
from ultralytics import YOLO
import time

# ==================== 配置参数 ====================
# 平台基础配置（模块级默认值，供内部函数使用）
PLATFORM_HOST = "http://58.40.19.96:8886"  # 平台地址
TICKET = "XwMCDMRiQ5HKYBr4SdN4dA=="  # ticket

# ==================== 认证中心接口 ====================
def get_token():
    """获取访问令牌 /ticketLogin"""
    url = f"{PLATFORM_HOST}/prod-api/auth/ticketLogin"
    headers = {"Content-Type": "application/json"}
    data = {"ticket": TICKET}
    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            token = result.get("token")
            print(f"✅ [get_token] Token获取成功: {token[:20]}...")
            return token
        else:
            print(f"❌ [get_token] Token获取失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [get_token] 异常: {str(e)}")
        return None

# ==================== 基础数据 - 飞机数据接口 ====================
def uav_list(token, params=None):
    """查询飞机列表 /openapi/ctuav/v4/uav/list [POST]"""
    if not token:
        print("❌ [uav_list] Token为空")
        return None
    if params is None:
        params = {}
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/uav/list"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    try:
        response = requests.post(url, json=params, headers=headers, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [uav_list] 飞机列表查询成功，共{len(result.get('data', []))}条")
            return result
        else:
            print(f"❌ [uav_list] 查询失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [uav_list] 异常: {str(e)}")
        return None

def uav_detail(token, uav_id):
    """查询飞机详细 /openapi/ctuav/v4/uav/detail [GET]"""
    if not token:
        print("❌ [uav_detail] Token为空")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/uav/detail"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"uavId": uav_id}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [uav_detail] 飞机详情查询成功: {result.get('data').get('uavName')}")
            return result
        else:
            print(f"❌ [uav_detail] 查询失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [uav_detail] 异常: {str(e)}")
        return None

# ==================== 基础数据 - 航线数据接口 ====================
def flyline_list(token, params=None):
    """查询航线列表 /openapi/ctuav/v4/flyline/list [POST]"""
    if not token:
        print("❌ [flyline_list] Token为空")
        return None
    if params is None:
        params = {}
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyline/list"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    try:
        response = requests.post(url, json=params, headers=headers, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flyline_list] 航线列表查询成功，共{len(result.get('data', []))}条")
            return result
        else:
            print(f"❌ [flyline_list] 查询失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyline_list] 异常: {str(e)}")
        return None

def flyline_import(token, file_path, name):
    """导入航线kmz文件 /openapi/ctuav/v4/flyline/import [POST form-data]"""
    if not token:
        print("❌ [flyline_import] Token为空")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyline/import"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with open(file_path, 'rb') as f:
            files = {'multipartFile': f}
            data = {'name': name}
            response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flyline_import] 航线文件导入成功")
            return result
        else:
            print(f"❌ [flyline_import] 导入失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyline_import] 异常: {str(e)}")
        return None

def flyline_export(token, fly_line_id):
    """导出航线kmz文件 /openapi/ctuav/v4/flyline/export [GET]"""
    if not token:
        print("❌ [flyline_export] Token为空")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyline/export"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"flyLineId": fly_line_id}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flyline_export] 航线导出成功，下载链接: {result.get('msg')}")
            return result
        else:
            print(f"❌ [flyline_export] 导出失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyline_export] 异常: {str(e)}")
        return None

def flyline_delete(token, fly_line_id, fly_line_version):
    """删除航线数据 /openapi/ctuav/v4/flyline/delete [POST]"""
    if not token:
        print("❌ [flyline_delete] Token为空")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyline/delete"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "flyLineId": fly_line_id,
        "flyLineVersion": fly_line_version
    }
    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flyline_delete] 航线删除成功")
            return result
        else:
            print(f"❌ [flyline_delete] 删除失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyline_delete] 异常: {str(e)}")
        return None

def flyline_upload(token, name, line_type, fly_line_version, fly_line_point_str, file_path):
    """导入kml文件 /openapi/ctuav/v4/flyline/upload [POST]"""
    if not token:
        print("❌ [flyline_upload] Token为空")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyline/upload"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with open(file_path, 'rb') as f:
            files = {'multipartFile': f}
            data = {
                "name": name,
                "lineType": line_type,
                "flyLineVersion": fly_line_version,
                "flyLinePointStr": fly_line_point_str
            }
            response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flyline_upload] KML文件导入成功")
            return result
        else:
            print(f"❌ [flyline_upload] 导入失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyline_upload] 异常: {str(e)}")
        return None

# ==================== 基础数据 - 飞行计划接口 ====================
def flyplan_list(token, params=None):
    """查询飞行计划列表 /openapi/ctuav/v4/flyplan/list [POST]"""
    if not token:
        print("❌ [flyplan_list] Token为空")
        return None
    if params is None:
        params = {}
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyplan/list"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    try:
        response = requests.post(url, json=params, headers=headers, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flyplan_list] 飞行计划列表查询成功，共{len(result.get('data', []))}条")
            return result
        else:
            print(f"❌ [flyplan_list] 查询失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyplan_list] 异常: {str(e)}")
        return None

def flyplan_add(token, params=None):
    """新建飞行计划 /openapi/ctuav/v4/flyplan/add [POST]"""
    if not token:
        print("❌ [flyplan_add] Token为空")
        return None
    if params is None:
        params = {}
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyplan/add"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    try:
        response = requests.post(url, json=params, headers=headers, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flyplan_add] 飞行计划新建成功")
            return result
        else:
            print(f"❌ [flyplan_add] 新建失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyplan_add] 异常: {str(e)}")
        return None

def flyplan_update(token, params):
    """更新飞行计划 /openapi/ctuav/v4/flyplan/update [POST]"""
    if not token:
        print("❌ [flyplan_update] Token为空")
        return None
    if not params.get("id"):
        print("❌ [flyplan_update] 缺少必填参数id")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyplan/update"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    try:
        response = requests.post(url, json=params, headers=headers, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flyplan_update] 飞行计划更新成功")
            return result
        else:
            print(f"❌ [flyplan_update] 更新失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyplan_update] 异常: {str(e)}")
        return None

def flyplan_delete(token, record_id):
    """删除飞行计划 /openapi/ctuav/v4/flyplan/delete [GET]"""
    if not token:
        print("❌ [flyplan_delete] Token为空")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyplan/delete"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"recordId": record_id}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flyplan_delete] 飞行计划删除成功")
            return result
        else:
            print(f"❌ [flyplan_delete] 删除失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyplan_delete] 异常: {str(e)}")
        return None

def flyplan_result(token, record_id):
    """查询飞行计划成果 /openapi/ctuav/v4/flyplan/result [GET]"""
    if not token:
        print("❌ [flyplan_result] Token为空")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyplan/result"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"recordId": record_id}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            # print(f"✅ [flyplan_result] 飞行计划成果查询成功，视频流数量: {len(result.get('data', {}).get('hlsList', []))}")
            return result
        else:
            print(f"❌ [flyplan_result] 查询失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyplan_result] 异常: {str(e)}")
        return None

# ==================== 飞行数据接口 ====================
def onlineuav_list(token):
    """查询在线飞机列表 /openapi/ctuav/v4/onlineuav/list [GET]"""
    if not token:
        print("❌ [onlineuav_list] Token为空")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/onlineuav/list"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [onlineuav_list] 在线飞机列表查询成功，共{len(result.get('data', []))}架")
            return result
        else:
            print(f"❌ [onlineuav_list] 查询失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [onlineuav_list] 异常: {str(e)}")
        return None

def flydata_list(token, record_id, dept_id, create_time=""):
    """查询在线飞机数据 /openapi/ctuav/v4/flydata/list [POST]"""
    if not token:
        print("❌ [flydata_list] Token为空")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flydata/list"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "recordId": record_id,
        "deptId": dept_id,
        "createTime": create_time
    }
    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flydata_list] 飞行实时数据查询成功，共{len(result.get('data', []))}条")
            return result
        else:
            print(f"❌ [flydata_list] 查询失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flydata_list] 异常: {str(e)}")
        return None

# ==================== 历史数据接口 ====================
def flyhistory_list(token, dept_id, uav_id):
    """查询飞行历史列表 /openapi/ctuav/v4/flyhistory/list [POST]"""
    if not token:
        print("❌ [flyhistory_list] Token为空")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyhistory/list"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "deptId": dept_id,
        "uavId": uav_id
    }
    try:
        response = requests.get(url, json=data, headers=headers, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flyhistory_list] 飞行历史列表查询成功，共{len(result.get('data', []))}条")
            return result
        else:
            print(f"❌ [flyhistory_list] 查询失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyhistory_list] 异常: {str(e)}")
        return None

def flyhistory_detail(token, record_id):
    """查询飞行历史详细 /openapi/ctuav/v4/flyhistory/detail [POST]"""
    if not token:
        print("❌ [flyhistory_detail] Token为空")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyhistory/detail"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {"recordId": record_id}
    try:
        response = requests.post(url, json=data, headers=headers, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flyhistory_detail] 飞行历史详情查询成功")
            return result
        else:
            print(f"❌ [flyhistory_detail] 查询失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyhistory_detail] 异常: {str(e)}")
        return None

def flyhistory_export(token, record_id):
    """导出飞行历史数据 /openapi/ctuav/v4/flyhistory/export [GET]"""
    if not token:
        print("❌ [flyhistory_export] Token为空")
        return None
    url = f"{PLATFORM_HOST}/prod-api/ctuav/openapi/ctuav/v4/flyhistory/export"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"recordId": record_id}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        result = response.json()
        if result.get("code") == 200:
            print(f"✅ [flyhistory_export] 飞行历史导出成功，文件: {result.get('msg')}")
            return result
        else:
            print(f"❌ [flyhistory_export] 导出失败: {result.get('msg')}")
            return None
    except Exception as e:
        print(f"❌ [flyhistory_export] 异常: {str(e)}")
        return None

# ==================== YOLOv8 实时识别 ====================
def yolov8_rtmp_detect(rtmp_url, model_path):
    """YOLOv8 对接RTMP流实时识别"""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(rtmp_url)
    if not cap.isOpened():
        print(f"❌ [yolov8_rtmp_detect] 无法打开RTMP流: {rtmp_url}")
        return
    print(f"✅ [yolov8_rtmp_detect] 成功打开RTMP流，开始实时识别...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ [yolov8_rtmp_detect] 无法读取视频帧，流可能已断开")
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 RTMP Stream Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ==================== 查询在线无人机视频流 ====================
def get_online_uav_stream(token, max_speed=100, min_speed=20):
    """
    查询当前在线无人机并获取其视频流数据，返回适用于HTTP流配置的字典。
    
    流程: get_token -> onlineuav_list -> flydata_list -> 提取5个关键字段
    
    Args:
        token (str): 访问令牌
        max_speed (int): 最高速度限制，默认100
        min_speed (int): 最低速度限制，默认20
    
    Returns:
        list[dict] | None: 成功时返回所有在线无人机视频流配置列表（适配Main_mp.py的HTTP流配置格式），
                           未查询到在线无人机或异常时返回None
    """
    if not token:
        print("❌ [get_online_uav_stream] Token is empty")
        return None
    
    try:
        # 1. Query online UAV list
        online_uav_result = onlineuav_list(token)
        if not online_uav_result or not online_uav_result.get("data"):
            print("❌ [get_online_uav_stream] No online UAVs found")
            return None
        
        online_uavs = online_uav_result["data"]
        print(f"📡 [get_online_uav_stream] Found {len(online_uavs)} online UAVs, fetching video streams...")
        
        stream_configs = []
        
        for uav in online_uavs:
            try:
                record_id = int(uav["recordId"])
                dept_id = int(uav["deptId"])
                uav_name = uav.get("uavName", "Unknown")
                
                # 2. Query real-time flight data for the UAV
                flydata = flydata_list(token, record_id, dept_id)
                if not flydata or not flydata.get("data") or len(flydata["data"]) == 0:
                    print(f"⚠️ [get_online_uav_stream] UAV '{uav_name}' has no real-time flight data, skipping")
                    continue
                
                # 3. Extract 5 key fields
                fly_info = flydata["data"][0]
                video_id = fly_info["recordId"]
                source_play_flv = uav["play_flv"]
                location = uav["uavName"]
                lat = fly_info["lat"]
                lng = fly_info["lng"]
                
                # 4. Assemble into HTTP stream configuration format for Main_mp.py
                stream_config = {
                    "id": str(video_id),
                    "source": source_play_flv,        # HTTP-FLV stream, low latency, directly supported by OpenCV
                    "use_stream": True,
                    "location": location,
                    "coordinates": {"lat": float(lat), "lng": float(lng)},
                    "max_speed": max_speed,
                    "min_speed": min_speed
                }
                
                stream_configs.append(stream_config)
                print(f"✅ [get_online_uav_stream] UAV '{uav_name}' video stream successfully obtained -> {source_play_flv}")
                
            except KeyError as ke:
                print(f"⚠️ [get_online_uav_stream] Missing UAV data field: {ke}, skipping")
                continue
            except Exception as e:
                print(f"⚠️ [get_online_uav_stream] Exception while processing UAV data: {e}, skipping")
                continue
        
        if not stream_configs:
            print("❌ [get_online_uav_stream] No valid video streams obtained from any online UAVs")
            return None
        
        print(f"✅ [get_online_uav_stream] Successfully obtained {len(stream_configs)} valid video stream configurations")
        return stream_configs
        
    except Exception as e:
        print(f"❌ [get_online_uav_stream] Exception: {str(e)}")
        return None

# ==================== 主函数 - 测试所有接口 ====================
if __name__ == "__main__":
    # # 1. 获取Token
    # token = get_token()
    # if not token:
    #     exit(1)
    # print("="*50)

    # # 2. 测试 飞机数据 接口
    # uavlist = uav_list(token, {"uavName": ""})  # 查询所有飞机
    # uav_detail(token, 184)  # 传入存在的uavId
    # print("="*50)

    # # 3. 测试 航线数据 接口
    # flyline_list(token, {"name": "", "lineType": 1})  # 查询线状航线
    # # flyline_import(token, "test.kmz", "测试航线")  # 需替换实际文件路径
    # # flyline_export(token, 758298618211895)  # 传入存在的flyLineId
    # # flyline_delete(token, 758298618211895, 4)  # 三维航线版本4
    # flyline_upload(token, "测试KML", 1, 4, '[{"height":10,"latitude":"32.04","longitude":"118.64"}]', "test.kml")
    # print("="*50)

    # # 4. 测试 飞行计划 接口
    # flyplan_list(token, {"planName": "", "state": 0})  # 查询待执行计划
    # # flyplan_add(token, {"planName": "测试计划", "uavId": 184, "state": 0})
    # # flyplan_update(token, {"id": 10867, "uavId": 184, "planName": "更新测试计划"})  # 传入存在的id
    # # flyplan_delete(token, 10867)  # 传入存在的recordId
    # flyplan_result(token, 6098)  # 传入存在的recordId
    # print("="*50)

    # # 5. 测试 飞行数据 接口
    # onlineuav_list(token)  # 查询在线飞机
    # flydata_list(token, 11208, 148)  # 传入存在的recordId和deptId
    # print("="*50)

    # # 6. 测试 历史数据 接口
    # flyhistory_list(token, 148, 184)  # 传入deptId和uavId
    # flyhistory_detail(token, 11263)  # 传入存在的recordId
    # flyhistory_export(token, 11263)  # 传入存在的recordId
    # print("="*50)

    # 7. 测试 YOLOv8 RTMP识别
    # yolov8_rtmp_detect(RTMP_STREAM_URL, YOLO_MODEL_PATH)

    # 视频流测试pipeline
    token = get_token()
    if not token:
        exit(1)
    # token = "eyJhbGciOiJIUzUxMiJ9.eyJ1c2VyX2lkIjo4MjAsInVzZXJfa2V5IjoiYWExYzViMzEtNzIyNS00MDdhLTk0YTItMmIzMWIwM2VjMjQ2IiwidXNlcm5hbWUiOiJjcSJ9.D687TP0o1jvZhH8Ro_uWT-mERqehk-aRlDARD6gcZZ7YPUn7u14ZUvxH5--ExTN7_FnVCDpPpFdtyD1e8ORnqg"
    print(f'token: {token}')
    print("="*50)

    uav_lists = uav_list(token)  # 查询所有飞机
    print("示例飞机数据:")
    print(uav_lists['data'][0])  # 打印第一条飞机数据
    print("="*50)

    # fly_plan_lists = flyplan_list(token, {"planName": "", "state": 0})  # 查询待执行计划
    # demo_plan = fly_plan_lists['data'][0]  # 获取第一条计划数据
    # print("示例飞行计划数据:")
    # print(demo_plan)
    # find_hls = False
    # print("查找含有视频流的飞行计划...")
    # for plan in (fly_plan_lists['data']):
    #     # print(f"检查飞行计划ID: {plan['id']}, Name: {plan['planName']}")
    #     tmp_result = flyplan_result(token, plan['id'])
    #     # print("飞行计划成果数据:")
    #     # print(tmp_result)
    #     if 'data' in tmp_result and 'hlsList' in tmp_result['data']:
    #         demo_plan = plan
    #         fly_plan_result_demo = tmp_result
    #         find_hls = True
    #         print(f"找到含有视频流的飞行计划，ID: {plan['id']}, Name: {plan['planName']}")
    #         print(plan)
    #         break
    #     else:
    #         continue
    # print("="*50)

    # if find_hls:
    #     fly_plan_result_demo = flyplan_result(token, demo_plan['id'])  # 传入存在的recordId
    #     print("示例飞行计划成果数据:")
    #     print(fly_plan_result_demo)  # 打印飞行计划成果数据
    # else:
    #     print("未找到包含视频流的飞行计划成果数据")
    # print("="*50)

    # 查询在线飞机列表
    online_uav_lists = onlineuav_list(token)
    print("在线飞机列表:")
    print(online_uav_lists)
    print("="*50)

    # {'msg': '操作成功', 'code': 200, 'data': [{'uavId': 10085, 'serialNumber': '1581F8HGX253B00A03LA', 'uavName': '重固-福泉山路-DJI3的无人机', 'deptId': 239, 'deptName': '重固镇', 'playUrl': 'webrtc://58.40.19.96:40080/index/api/webrtc?app=live_1&stream=10085&type=play&sign=9d65ca79708368123ffcb94470c01938', 'recordId': 14228, 'play_rtmp': 'rtmp://58.40.19.96:41935/live_1/10085?sign=9d65ca79708368123ffcb94470c01938', 'play_rtsp': 'rtsp://58.40.19.96:0/live_1/10085?sign=9d65ca79708368123ffcb94470c01938', 'play_flv': 'http://58.40.19.96:40080/live_1/10085.live.flv?sign=9d65ca79708368123ffcb94470c01938', 'play_hls': 'http://58.40.19.96:40080/live_1/10085/hls.m3u8?sign=9d65ca79708368123ffcb94470c01938', 'play_wsflv': 'ws://58.40.19.96:40080/live_1/10085.live.flv?sign=9d65ca79708368123ffcb94470c01938', 'play_wshls': 'ws://58.40.19.96:40080/live_1/10085/hls.m3u8?sign=9d65ca79708368123ffcb94470c01938', 'pushUrl': 'rtmp://58.40.19.96:41935/live_1/10085?sign=9d65ca79708368123ffcb94470c01938', 'push_rtc': 'webrtc://58.40.19.96:40080/index/api/webrtc?app=live_1&stream=10085&type=push&sign=9d65ca79708368123ffcb94470c01938', 'push_rtsp': 'rtsp://58.40.19.96:0/live_1/10085?sign=9d65ca79708368123ffcb94470c01938'}]}

    # # 飞行历史列表查询
    # fly_history_lists = flyhistory_list(token, demo_plan['deptId'], demo_plan['uavId'])

    if len(online_uav_lists["data"]) > 0:
        record_id, deptId = int(online_uav_lists["data"][0]["recordId"]), int(online_uav_lists["data"][0]["deptId"])
        print("deptId", deptId)
        print("recordId", record_id)
        print("="*50)

        flydata = flydata_list(token, record_id, deptId)  # 传入存在的recordId和deptId
        print("飞行实时数据:")
        print(flydata["data"][-1])  # 打印飞行实时数据
        print("="*50)

        video_id = flydata["data"][0]["recordId"]
        source_play_flv = online_uav_lists["data"][0]["play_flv"]
        location = online_uav_lists["data"][0]["uavName"]
        lat = flydata["data"][0]["lat"]
        lng = flydata["data"][0]["lng"]
        print(f"视频流ID: {video_id}")
        print(f"视频流URL: {source_play_flv}") 
        print(f"无人机: {location}, 坐标: ({lat}, {lng})")
        print("="*50)
        

        # print("飞行历史列表查询...")
        # # fly_history_list = flyhistory_list(token, dept_id=deptId, uav_id=uavId) # ["videoUrls"]
        # # print(f"加载记录: {fly_history_list}")
        # print("未能加载飞行历史列表")
    else:
        print("未查询到在线无人机")
        print("="*50)
    