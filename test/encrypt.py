import time
import requests
from flask import Flask, request, jsonify
from encrypt_util import EncryptUtil  # 导入 EncryptUtil 类
from apscheduler.schedulers.blocking import BlockingScheduler

app = Flask(__name__)

# 密钥
key = "f0faa3dac9684f13921aefd14b385914"  # 32位十六进制字符串

# 测试接口 URL
test_url = "http://22.53.1.145:9101/sms2/getToken"
# 正式接口 URL
formal_url = "http://25.86.160.175:29101/sms2/getToken"
# 选择使用测试接口还是正式接口，这里默认使用测试接口
url = test_url

@app.route('/getToken111', methods=['POST'])
def my_task():
    print("执行任务")
@app.route('/getToken', methods=['POST'])
def getToken():
    try:
        # 获取请求中的 JSON 数据
        data = request.json

        # 获取登录名和密码
        login_name = data.get('loginName')
        password = data.get('password')

        if not login_name or not password:
            return jsonify({"error": "loginName and password are required"}), 400

        # 获取当前时间戳（毫秒）
        timestamp = str(int(time.time() * 1000))

        # 生成要加密的数据
        data_to_encrypt = f'{{"loginName":"{login_name}","password":"{password}","time":"{timestamp}"}}'

        # 加密
        encrypted = EncryptUtil.encrypt(key, data_to_encrypt)

        # 解密操作
        decrypted = EncryptUtil.decrypt(key, encrypted)
        print("Decrypted:", decrypted)

        # 构建请求体
        request_body = encrypted

        # 发送 POST 请求获取 token
        response = requests.post(url, data=request_body)

        # 检查响应状态码
        if response.status_code == 200:
            result = response.json()
            code = result.get("code")
            msg = result.get("msg")
            if code == 200:
                token = result.get("token")
                print(f"成功获取 token: {token}")
                return jsonify({"encrypted": encrypted, "Decrypted": decrypted, "token": token}), 200
            else:
                print(f"获取 token 失败，错误码: {code}，提示信息: {msg}")
                return jsonify({"encrypted": encrypted, "Decrypted": decrypted, "error": f"获取 token 失败，错误码: {code}，提示信息: {msg}"}), 500
        else:
            print(f"请求失败，状态码: {response.status_code}")
            return jsonify({"encrypted": encrypted, "Decrypted": decrypted, "error": f"请求失败，状态码: {response.status_code}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # 创建调度器
    scheduler = BlockingScheduler()
    # 添加任务，设置每 10 秒执行一次
    scheduler.add_job(my_task, 'interval', seconds=10)
    # 启动调度器
    scheduler.start()

    app.run(debug=True)