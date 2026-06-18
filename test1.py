import logging
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify
from push_infos import push_images_info

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建 Flask 应用实例
app = Flask(__name__)

# 创建后台调度器
scheduler = BackgroundScheduler()

# 定义任务默认配置
DEFAULT_INTERVAL_SECONDS = 10
NO_IMAGE_INTERVAL_MINUTES = 5


# 定义定时任务函数
def scheduled_push_images_info():
    global job_id
    try:
        logger.info("开始执行定时任务：处理图片")
        with app.app_context():
            result = push_images_info()
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], dict) and result[0].get(
                    "error") == "No valid image data received":
                logger.info("未获取到有效图片数据，下次任务将在 5 分钟后执行。")
                # 调整下一次任务的执行间隔为 5 分钟
                scheduler.reschedule_job(job_id, trigger='interval', minutes=NO_IMAGE_INTERVAL_MINUTES)
            else:
                logger.info(f"定时任务执行结果: {result}")
                # 恢复为 10 秒的执行间隔
                scheduler.reschedule_job(job_id, trigger='interval', seconds=DEFAULT_INTERVAL_SECONDS)
    except Exception as e:
        logger.error(f"执行定时任务时发生错误: {e}")


if __name__ == "__main__":
    # 添加定时任务，初始为 10 秒执行一次，设置最大并发实例数为 1，合并错过的调度
    job = scheduler.add_job(scheduled_push_images_info, 'interval', seconds=DEFAULT_INTERVAL_SECONDS,
                            max_instances=1, coalesce=True)
    job_id = job.id

    # 立即执行一次定时任务
    scheduled_push_images_info()

    # 启动调度器
    scheduler.start()
    try:
        # 让程序保持运行
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        # 关闭调度器
        scheduler.shutdown()
