from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename

from ai_interview.tools.QuestionBank import QuestionBank

app = Flask(__name__)

# 配置文件上传相关设置
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'zip', 'rar', 'json'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 限制文件大小为16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
# 初始化问题库
question_bank = QuestionBank()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/rag')
def rag():
    return render_template('chat.html')


@app.route("/index/<int:id>", )
def index(id):
    if id == 1:
        return 'first'
    elif id == 2:
        return 'second'
    elif id == 3:
        return 'thrid'
    else:
        return 'hello world!'


@app.route('/upload', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400

        file = request.files['file']
        question_type = request.form.get('question_type', 'tech')  # 获取问题类型，默认为 tech

        # 检查文件名是否为空
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400

        # 检查文件类型
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件类型'}), 400

        try:
            # 确保上传目录存在
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            # 安全地获取文件名
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # 保存文件
            file.save(file_path)

            try:
                # 尝试加载文件到问题库
                question_bank.load_file_json(file_path, question_type=question_type)
                return jsonify({
                    'message': '文件上传并加载成功！',
                    'filename': filename,
                    'question_type': question_type
                }), 200
            except Exception as e:
                # 如果加载到问题库失败，删除已上传的文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({'error': f'文件加载到问题库失败：{str(e)}'}), 500

        except Exception as e:
            return jsonify({'error': f'文件上传失败：{str(e)}'}), 500

    # GET 请求返回上传页面
    return render_template('upload.html')


@app.route('/query', methods=['POST'])
def query():
    try:
        if not request.json or 'question' not in request.json:
            return jsonify({
                'status': 'error',
                'message': '请求格式错误，缺少问题参数'
            }), 400

        query = request.json['question']
        question_type = question_bank.judge_question_type(query)
        print(f"判断的问题类型: {question_type}")

        # 根据判断的类型进行搜索
        results = question_bank.search_questions(query, question_type, k=1)

        if not results:
            return jsonify({
                'status': 'not_found',
                'message': '未找到相关答案',
                'results': []
            }), 404

        # 格式化搜索结果
        formatted_results = []
        for result in results:
            formatted_results.append({
                'question': result['question'],
                'answer': result['answer'],
                'similarity_score': result['similarity_score'],
                'source': result['source'],
                'type': result['type']
            })

        print("搜索结果:", formatted_results)
        return jsonify({
            'status': 'success',
            'message': '成功找到相关答案',
            'question_type': question_type,
            'results': formatted_results
        }), 200

    except Exception as e:
        print(f"查询过程发生错误: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'查询过程发生错误：{str(e)}'
        }), 500


if __name__ == '__main__':
    app.run()
