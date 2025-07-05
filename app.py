from dotenv import load_dotenv
import pyodbc
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid

load_dotenv() 
db_server = os.getenv('DB_SERVER')
db_name = os.getenv('DB_NAME')
db_username = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_port = os.getenv('DB_PORT')
db_driver = os.getenv('DRIVER')

app = Flask(__name__)
CORS(app)

@app.route('/register', methods=['POST'])
def register():
    machine_id = str(uuid.uuid4())
    created_at = datetime.datetime.now()
    try:
        with pyodbc.connect('DRIVER='+db_driver+';SERVER=tcp:'+db_server+';PORT='+db_port+';DATABASE='+db_name+';UID='+db_username+';PWD='+ db_password) as conn:
            with conn.cursor() as cursor:
                create_registry = """
                INSERT INTO machine (machine_id, created_at)
                VALUES (?, ?);"""
                cursor.execute(create_registry, (machine_id, created_at,))
    except Exception as e:
        print("Error while reading data from SQL Server:", e)
        return {"error": str(e)}, 500

    return jsonify({"machine_id": machine_id,
                    "created_at": created_at}), 200



@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    content_type = data.get('content_type')
    content = data.get('content')   
    machine_id = data.get('machine_id')
    chat_group_id = data.get('chat_group_id')
    sources = data.get('sources')

    coupon_code = "COUPON12345"
    return jsonify({
        "coupon_code": coupon_code}), 200


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="8080")