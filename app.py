from dotenv import load_dotenv
import pyodbc
import datetime
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import os
import uuid
from urllib.parse import quote_plus, unquote
from utils import load_coupons, find_best_coupon

load_dotenv() 
db_server = os.getenv('DB_SERVER')
db_name = os.getenv('DB_NAME')
db_username = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_port = os.getenv('DB_PORT')
db_driver = os.getenv('DRIVER')

app = Flask(__name__)
CORS(app)

vectorizer, tfidf_matrix, coupons = load_coupons()

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

    return jsonify({"machineId": machine_id,
                    "createdAt": created_at}), 200


@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    content_type = data.get('contentType')
    content = data.get('content')   
    machine_id = data.get('machineId')
    chat_group_id = data.get('chatGroupId')
    sources = data.get('sources')
    string_sources = str(sources)
    chat_id = "CHAT_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    created_at = datetime.datetime.now()
    final_url = []
    try:
        with pyodbc.connect('DRIVER='+db_driver+';SERVER=tcp:'+db_server+';PORT='+db_port+';DATABASE='+db_name+';UID='+db_username+';PWD='+ db_password) as conn:
            with conn.cursor() as cursor:
                create_registry = """
                INSERT INTO chat_table (chat_id, content_type, content, created_at, machine_id, chat_group_id, sources)
                VALUES (?, ?, ?, ?, ?, ?, ?);"""
                cursor.execute(create_registry, (chat_id, content_type, content, created_at, machine_id, chat_group_id, string_sources,))
    except Exception as e:
        print("Error while reading data from SQL Server:", e)

    if content_type == "prompt" :
        # best_coupon = find_best_coupon(vectorizer, tfidf_matrix, coupons, content, 2)
        # print("Best matching coupon_group_id:", best_coupon)
        return jsonify({}), 200
    else:
        best_coupon = find_best_coupon(vectorizer, tfidf_matrix, coupons, content, 2)

        for coupon_val in best_coupon:
            coupon_code = coupon_val[0]
            product_url = coupon_val[1]
            new_url = f"{product_url}?coupon_code={coupon_code}&chat_id={chat_id}"
            encoded_url = quote_plus(new_url)
            updated_url = f"https://tradelogsai.eastus.cloudapp.azure.com/redirect?chatId={chat_id}&targetUrl={encoded_url}"
            final_url.append({
                "productUrl": updated_url
            })

        return jsonify(final_url), 200
    
@app.route('/redirect')
def redirect_to_target():
    encoded_target_url  = request.args.get('targetUrl')
    chat_id = request.args.get('chatId')

    target_url = unquote(encoded_target_url)
    return redirect(target_url)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port="8080")