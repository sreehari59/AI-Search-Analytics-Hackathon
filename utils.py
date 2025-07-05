from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import pyodbc

load_dotenv() 
db_server = os.getenv('DB_SERVER')
db_name = os.getenv('DB_NAME')
db_username = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_port = os.getenv('DB_PORT')
db_driver = os.getenv('DRIVER')

def rows_to_dict_list(cursor):
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

def load_coupons():
    try:
        with pyodbc.connect('DRIVER='+db_driver+';SERVER=tcp:'+db_server+';PORT='+db_port+';DATABASE='+db_name+';UID='+db_username+';PWD='+ db_password) as conn:
            with conn.cursor() as cursor:
                get_coupons_query = """
                select * from coupon;"""
                cursor.execute(get_coupons_query)
                coupons = rows_to_dict_list(cursor)
    except Exception as e:
        print("Error while reading data from SQL Server:", e)
        return {"error": str(e)}, 500
    
    corpus = [c["coupon_keywords"] for c in coupons]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return vectorizer, tfidf_matrix, coupons 

def find_best_coupon(vectorizer, tfidf_matrix, coupons, user_query, top_n=2):
    query_vec = vectorizer.transform([user_query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return [(coupons[i]["coupon_group_id"], coupons[i]["product_url"]) for i in top_indices]