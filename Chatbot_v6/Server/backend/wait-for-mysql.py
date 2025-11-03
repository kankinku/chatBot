#!/usr/bin/env python3
"""MySQL 연결 대기 스크립트"""
import time
import sys

try:
    import pymysql
except ImportError:
    import MySQLdb as pymysql

for i in range(30):
    try:
        conn = pymysql.connect(
            host='mysql',
            user='chatbot_user',
            password='1234',
            database='chatbot_db',
            connect_timeout=5
        )
        conn.close()
        print('MySQL is ready!')
        sys.exit(0)
    except Exception as e:
        print(f'MySQL is not ready yet. Waiting... ({i+1}/30)')
        time.sleep(2)

print('MySQL connection failed after 60 seconds')
sys.exit(1)


