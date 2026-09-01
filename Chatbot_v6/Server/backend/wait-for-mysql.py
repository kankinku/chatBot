#!/usr/bin/env python3
"""MySQL 연결 대기 스크립트"""
import os
import time
import sys

try:
    import pymysql
except ImportError:
    import MySQLdb as pymysql

def required_env(name):
    value = os.environ.get(name, '').strip()
    if not value:
        print(f'{name} must be set')
        sys.exit(1)
    return value


MYSQL_HOST = required_env('MYSQL_HOST')
MYSQL_PORT = int(required_env('MYSQL_PORT'))
MYSQL_USER = required_env('MYSQL_USER')
MYSQL_PASSWORD = required_env('MYSQL_PASSWORD')
MYSQL_DATABASE = required_env('MYSQL_DATABASE')

for i in range(30):
    try:
        conn = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
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


