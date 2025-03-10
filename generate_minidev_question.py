import glob
import os
import json
import sqlite3

data_root = "/data/koushurui/Data/text2sql/minibird/MINIDEV/dev_databases"
def get_create_table_statements(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    schemas = ""
    
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    for table_name, create_sql in tables:
        schemas += create_sql+"\n"  # 直接输出 CREATE TABLE 语句

    conn.close()
    return schemas

def sqlite_to_mysql_type(sqlite_type):
    """转换 SQLite 数据类型为 MySQL 数据类型"""
    mapping = {
        "INTEGER": "INT",
        "TEXT": "VARCHAR(255)",
        "REAL": "FLOAT",
        "DATETIME": "DATETIME",
        "BLOB": "BLOB",
        "BOOLEAN": "TINYINT(1)"
    }
    return mapping.get(sqlite_type.upper(), sqlite_type)

def convert_sqlite_to_mysql(database_file):
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    # 获取所有表的名字
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    schemas = ""
    
    for table in tables:
        table_name = table[0]
        schemas += f"CREATE TABLE {table_name} (\n"

        # 获取表的列信息
        cursor.execute(f"PRAGMA table_info(\"{table_name}\");")
        columns = cursor.fetchall()
        column_definitions = []

        for column in columns:
            col_name = column[1]
            col_type = sqlite_to_mysql_type(column[2])
            col_notnull = "NOT NULL" if column[3] else ""
            col_default = f"DEFAULT {column[4]}" if column[4] else ""
            col_autoincrement = "AUTO_INCREMENT" if column[5] else ""
            column_definition = f"{col_name} {col_type} {col_notnull} {col_default} {col_autoincrement}".strip()
            column_definitions.append(column_definition)

        # 输出建表语句
        schemas += (",\n".join(column_definitions)+"\n")
        schemas += ");\n\n"
        print()

    conn.close()
    return schemas


def convert_sqlite_to_postgresql(db_file):
    # 连接 SQLite 数据库
    sqlite_conn = sqlite3.connect(db_file)
    cursor = sqlite_conn.cursor()

    # 获取所有表的名称
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    postgresql_schema = ""

    # 遍历每个表
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info(\"{table_name}\");")
        columns = cursor.fetchall()

        # 开始构建 PostgreSQL 表的 SQL
        postgresql_schema += f"CREATE TABLE {table_name} (\n"
        for column in columns:
            column_name = column[1]
            column_type = column[2]
            not_null = "NOT NULL" if column[3] else ""
            default_value = f"DEFAULT {column[4]}" if column[4] else ""
            
            # 对于 PostgreSQL，处理数据类型
            if column_type == "INTEGER":
                column_type = "INTEGER"
            elif column_type == "TEXT":
                column_type = "TEXT"
            elif column_type == "REAL":
                column_type = "REAL"
            elif column_type == "BLOB":
                column_type = "BYTEA"

            postgresql_schema += f"    {column_name} {column_type} {not_null} {default_value},\n"

        postgresql_schema = postgresql_schema.rstrip(",\n") + "\n);\n\n"

    # 打印 PostgreSQL 建表语句
    return postgresql_schema


if __name__ == "__main__":
    # sqlite
    sqlite_tables = {}
    for db_id in os.listdir(data_root):
        sqlite_tables[db_id] = get_create_table_statements(os.path.join(data_root, db_id, db_id+".sqlite"))
    # mysql
    mysql_tables = {}    
    for db_id in os.listdir(data_root):
        mysql_tables[db_id] = convert_sqlite_to_mysql(os.path.join(data_root, db_id, db_id+".sqlite"))
    # postgresql
    postgresql_tables = {}
    for db_id in os.listdir(data_root):
        postgresql_tables[db_id] = convert_sqlite_to_postgresql(os.path.join(data_root, db_id, db_id+".sqlite"))
        
    json_root = '/data/koushurui/Data/text2sql/minibird/MINIDEV'
    mini_dev_sqlite = json.load(open(os.path.join(json_root, "mini_dev_sqlite.json"), "r"))
    mini_dev_mysql = json.load(open(os.path.join(json_root, "mini_dev_mysql.json"), "r"))
    mini_dev_postsql = json.load(open(os.path.join(json_root, "mini_dev_postgresql.json"), "r"))
    
    
    def create_conversation(question):
        prompt = "/* Given the following database schema: */\n" + sqlite_tables[question["db_id"]] + "\n/* Answer the following:\n" + question["question"] + "\n" + question["evidence"] + "\*\nSELECT "
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": question["SQL"]}
            ]
        }
    
    def create_dataset(question):
        prompt = "/* Given the following database schema: */\n" + sqlite_tables[question["db_id"]] + "\n/* Answer the following:\n" + question["question"] + "\n" + question["evidence"] + "\*\nSELECT "
        return {
            "question_id": question["question_id"],
            "prompt": prompt,
            "SQL": question["SQL"],
            "db_id": question["db_id"],
            "difficulty": question["difficulty"],
            "evidence": question["evidence"]
        }
        
    
    sqlite_data = []
    for question in mini_dev_sqlite:
        sqlite_data.append(create_dataset(question))
    with open(os.path.join("/home/koushurui/Documents/Code/text2sql/DAIL-SQL/dataset/process", "MINIDEV", "sqlite_data.json"), "w") as f:
        json.dump(sqlite_data, f, indent=4)
        
        
    mysql_data = []
    for question in mini_dev_mysql:
        mysql_data.append(create_dataset(question))
    with open(os.path.join("/home/koushurui/Documents/Code/text2sql/DAIL-SQL/dataset/process", "MINIDEV", "mysql_data.json"), "w") as f:
        json.dump(mysql_data, f, indent=4)
        
        
    postgresql_data = []
    for question in mini_dev_postsql:
        postgresql_data.append(create_dataset(question))
    with open(os.path.join("/home/koushurui/Documents/Code/text2sql/DAIL-SQL/dataset/process", "MINIDEV", "postgresql_data.json"), "w") as f:
        json.dump(postgresql_data, f, indent=4)