import os
import sqlite3
import mysql.connector
from mysql.connector import Error as MySQLError
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    SQLITE_PATH = "/tmp/mobile_price_db.sqlite"
else:
    SQLITE_PATH = os.path.join(Config.BASE_DIR, "mobile_price_db.sqlite")

def get_mysql_connection(with_db=True):
    """Attempt connecting to MySQL Server."""
    try:
        kwargs = {
            "host": Config.MYSQL_HOST,
            "port": Config.MYSQL_PORT,
            "user": Config.MYSQL_USER,
            "password": Config.MYSQL_PASSWORD,
            "connection_timeout": 3
        }
        if with_db:
            kwargs["database"] = Config.MYSQL_DATABASE
        conn = mysql.connector.connect(**kwargs)
        return conn
    except MySQLError as e:
        return None

def get_sqlite_connection():
    """Fallback local database connection."""
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def is_mysql_available():
    """Checks if MySQL connection succeeds with current credentials."""
    conn = get_mysql_connection(with_db=False)
    if conn and conn.is_connected():
        conn.close()
        return True
    return False

def init_db():
    """
    Initializes database schema.
    Tries MySQL first. If credentials are not yet configured or server is inaccessible,
    initializes local SQLite fallback so the system is fully operational.
    """
    mysql_ok = False
    error_msg = ""

    # Attempt MySQL initialization
    try:
        conn = get_mysql_connection(with_db=False)
        if conn and conn.is_connected():
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{Config.MYSQL_DATABASE}`;")
            cursor.close()
            conn.close()

            db_conn = get_mysql_connection(with_db=True)
            if db_conn and db_conn.is_connected():
                cursor = db_conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ram_gb INT NOT NULL,
                    storage_gb INT NOT NULL,
                    battery_mah INT NOT NULL,
                    primary_camera_mp INT NOT NULL,
                    front_camera_mp INT NOT NULL,
                    cpu_speed_ghz FLOAT NOT NULL,
                    cpu_cores INT NOT NULL,
                    screen_size_inch FLOAT NOT NULL,
                    has_5g BOOLEAN NOT NULL DEFAULT 0,
                    predicted_price DECIMAL(10, 2) NOT NULL,
                    price_tier VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                db_conn.commit()
                cursor.close()
                db_conn.close()
                mysql_ok = True
                logger.info("Successfully connected to MySQL and initialized tables.")
    except Exception as e:
        error_msg = str(e)
        logger.warning(f"MySQL connection attempt: {e}")

    # Initialize SQLite fallback database
    try:
        sq_conn = get_sqlite_connection()
        sq_cursor = sq_conn.cursor()
        sq_cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ram_gb INTEGER NOT NULL,
            storage_gb INTEGER NOT NULL,
            battery_mah INTEGER NOT NULL,
            primary_camera_mp INTEGER NOT NULL,
            front_camera_mp INTEGER NOT NULL,
            cpu_speed_ghz REAL NOT NULL,
            cpu_cores INTEGER NOT NULL,
            screen_size_inch REAL NOT NULL,
            has_5g INTEGER NOT NULL DEFAULT 0,
            predicted_price REAL NOT NULL,
            price_tier TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """)
        sq_conn.commit()
        sq_conn.close()
    except Exception as e:
        logger.error(f"Failed to initialize SQLite fallback: {e}")

    if mysql_ok:
        return True, "MySQL database connected and initialized."
    else:
        return False, "MySQL credentials needed in .env. SQLite fallback active."

def save_prediction(spec_data, predicted_price, price_tier):
    """
    Saves prediction to MySQL if available, or SQLite fallback.
    """
    # 1. Try MySQL
    mysql_conn = get_mysql_connection(with_db=True)
    if mysql_conn:
        try:
            cursor = mysql_conn.cursor()
            query = """
            INSERT INTO predictions (
                ram_gb, storage_gb, battery_mah, primary_camera_mp,
                front_camera_mp, cpu_speed_ghz, cpu_cores, screen_size_inch,
                has_5g, predicted_price, price_tier
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            values = (
                int(spec_data["ram_gb"]),
                int(spec_data["storage_gb"]),
                int(spec_data["battery_mah"]),
                int(spec_data["primary_camera_mp"]),
                int(spec_data["front_camera_mp"]),
                float(spec_data["cpu_speed_ghz"]),
                int(spec_data["cpu_cores"]),
                float(spec_data["screen_size_inch"]),
                1 if spec_data.get("has_5g") in [True, 1, "1", "true"] else 0,
                float(predicted_price),
                price_tier
            )
            cursor.execute(query, values)
            mysql_conn.commit()
            record_id = cursor.lastrowid
            cursor.close()
            mysql_conn.close()
            return True, f"Saved to MySQL [ID: #{record_id}]"
        except Exception as e:
            logger.error(f"MySQL insert error: {e}")

    # 2. Fallback to SQLite
    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        query = """
        INSERT INTO predictions (
            ram_gb, storage_gb, battery_mah, primary_camera_mp,
            front_camera_mp, cpu_speed_ghz, cpu_cores, screen_size_inch,
            has_5g, predicted_price, price_tier
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        values = (
            int(spec_data["ram_gb"]),
            int(spec_data["storage_gb"]),
            int(spec_data["battery_mah"]),
            int(spec_data["primary_camera_mp"]),
            int(spec_data["front_camera_mp"]),
            float(spec_data["cpu_speed_ghz"]),
            int(spec_data["cpu_cores"]),
            float(spec_data["screen_size_inch"]),
            1 if spec_data.get("has_5g") in [True, 1, "1", "true"] else 0,
            float(predicted_price),
            price_tier
        )
        cursor.execute(query, values)
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        return True, f"Saved to Fallback Storage [ID: #{record_id}]"
    except Exception as e:
        logger.error(f"Fallback insert error: {e}")
        return False, str(e)

def get_all_predictions(limit=50):
    """
    Retrieves prediction history from MySQL if connected, otherwise from SQLite.
    """
    # 1. Try MySQL
    mysql_conn = get_mysql_connection(with_db=True)
    if mysql_conn:
        try:
            cursor = mysql_conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT %s;", (limit,))
            records = cursor.fetchall()
            cursor.close()
            mysql_conn.close()
            for r in records:
                if "predicted_price" in r:
                    r["predicted_price"] = float(r["predicted_price"])
                if "created_at" in r and r["created_at"]:
                    r["created_at"] = r["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            return records
        except Exception as e:
            logger.error(f"MySQL select error: {e}")

    # 2. Try SQLite
    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?;", (limit,))
        rows = cursor.fetchall()
        records = [dict(row) for row in rows]
        conn.close()
        for r in records:
            if "predicted_price" in r:
                r["predicted_price"] = float(r["predicted_price"])
        return records
    except Exception as e:
        logger.error(f"Fallback select error: {e}")
        return []

def delete_prediction(record_id):
    """
    Deletes a prediction from MySQL and SQLite fallback.
    """
    deleted = False
    # MySQL
    mysql_conn = get_mysql_connection(with_db=True)
    if mysql_conn:
        try:
            cursor = mysql_conn.cursor()
            cursor.execute("DELETE FROM predictions WHERE id = %s;", (record_id,))
            mysql_conn.commit()
            if cursor.rowcount > 0:
                deleted = True
            cursor.close()
            mysql_conn.close()
        except Exception as e:
            logger.error(f"MySQL delete error: {e}")

    # SQLite
    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = ?;", (record_id,))
        conn.commit()
        if cursor.rowcount > 0:
            deleted = True
        conn.close()
    except Exception as e:
        logger.error(f"SQLite delete error: {e}")

    return deleted, "Record deleted." if deleted else "Record not found."

def test_connection():
    """
    Returns connection status and human-readable mode info.
    """
    if is_mysql_available():
        return True, "MySQL Server Connected"
    return False, "MySQL pending credentials (using local fallback)"
