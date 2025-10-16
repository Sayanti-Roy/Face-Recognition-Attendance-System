import mysql.connector
from mysql.connector import Error
import config

def initialize_database():
    """
    Connects to MySQL, creates the database and required tables if they don't exist.
    This function should be run once when the application starts.
    """
    try:
        # Step 1: Connect to MySQL Server to create the database
        conn_server = mysql.connector.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASS
        )
        if conn_server.is_connected():
            cursor = conn_server.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {config.DB_NAME}")
            print(f"Database '{config.DB_NAME}' is ready.")
            cursor.close()
            conn_server.close()
    except Error as e:
        print(f"Fatal Error: Could not connect to MySQL Server or create database.")
        print(f"Please check your config.py and ensure MySQL is running.")
        print(f"Details: {e}")
        exit() # Exit the app if we can't establish a DB connection

    try:
        # Step 2: Connect to the specific database to create the tables
        conn_db = mysql.connector.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASS,
            database=config.DB_NAME
        )
        if conn_db.is_connected():
            cursor = conn_db.cursor()
            
            # SQL to create 'users' table
            create_users_table = """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                role VARCHAR(100) DEFAULT 'Student',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB;
            """
            cursor.execute(create_users_table)
            
            # SQL to create 'attendance' table
            create_attendance_table = """
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            ) ENGINE=InnoDB;
            """
            cursor.execute(create_attendance_table)
            
            print("Tables 'users' and 'attendance' are ready.")
            cursor.close()
            conn_db.close()

    except Error as e:
        print(f"Fatal Error: Could not create tables.")
        print(f"Details: {e}")
        exit()
