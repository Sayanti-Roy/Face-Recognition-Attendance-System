import mysql.connector
from mysql.connector import Error
from datetime import date
import config  # Import your configuration

def create_db_connection():
    """Creates and returns a database connection object."""
    try:
        connection = mysql.connector.connect(
            host=config.DB_HOST,
            user=config.DB_USER,
            password=config.DB_PASS,
            database=config.DB_NAME
        )
        # print("MySQL Database connection successful") # Commented out for cleaner logs
        return connection
    except Error as e:
        print(f"The error '{e}' occurred")
        return None

def get_or_create_user(connection, name):
    """Gets a user's ID by name, creating the user if they don't exist."""
    cursor = connection.cursor()
    
    query_select = "SELECT id FROM users WHERE name = %s"
    cursor.execute(query_select, (name,))
    result = cursor.fetchone()
    
    if result:
        user_id = result[0]
    else:
        query_insert = "INSERT INTO users (name, role) VALUES (%s, %s)"
        cursor.execute(query_insert, (name, 'Student')) 
        connection.commit()
        user_id = cursor.lastrowid
        print(f"New user '{name}' created with ID: {user_id}")
        
    return user_id

def mark_attendance(connection, name):
    """Marks attendance for a user, avoiding duplicate entries for the same day."""
    try:
        user_id = get_or_create_user(connection, name)
        cursor = connection.cursor()
        
        today = date.today()
        query_check = "SELECT id FROM attendance WHERE user_id = %s AND DATE(timestamp) = %s"
        cursor.execute(query_check, (user_id, today))
        
        if cursor.fetchone():
            return

        query_insert = "INSERT INTO attendance (user_id) VALUES (%s)"
        cursor.execute(query_insert, (user_id,))
        connection.commit()
        print(f"Successfully marked attendance for {name} ({today}).")
        
    except Error as e:
        print(f"The error '{e}' occurred")

def get_todays_attendance(connection):
    """Fetches all attendance records for the current day."""
    return get_attendance_for_date(connection, date.today())

def get_attendance_for_date(connection, specific_date):
    """Fetches all attendance records for a specific date."""
    cursor = connection.cursor()
    query = """
        SELECT u.name, a.timestamp
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        WHERE DATE(a.timestamp) = %s
        ORDER BY a.timestamp DESC
    """
    try:
        # The specific_date object needs to be passed in the tuple
        cursor.execute(query, (specific_date,))
        records = cursor.fetchall()
        return records
    except Error as e:
        print(f"The error '{e}' occurred")
        return []

def get_all_users(connection):
    """Fetches all users from the database."""
    cursor = connection.cursor(dictionary=True) # Use dictionary cursor
    query = "SELECT id, name, role FROM users ORDER BY name ASC"
    try:
        cursor.execute(query)
        users = cursor.fetchall()
        return users
    except Error as e:
        print(f"The error '{e}' occurred")
        return []

def delete_user(connection, user_id):
    """Deletes a user and their attendance records."""
    cursor = connection.cursor()
    try:
        # First, delete attendance records to maintain referential integrity
        cursor.execute("DELETE FROM attendance WHERE user_id = %s", (user_id,))
        # Then, delete the user
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        connection.commit()
        print(f"Successfully deleted user with ID {user_id}")
    except Error as e:
        print(f"The error '{e}' occurred when trying to delete user {user_id}")

