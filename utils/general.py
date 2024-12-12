from datetime import datetime

def generate_timestamp():
    """Generate a timestamp in the format of 'YYYY-MM-DD_HH-MM-SS'"""
    return datetime.now().strftime("%Y-%m-%d %H-%M-%S")