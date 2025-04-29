import os
import sys

# Chuyển tới thư mục chứa manage.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Thiết lập biến môi trường cho Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "_smarthome.settings")

# Khởi chạy Django server
from django.core.management import execute_from_command_line
execute_from_command_line(["manage.py", "runserver", "127.0.0.1:8000", "--noreload"])

