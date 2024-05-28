# gunicorn.conf.py

# Number of worker processes
workers = 2

# Worker timeout
timeout = 120

# Log level
loglevel = 'info'

# Bind to port
bind = '0.0.0.0:{}'.format(os.environ.get('PORT', 5000))
