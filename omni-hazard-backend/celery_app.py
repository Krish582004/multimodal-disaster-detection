from celery import Celery

# Initialize the Celery application to connect to your local Redis server
celery = Celery(
    "hazard_orchestrator",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# Autodiscover background tasks from the tasks.py file
celery.conf.imports = ('tasks',)