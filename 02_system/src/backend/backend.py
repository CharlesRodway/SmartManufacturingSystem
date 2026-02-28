# Backend Service
# Digital Systems Project - Charles Rodway
#
# Consumes messages from all lathe RabbitMQ exchanges.
# Holds the latest state and reading history for every lathe in memory.
# Logs CRITICAL alerts to a persistent JSON file.
# Exposes a REST API that the dashboard reads from.
#
# Endpoints:
#   GET /health              - backend health check
#   GET /state               - full system state (latest reading per lathe)
#   GET /state/{lathe}       - single lathe latest state
#   GET /history/{lathe}     - last N readings for a lathe (for charts)
#   GET /alerts              - all CRITICAL alerts logged so far

import os
import json
import time
import threading
from collections import deque
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pika
import uvicorn


# settings

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "admin")
RABBITMQ_PASS = os.environ.get("RABBITMQ_PASS", "password")
CONFIG_PATH = "/app/config/lathes.json"
ALERTS_PATH = "/app/config/alerts.json"

# how many readings to keep in memory per lathe for the history charts
HISTORY_LENGTH = 500


# shared state - updated by consumer threads, read by API endpoints

system_state = {}
# deque automatically drops oldest readings once max length is reached
reading_history = {}
maintenance_alerts = []
state_lock = threading.Lock()


# clear alerts on startup - since we replay the same dataset each run,
# persisting old alerts would cause duplicates to accumulate over time

def load_alerts():
    global maintenance_alerts
    maintenance_alerts = []
    try:
        with open(ALERTS_PATH, 'w') as f:
            json.dump([], f)
    except Exception as e:
        print(f"Could not clear alerts file: {e}")


def save_alerts():
    try:
        with open(ALERTS_PATH, 'w') as f:
            json.dump(maintenance_alerts, f, indent=2)
    except Exception as e:
        print(f"Could not save alerts: {e}")


# fastapi app

app = FastAPI(title="CNC Predictive Maintenance Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/state")
def get_full_state():
    with state_lock:
        return dict(system_state)


@app.get("/state/{lathe_name}")
def get_lathe_state(lathe_name: str):
    with state_lock:
        if lathe_name not in system_state:
            raise HTTPException(status_code=404, detail=f"{lathe_name} not found")
        return system_state[lathe_name]


@app.get("/history/{lathe_name}")
def get_lathe_history(lathe_name: str, n: int = 200):
    # returns the last n readings for a lathe, used for the history charts
    with state_lock:
        if lathe_name not in reading_history:
            return []
        history = list(reading_history[lathe_name])
        return history[-n:] if len(history) > n else history


@app.get("/alerts")
def get_alerts():
    # returns all logged CRITICAL alerts, newest first
    with state_lock:
        return list(reversed(maintenance_alerts))


# rabbitmq message handler - called for every message received

def on_message(ch, method, properties, body):
    try:
        message = json.loads(body)
        lathe = message.get("lathe")
        if not lathe:
            return

        with state_lock:
            # update latest state
            system_state[lathe] = message

            # add to history for charts
            if lathe not in reading_history:
                reading_history[lathe] = deque(maxlen=HISTORY_LENGTH)

            # store a compact version of the reading for the history chart
            history_entry = {
                "reading": message.get("reading"),
                "timestamp": message.get("timestamp"),
                "machine_status": message.get("machine_status"),
                "bearings": {
                    b: {
                        "score": data.get("score"),
                        "status": data.get("status"),
                        "kurtosis": data.get("kurtosis"),
                        "rms": data.get("rms"),
                        "streak": data.get("streak")
                    }
                    for b, data in message.get("bearings", {}).items()
                }
            }
            reading_history[lathe].append(history_entry)

            # check for new CRITICAL alerts - log them if not already logged
            for bearing_name, bearing_data in message.get("bearings", {}).items():
                if bearing_data.get("latched") and bearing_data.get("streak", 0) >= 50:
                    # check if we already logged this alert
                    already_logged = any(
                        a["lathe"] == lathe and a["bearing"] == bearing_name
                        for a in maintenance_alerts
                    )
                    if not already_logged:
                        alert = {
                            "lathe": lathe,
                            "bearing": bearing_name,
                            "timestamp": message.get("timestamp"),
                            "reading": message.get("reading"),
                            "score": bearing_data.get("score"),
                            "kurtosis": bearing_data.get("kurtosis"),
                            "rms": bearing_data.get("rms"),
                            "resolved": False
                        }
                        maintenance_alerts.append(alert)
                        save_alerts()
                        print(f"ALERT logged: {lathe} {bearing_name} CRITICAL")

    except Exception as e:
        print(f"Error processing message: {e}")


# rabbitmq connection

def connect_rabbitmq():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=30
    )
    for attempt in range(10):
        try:
            connection = pika.BlockingConnection(parameters)
            return connection
        except Exception as e:
            print(f"RabbitMQ not ready ({attempt+1}/10)... {e}")
            time.sleep(5)
    raise RuntimeError("Could not connect to RabbitMQ")


def run_consumer_for_lathe(lathe_name):
    # each lathe gets its own connection and thread so they don't block each other
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        credentials=credentials,
        heartbeat=60,
        blocked_connection_timeout=30
    )
    while True:
        try:
            print(f"[{lathe_name}] Connecting to RabbitMQ...")
            connection = pika.BlockingConnection(parameters)
            channel = connection.channel()
            channel.exchange_declare(exchange=lathe_name, exchange_type='fanout', durable=True)
            result = channel.queue_declare(queue='', exclusive=True)
            queue_name = result.method.queue
            channel.queue_bind(exchange=lathe_name, queue=queue_name)
            channel.basic_consume(queue=queue_name, on_message_callback=on_message, auto_ack=True)
            print(f"[{lathe_name}] Connected. Consuming messages...")
            while True:
                connection.process_data_events(time_limit=0.1)
                time.sleep(0.05)
        except Exception as e:
            print(f"[{lathe_name}] Consumer error: {e}. Reconnecting in 5s...")
            time.sleep(5)


def start_consumers():
    print("Starting RabbitMQ consumers...")
    time.sleep(5)
    with open(CONFIG_PATH) as f:
        lathes_config = json.load(f)
    count = 0
    for lathe_name, config in lathes_config.items():
        if config.get("status") == "monitoring":
            t = threading.Thread(
                target=run_consumer_for_lathe,
                args=(lathe_name,),
                daemon=True,
                name=f"consumer-{lathe_name}"
            )
            t.start()
            count += 1
            print(f"  Started consumer thread for {lathe_name}")
    print(f"Started {count} consumer threads.")


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("CNC PREDICTIVE MAINTENANCE - BACKEND SERVICE")
    print(f"{'='*60}")
    print(f"RabbitMQ: {RABBITMQ_HOST}")
    print(f"Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    load_alerts()
    start_consumers()

    print("Starting API on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)