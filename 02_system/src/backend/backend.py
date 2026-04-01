# Backend Service
# Digital Systems Project - Charles Rodway
#
#consumes RabbitMQ messages from edge containers, holds system state and exposes REST API for the dashboard.


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
CONFIG_PATH      = "/app/config/lathes.json"
HYDRAULIC_CONFIG = "/app/config/hydraulics.json"
ALERTS_PATH      = "/app/config/alerts.json"

HISTORY_LENGTH = 500


# shared state

system_state      = {}
reading_history   = {}
hydraulic_state   = {}
hydraulic_history = {}
maintenance_alerts = []
state_lock = threading.Lock()


# alerts

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


# endpoints for bearings

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
    with state_lock:
        if lathe_name not in reading_history:
            return []
        history = list(reading_history[lathe_name])
        return history[-n:] if len(history) > n else history


@app.get("/alerts")
def get_alerts():
    with state_lock:
        return list(reversed(maintenance_alerts))


# endpooints for hydraulics

@app.get("/hydraulic/state")
def get_hydraulic_state():
    with state_lock:
        return dict(hydraulic_state)


@app.get("/hydraulic/state/{unit_name}")
def get_hydraulic_unit_state(unit_name: str):
    with state_lock:
        if unit_name not in hydraulic_state:
            raise HTTPException(status_code=404, detail=f"{unit_name} not found")
        return hydraulic_state[unit_name]


@app.get("/hydraulic/history/{unit_name}")
def get_hydraulic_history(unit_name: str, n: int = 100):
    with state_lock:
        if unit_name not in hydraulic_history:
            return []
        history = list(hydraulic_history[unit_name])
        return history[-n:] if len(history) > n else history


# bearing message handler

def on_bearing_message(ch, method, properties, body):
    try:
        message = json.loads(body)
        lathe = message.get("lathe")
        if not lathe:
            return

        with state_lock:
            system_state[lathe] = message

            if lathe not in reading_history:
                reading_history[lathe] = deque(maxlen=HISTORY_LENGTH)

            history_entry = {
                "reading":        message.get("reading"),
                "timestamp":      message.get("timestamp"),
                "machine_status": message.get("machine_status"),
                "bearings": {
                    b: {
                        "score":    data.get("score"),
                        "status":   data.get("status"),
                        "kurtosis": data.get("kurtosis"),
                        "rms":      data.get("rms"),
                        "streak":   data.get("streak")
                    }
                    for b, data in message.get("bearings", {}).items()
                }
            }
            reading_history[lathe].append(history_entry)

            # log CRITICAL alerts once per bearing per run
            for bearing_name, bearing_data in message.get("bearings", {}).items():
                if bearing_data.get("latched") and bearing_data.get("streak", 0) >= 50:
                    already_logged = any(
                        a["lathe"] == lathe and a["bearing"] == bearing_name
                        for a in maintenance_alerts
                    )
                    if not already_logged:
                        alert = {
                            "lathe":     lathe,
                            "bearing":   bearing_name,
                            "timestamp": message.get("timestamp"),
                            "reading":   message.get("reading"),
                            "score":     bearing_data.get("score"),
                            "kurtosis":  bearing_data.get("kurtosis"),
                            "rms":       bearing_data.get("rms"),
                            "resolved":  False
                        }
                        maintenance_alerts.append(alert)
                        save_alerts()
                        print(f"ALERT logged: {lathe} {bearing_name} CRITICAL")

    except Exception as e:
        print(f"Error processing bearing message: {e}")


# hydrailic message handler

def on_hydraulic_message(ch, method, properties, body):
    try:
        message = json.loads(body)
        unit = message.get("unit")
        if not unit:
            return

        with state_lock:
            hydraulic_state[unit] = message

            if unit not in hydraulic_history:
                hydraulic_history[unit] = deque(maxlen=HISTORY_LENGTH)

            # store compact history entry for trend charts
            history_entry = {
                "cycle":      message.get("cycle"),
                "timestamp":  message.get("timestamp"),
                "components": message.get("components", {})
            }
            hydraulic_history[unit].append(history_entry)

            # log CRITICAL hydraulic alerts once per component per run
            if message.get("unit_status") == "CRITICAL":
                for comp_name, comp_data in message.get("components", {}).items():
                    if comp_data.get("status") == "CRITICAL":
                        already_logged = any(
                            a.get("unit") == unit and a.get("component") == comp_name
                            for a in maintenance_alerts
                        )
                        if not already_logged:
                            alert = {
                                "type":      "hydraulic",
                                "unit":      unit,
                                "component": comp_name,
                                "label":     comp_data.get("label", "Unknown"),
                                "timestamp": message.get("timestamp"),
                                "cycle":     message.get("cycle"),
                                "resolved":  False
                            }
                            maintenance_alerts.append(alert)
                            save_alerts()
                            print(f"ALERT logged: {unit} {comp_name} CRITICAL ({comp_data.get('label')})")

    except Exception as e:
        print(f"Error processing hydraulic message: {e}")


# rabbitmq consumers

def run_consumer_for_lathe(lathe_name):
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters  = pika.ConnectionParameters(
        host=RABBITMQ_HOST, credentials=credentials,
        heartbeat=60, blocked_connection_timeout=30
    )
    while True:
        try:
            print(f"[{lathe_name}] Connecting to RabbitMQ...")
            connection = pika.BlockingConnection(parameters)
            channel    = connection.channel()
            channel.exchange_declare(exchange=lathe_name, exchange_type='fanout', durable=True)
            result     = channel.queue_declare(queue='', exclusive=True)
            channel.queue_bind(exchange=lathe_name, queue=result.method.queue)
            channel.basic_consume(
                queue=result.method.queue,
                on_message_callback=on_bearing_message,
                auto_ack=True
            )
            print(f"[{lathe_name}] Connected. Consuming messages...")
            while True:
                connection.process_data_events(time_limit=0.1)
                time.sleep(0.05)
        except Exception as e:
            print(f"[{lathe_name}] Consumer error: {e}. Reconnecting in 5s...")
            time.sleep(5)


def run_consumer_for_hydraulic(unit_name):
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    parameters  = pika.ConnectionParameters(
        host=RABBITMQ_HOST, credentials=credentials,
        heartbeat=60, blocked_connection_timeout=30
    )
    while True:
        try:
            print(f"[{unit_name}] Connecting to RabbitMQ...")
            connection = pika.BlockingConnection(parameters)
            channel    = connection.channel()
            channel.exchange_declare(exchange=unit_name, exchange_type='fanout', durable=True)
            result     = channel.queue_declare(queue='', exclusive=True)
            channel.queue_bind(exchange=unit_name, queue=result.method.queue)
            channel.basic_consume(
                queue=result.method.queue,
                on_message_callback=on_hydraulic_message,
                auto_ack=True
            )
            print(f"[{unit_name}] Connected. Consuming messages...")
            while True:
                connection.process_data_events(time_limit=0.1)
                time.sleep(0.05)
        except Exception as e:
            print(f"[{unit_name}] Consumer error: {e}. Reconnecting in 5s...")
            time.sleep(5)


def start_consumers():
    print("Starting RabbitMQ consumers...")
    time.sleep(5)

    # bearing consumers
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

    # hydraulic consumers
    try:
        with open(HYDRAULIC_CONFIG) as f:
            hydraulics_config = json.load(f)
        for unit_name, config in hydraulics_config.items():
            if config.get("status") == "monitoring":
                t = threading.Thread(
                    target=run_consumer_for_hydraulic,
                    args=(unit_name,),
                    daemon=True,
                    name=f"consumer-{unit_name}"
                )
                t.start()
                count += 1
                print(f"  Started consumer thread for {unit_name}")
    except FileNotFoundError:
        print("  No hydraulics.json found — skipping hydraulic consumers")

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
