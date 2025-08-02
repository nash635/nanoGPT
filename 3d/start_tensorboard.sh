#!/bin/bash

# Start TensorBoard server for monitoring 3D parallel training
# Usage: bash start_tensorboard.sh [port]

PORT=${1:-6006}
LOG_DIR="runs/3d_parallel_training"

echo "Starting TensorBoard server..."
echo "Log directory: $LOG_DIR"
echo "Port: $PORT"
echo "Access TensorBoard at: http://localhost:$PORT"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Start TensorBoard
tensorboard --logdir="$LOG_DIR" --port="$PORT" --host=0.0.0.0
