echo "Starting Experiments..."

# --- Hopper ---
echo "Running Hopper..."
python3 main.py --config configs/hopper/hopper_trajectory.yaml

# --- Mountain Car ---
echo "Running Mountain Car..."
python3 main.py --config configs/mountaincar/mountaincar_trajectory.yaml

# --- Mountain Car Continuous ---
echo "Running Mountain Car Continuous..."
python3 main.py --config configs/mountaincarcontinuous/mountaincar_continuous_trajectory.yaml

# --- Inverted Pendulum ---
echo "Running Inverted Pendulum..."
python3 main.py --config configs/invertedpendulum/invertedpendulum_trajectory.yaml

# --- Inverted Double Pendulum ---
echo "Running Inverted Double Pendulum..."
python3 main.py --config configs/inverted_double_pendulum/inverteddoublependulum_trajectory.yaml

# --- Walker2d ---
echo "Running Walker2d..."
python3 main.py --config configs/walker2d/walker2d_trajectory.yaml

# --- Reacher ---
echo "Running Reacher..."
python3 main.py --config configs/reacher/reacher_trajectory.yaml

# --- Swimmer ---
echo "Running Swimmer..."
python3 main.py --config configs/swimmer/swimmer_trajectory.yaml

# --- Pong ---
echo "Running Pong..."
python3 main.py --config configs/pong/pong_trajectory.yaml

# --- Nav ---
echo "Running Nav..."
python3 main.py --config configs/nav/nav_trajectory.yaml

echo "All experiments completed."