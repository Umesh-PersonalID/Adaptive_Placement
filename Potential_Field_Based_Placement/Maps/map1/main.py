import os
import subprocess

# Create frame folder if it doesn't exist
os.makedirs("frames", exist_ok=True)

# Range of K_COVER values from 1.5 to 20 with step 0.5
k_values = [round(x * 0.5, 2) for x in range(100, 150)]  # 0.5 * 3 = 1.5 to 0.5 * 40 = 20.0



for k in k_values:
    print(f"\nRunning simulation for K_COVER = {k}...\n")
    
    # Call the simulation script with K_COVER as an argument
    result = subprocess.run(
        ["python", "Constrain_coverage_map1.py", str(k)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Log output for review
    log_file = f"log_K_{k}.txt"
    with open(log_file, "w") as f:
        f.write(result.stdout)
        f.write("\n==== STDERR ====\n")
        f.write(result.stderr)

    # Move last frame image to frames folder if it exists
    frame_name = f"last_frame_K_{k}.png"
    if os.path.exists("last_frame.png"):
        os.rename("last_frame.png", os.path.join("frames", frame_name))
        print(f"Saved last frame as {frame_name}\n")
    else:
        print(f"No frame found for K_COVER = {k}\n")