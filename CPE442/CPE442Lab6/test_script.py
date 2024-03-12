import subprocess

# List of targets to build and run
targets = ["sobel_threaded", "sobel_threaded_opt", "sobel_threaded_O3"]

# Build each target
for target in targets:
    print(f"Building {target}...")
    subprocess.run(["make", target])

# Run each target and capture output
for target in targets:
    print(f"Running {target}...")
    result = subprocess.run(["./" + target], capture_output=True, text=True)
    print(result.stdout)  # Display standard output
    print(result.stderr)  # Display standard error

# Clean up object files and executables
subprocess.run(["make", "clean"])
