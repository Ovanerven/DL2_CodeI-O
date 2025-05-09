# This script is used to install the packages listed in the requirements.txt file
# We need rust for some of the packages. Install it from https://rustup.rs/ and then run
# set PATH=%PATH%;C:\Users\%USERNAME%\.cargo\bin before we install all the required packages.
import subprocess
import sys
import time

def install_package(package):
    print(f"Attempting to install {package}")
    try:
        # Run pip install command and capture output
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully installed {package}")
            return True
        else:
            print(f"❌ Failed to install {package}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception while installing {package}: {str(e)}")
        return False

def main():
    requirements_file = "Code_IO_aug/requirements.txt"
    
    # Track statistics
    successful = []
    failed = []
    
    try:
        with open(requirements_file, 'r') as file:
            packages = [line.strip() for line in file if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print(f"Error: Could not find the requirements file at {requirements_file}")
        return
    
    total_packages = len(packages)
    print(f"Found {total_packages} packages to install")
    
    start_time = time.time()
    
    for i, package in enumerate(packages, 1):
        print(f"\n[{i}/{total_packages}] Processing: {package}")
        
        if install_package(package):
            successful.append(package)
        else:
            failed.append(package)
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*50)
    print(f"Installation Summary ({elapsed_time:.1f} seconds)")
    print("="*50)
    print(f"Total packages: {total_packages}")
    print(f"Successfully installed: {len(successful)} ({len(successful)/total_packages*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/total_packages*100:.1f}%)")
    
    if failed:
        print("\nFailed packages:")
        for pkg in failed:
            print(f"  - {pkg}")
        
        # Save failed packages to a file
        with open("failed_packages.txt", "w") as f:
            for pkg in failed:
                f.write(f"{pkg}\n")
        print("\nList of failed packages saved to 'failed_packages.txt'")

if __name__ == "__main__":
    print("Starting package installation...")
    main()