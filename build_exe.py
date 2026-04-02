import os
import subprocess
import shutil

def build():
    print("--------------------------------------------------")
    print("Cleaning up previous build folders...")
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"Removed {folder}")

    if os.path.exists('TumorClassificationApp.spec'):
        os.remove('TumorClassificationApp.spec')
        print("Removed specification file")
    print("Cleanup complete.")
    print("--------------------------------------------------")

    print("Starting FIX build process for Tumor Classification EXE...")
    print("Including comprehensive Keras hidden imports...")
    
    # Data files (source;dest)
    data_args = [
        "--add-data", "keras_model.h5;.",
        "--add-data", "labels.txt;.",
        "--add-data", "web_app;web_app",
    ]
    
    # Key fix: collect-all tf_keras and explicit hidden imports for core engines
    # Also including h5py and internal keras src
    hidden_imports = [
        "--collect-all", "tf_keras",
        "--hidden-import", "tf_keras.src.engine.base_layer_v1",
        "--hidden-import", "tf_keras.src.engine.control_flow_util",
        "--hidden-import", "tf_keras.src.saving.saving_lib",
        "--hidden-import", "h5py.defs",
        "--hidden-import", "h5py.utils",
        "--hidden-import", "h5py.h5ac",
        "--hidden-import", "h5py._proxy",
    ]
    
    cmd = [
        "py", "-3.12", "-m", "PyInstaller",
        "--onefile",
        "--name", "TumorClassificationApp",
        "app.py"
    ] + data_args + hidden_imports

    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("--------------------------------------------------")
        print("SUCCESS! Your FIXED EXE is in the 'dist' folder.")
        print("--------------------------------------------------")
    else:
        print("Build failed.")

if __name__ == "__main__":
    build()
