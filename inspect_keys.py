
import safetensors.torch
import sys
import os

def inspect(path, outfile):
    outfile.write(f"--- Inspecting {os.path.basename(path)} ---\n")
    try:
        with safetensors.torch.safe_open(path, framework="pt", device="cpu") as f:
            keys = f.keys()
            outfile.write(f"Total keys: {len(keys)}\n")
            outfile.write("First 10 keys:\n")
            for k in list(keys)[:10]:
                outfile.write(f"  {k}\n")
    except Exception as e:
        outfile.write(f"Error reading {path}: {e}\n")
    outfile.write("\n")

if __name__ == "__main__":
    base_path = r"d:\USERFILES\ComfyUI\ComfyUI\custom_nodes\ComfyUI-nunchaku-unofficial-loader"
    files = [
        "bluePencil_clip_l.safetensors",
        "bluePencil_clip_g.safetensors",
        "bluePencilXL_v031_integrated.safetensors" 
    ]
    
    with open("keys_log.txt", "w") as f:
        for fname in files:
            inspect(os.path.join(base_path, fname), f)
