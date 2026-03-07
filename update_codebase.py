import os
import re

codebase_file = "src/utils/codebase.ts"

with open(codebase_file, "r") as f:
    content = f.read()

# Map of variable names to file paths
file_map = {
    "install_sh": "agentq_upgrade_v2.1/install.sh",
    "predictive_anomaly_py": "agentq_upgrade_v2.1/modules/predictive_anomaly.py",
    "contextual_reasoning_py": "agentq_upgrade_v2.1/modules/contextual_reasoning.py",
    "agentq_config_patch_json": "agentq_upgrade_v2.1/config/agentq_config_patch.json",
    "rollback_sh": "agentq_upgrade_v2.1/rollback.sh",
    "cmake_lists_txt": "src/ai-core/cpp/CMakeLists.txt",
    "qcos_inference_stub_cpp": "src/ai-core/cpp/qcos_inference_stub.cpp",
    "bridge_server_py": "src/ai-core/system/bridge_server.py",
    "hybrid_model_py": "src/ai-core/scripts/hybrid_model.py",
    "clnn_qnn_py": "src/ai-core/models/clnn_qnn.py",
    "ipsnn_qnn_py": "src/ai-core/models/ipsnn_qnn.py",
    "os_kernel_net_py": "src/ai-core/models/os_kernel_net.py",
    "iai_ips_qnn_py": "src/ai-core/models/iai_ips_qnn.py",
    "instinct_synthesis_py": "src/ai-core/training/instinct_synthesis.py",
    "training_setup_py": "src/ai-core/training/training_setup.py",
    "save_weights_py": "src/ai-core/training/save_weights.py",
    "qllm_core_py": "qllm/qllm_core.py",
    "qllm_ts": "qllm/QLLM.ts"
}

for var_name, file_path in file_map.items():
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            file_content = f.read()
            
        # Escape backticks and ${}
        file_content = file_content.replace('`', '\\`').replace('${', '\\${')
        
        # Replace the entire const block
        pattern = r"const " + var_name + r" = `[\s\S]*?`;"
        replacement = f"const {var_name} = `\n{file_content}\n`;"
        content = re.sub(pattern, replacement, content)
    else:
        print(f"File not found: {file_path}")

with open(codebase_file, "w") as f:
    f.write(content)

print("codebase.ts updated successfully.")
