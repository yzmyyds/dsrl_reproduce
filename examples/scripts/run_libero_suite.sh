#!/bin/bash

SUITE_NAME=${1:-"libero_spatial"}
proj_name=DSRL_pi0_Libero

# 通过参数读取范围和 GPU
gpu_id=${2:-"0"}
START_ID=${3:-"0"}
END_ID=${4:-"0"}

if [ -n "$5" ]; then
    GROUP_ID="$5"
    echo "📌 Using manual Group ID: $GROUP_ID"
else
    GROUP_ID="${SUITE_NAME}_$(date +%m%d_%H%M)"
    echo "🕒 Using auto-generated Group ID: $GROUP_ID"
fi

# --- 关键修正：必须加 $ 符号 ---
export PYTHONPATH=$(pwd):$(pwd)/LIBERO:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$gpu_id
# 在 CUDA 隔离下，该卡在进程内逻辑索引为 0
export MUJOCO_EGL_DEVICE_ID=0

# 全局环境配置
export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export OPENPI_DATA_HOME=./openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false

run_task() {
    local task_id=$1
    export EXP=./logs/${proj_name}/${GROUP_ID}/task_${task_id}
    mkdir -p "$EXP"

    echo "🚀 [GPU $gpu_id] Launching Task $task_id"

    # 注意：JAX_VISIBLE_DEVICES=0 后面不要加反斜杠，直接启动 python
    PYTHONPATH=$(pwd):$(pwd)/LIBERO \
    CUDA_VISIBLE_DEVICES=$gpu_id \
    MUJOCO_EGL_DEVICE_ID=0 \
    JAX_VISIBLE_DEVICES=0 \
    python3 examples/launch_train_sim.py \
    --algorithm pixel_sac \
    --env libero \
    --suite_name "$SUITE_NAME" \
    --task_id "$task_id" \
    --launch_group_id "$GROUP_ID" \
    --prefix "gpu${gpu_id}_task_${task_id}" \
    --wandb_project "${proj_name}" \
    --batch_size 256 \
    --max_steps 500000 \
    --eval_interval 10000 \
    --multi_grad_step 20 \
    --resize_image 64 \
    --hidden_dims 128
}

# 循环顺序执行
for (( i=START_ID; i<=END_ID; i++ ))
do
    echo "=========================================================="
    echo "🚀 Starting Task $i on GPU $gpu_id"
    echo "=========================================================="
    
    run_task $i

done

echo "🎉 GPU $gpu_id tasks ($START_ID to $END_ID) completed!"