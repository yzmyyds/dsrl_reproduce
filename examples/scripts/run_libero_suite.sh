#!/bin/bash

SUITE_NAME=${1:-"libero_spatial"}
GROUP_ID="${SUITE_NAME}_$(date +%m%d_%H%M)"
proj_name=DSRL_pi0_Libero

# 全局环境配置 (EGL渲染等)
export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl  
export OPENPI_DATA_HOME=./openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 环境配置函数
run_task() {
    local gpu_id=$1
    local task_id=$2
    
    # 检查 task_id 是否超过了 9 (因为 Suite 只有 10 个任务)
    if [ $task_id -gt 9 ]; then
        return
    fi

    echo "⚡ Launching Task $task_id on GPU $gpu_id..."

    # 设置特定任务的显卡 ID
    export CUDA_VISIBLE_DEVICES=$gpu_id
    export MUJOCO_EGL_DEVICE_ID=$gpu_id
    export EXP=./logs/${proj_name}/${GROUP_ID}/task_${task_id}

    python3 examples/launch_train_sim.py \
    --algorithm pixel_sac \
    --env libero \
    --suite_name "$SUITE_NAME" \
    --task_id $task_id \
    --launch_group_id "$GROUP_ID" \
    --prefix "gpu${gpu_id}_task_${task_id}" \
    --wandb_project "${proj_name}" \
    --batch_size 256 \
    --discount 0.999 \
    --seed 0 \
    --max_steps 500000 \
    --eval_interval 10000 \
    --log_interval 500 \
    --eval_episodes 10 \
    --multi_grad_step 20 \
    --start_online_updates 500 \
    --resize_image 64 \
    --action_magnitude 1.0 \
    --query_freq 20 \
    --hidden_dims 128 &
}

# 循环：每次处理 4 个任务 (2张卡 x 2任务)
for (( i=0; i<=9; i+=4 ))
do
    echo "=========================================================="
    echo "🚀 Parallel Batch Start: Tasks $i to $((i+3))"
    echo "=========================================================="

    run_task 1 $i
    sleep 15 # 给 JAX 留出编译和分配显存的时间
    run_task 1 $((i+1))
    
    sleep 15
    run_task 3 $((i+2))
    sleep 15
    run_task 3 $((i+3))

    echo "⏳ Waiting for this batch to finish before next 4 tasks..."
    wait # 等这四个任务跑完 500,000 步
done

echo "🎉 All tasks in $SUITE_NAME completed!"